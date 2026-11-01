#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Detector proposals are only an allow-list for UiMaskCS, never the final
// protection mask. Coordinates are normalised, half-open source-image bounds.
struct UiRegion {
    float left, top, right, bottom, confidence;
};

constexpr size_t kMaxUiRegions = 32;
constexpr uint32_t kUiDetectSize = 640;
constexpr size_t kUiDetectProposals = 8400;
constexpr int kUiDetectIntervalMs = 200;
constexpr int kUiRegionsLifetimeMs = 600;
constexpr float kUiRegionAcquireConfidence = 0.10f;
constexpr float kUiRegionContinueConfidence = 0.075f;
constexpr float kUiRegionContinueIou = 0.60f;

// Pair-owned diagnostics: exactly the allow-list used to build that pair's
// native mask, never the worker's newer result. Fixed storage avoids a heap
// allocation each time the presentation thread copies a flow-ring entry.
struct UiRegionDebug {
    bool maskEnabled = false;
    bool detectorEnabled = false;
    bool valid = false;
    uint32_t count = 0;
    std::array<UiRegion, kMaxUiRegions> regions{};
};

inline bool uiRegionsFresh(uint64_t sourceId, uint64_t currentId, double ageMs) {
    return sourceId <= currentId && ageMs >= 0.0 && ageMs <= kUiRegionsLifetimeMs;
}

inline bool uiRegionsCanContinue(uint64_t sourceId, uint64_t currentId, double ageMs) {
    return sourceId < currentId && uiRegionsFresh(sourceId, currentId, ageMs);
}

inline float uiRegionIou(const UiRegion& a, const UiRegion& b) {
    const float intersection = std::max(0.0f, std::min(a.right, b.right) - std::max(a.left, b.left)) *
                               std::max(0.0f, std::min(a.bottom, b.bottom) - std::max(a.top, b.top));
    const float total = (a.right - a.left) * (a.bottom - a.top) +
                        (b.right - b.left) * (b.bottom - b.top) - intersection;
    return total > 0 && std::isfinite(total) ? intersection / total : 0.0f;
}

// YOLOv8 single-class output [1,5,N]: cx,cy,w,h (640px canvas), probability.
// Inverse of the centred letterbox in UiDetectInputCS, including integer rounding.
// Optional history changes admission/ranking, NEVER supplies output geometry:
// new icons still need 0.10; a current proposal >=0.075 with >=0.60 IoU to a
// previous icon may continue. Supported existing icons get capacity priority
// instead of oscillating out of the 32-box list as unrelated scores change.
// The caller must discard history on expiry, invalidation or out-of-order jobs.
inline std::vector<UiRegion> decodeUiRegions(const float* data, size_t n, uint32_t width,
                                            uint32_t height, float threshold = kUiRegionAcquireConfidence,
                                            const std::vector<UiRegion>& previous = {}) {
    if (!data || !n || !width || !height) return {};
    const float scale = std::min(float(kUiDetectSize) / width, float(kUiDetectSize) / height);
    const float rw = std::max(1.0f, std::floor(width * scale + 0.5f));
    const float rh = std::max(1.0f, std::floor(height * scale + 0.5f));
    const float px = std::floor((kUiDetectSize - rw) * 0.5f);
    const float py = std::floor((kUiDetectSize - rh) * 0.5f);
    auto padded = [&](UiRegion r) {
        r.left = std::max(0.0f, r.left - 2.0f / rw);
        r.top = std::max(0.0f, r.top - 2.0f / rh);
        r.right = std::min(1.0f, r.right + 2.0f / rw);
        r.bottom = std::min(1.0f, r.bottom + 2.0f / rh);
        return r;
    };
    const float minimum = previous.empty() ? threshold : std::min(threshold, kUiRegionContinueConfidence);
    struct Candidate { UiRegion region; bool continuing; };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < n; ++i) {
        const float x = data[i], y = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
        const float score = data[4 * n + i];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h) ||
            !std::isfinite(score) || score < minimum || score > 1.0f || w <= 0 || h <= 0) continue;
        UiRegion r{std::clamp((x - w * 0.5f - px) / rw, 0.0f, 1.0f),
                   std::clamp((y - h * 0.5f - py) / rh, 0.0f, 1.0f),
                   std::clamp((x + w * 0.5f - px) / rw, 0.0f, 1.0f),
                   std::clamp((y + h * 0.5f - py) / rh, 0.0f, 1.0f), score};
        const float area = (r.right - r.left) * (r.bottom - r.top);
        // Reject padding-only/degenerate boxes and whole-scene detections.
        // Large menus may be missed: safer than authorising half the game view.
        if (!(area > 0.0f && area <= 0.5f)) continue;
        const UiRegion bounds = padded(r);
        bool continuing = false;
        for (size_t j = 0; j < std::min(previous.size(), kMaxUiRegions); ++j) {
            if (uiRegionIou(bounds, previous[j]) >= kUiRegionContinueIou) { continuing = true; break; }
        }
        if (score >= threshold || continuing) candidates.push_back({r, continuing});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.continuing != b.continuing) return a.continuing;
        return a.region.confidence > b.region.confidence;
    });
    std::vector<UiRegion> kept;
    std::array<bool, kMaxUiRegions> matched{};
    for (const Candidate& candidate : candidates) {
        const UiRegion& r = candidate.region;
        size_t match = kMaxUiRegions;
        if (candidate.continuing) {
            const UiRegion bounds = padded(r);
            float overlap = kUiRegionContinueIou;
            for (size_t j = 0; j < std::min(previous.size(), kMaxUiRegions); ++j) {
                const float iou = uiRegionIou(bounds, previous[j]);
                if (!matched[j] && iou >= overlap) { match = j; overlap = iou; }
            }
        }
        // A single earlier icon cannot spawn multiple weak continuations.
        if (r.confidence < threshold && match == kMaxUiRegions) continue;
        bool duplicate = false;
        for (const UiRegion& k : kept) {
            if (uiRegionIou(r, k) > 0.5f) { duplicate = true; break; }
        }
        if (!duplicate) {
            kept.push_back(r);
            if (match != kMaxUiRegions) matched[match] = true;
        }
        if (kept.size() == kMaxUiRegions) break;
    }
    // Two detector pixels of padding cover antialiased edges, not whole panels.
    for (UiRegion& r : kept) r = padded(r);
    return kept;
}
