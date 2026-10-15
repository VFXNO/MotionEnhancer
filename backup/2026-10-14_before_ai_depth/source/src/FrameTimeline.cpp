#include "FrameTimeline.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr int64_t kSecond = 10000000;
constexpr int64_t kMs = 10000;
// Fraction of the source period within which an arrival counts as "on the
// cadence grid" (or, measured from the previous accepted frame, as an
// update to that same frame).
constexpr double kGridTolerance = 0.3;
// A frame held for deferral is considered real (accepted) unless a newer
// frame supersedes it within this fraction of a period.
constexpr double kDeferWindow = 0.7;
constexpr size_t kCadenceWindow = 16;
constexpr size_t kCadenceMinSamples = 8;
}

void FrameTimeline::configure(uint32_t sourceFpsOverride, double refreshHz, uint32_t outputMultiplier) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sourceFpsOverride = sourceFpsOverride;
    m_outputMultiplier = outputMultiplier;
    m_refreshInterval = static_cast<int64_t>(static_cast<double>(kSecond) / std::max(1.0, refreshHz) + 0.5);
    if (sourceFpsOverride > 0) {
        m_period = kSecond / static_cast<int64_t>(sourceFpsOverride);
        m_locked = true;
    }
    // Latency budget: the next source frame (one period) must have arrived
    // and had its flow computed (a couple of refreshes) before media time
    // reaches the previous one. Starvation grows it; sustained slack
    // shrinks it back toward the minimum.
    m_minLatencyBudget = m_period + m_refreshInterval + 4 * kMs;
    m_maxLatencyBudget = 3 * m_period + 60 * kMs;
    m_latencyBudget = m_period + 2 * m_refreshInterval + 10 * kMs;
}

// ---------------------------------------------------------------- cadence

void FrameTimeline::observeInterval(int64_t interval) {
    if (interval <= 0 || interval > kSecond) return;
    m_intervals.push_back(interval);
    while (m_intervals.size() > kCadenceWindow) m_intervals.pop_front();
    updateCadence();
}

void FrameTimeline::updateCadence() {
    if (m_sourceFpsOverride > 0) return;
    if (m_intervals.size() < kCadenceMinSamples) {
        m_locked = false;
        return;
    }
    std::vector<int64_t> sorted(m_intervals.begin(), m_intervals.end());
    std::sort(sorted.begin(), sorted.end());
    // Mean of the middle half: robust against UI ticks and against the
    // alternating short/long intervals of a 24 fps video repainted at 60 Hz.
    const size_t lo = sorted.size() / 4;
    const size_t hi = sorted.size() - lo;
    double sum = 0.0;
    for (size_t i = lo; i < hi; ++i) sum += static_cast<double>(sorted[i]);
    const int64_t period = static_cast<int64_t>(sum / static_cast<double>(hi - lo) + 0.5);
    size_t near = 0;
    for (int64_t v : sorted) {
        if (std::llabs(v - period) <= period / 4) ++near;
    }
    const bool wasLocked = m_locked;
    const int64_t oldPeriod = m_period;
    m_locked = near * 4 >= sorted.size() * 3;
    if (m_locked) {
        m_period = period;
        const bool changed = std::llabs(m_period - oldPeriod) > oldPeriod / 5;
        if (!wasLocked || changed) {
            m_minLatencyBudget = m_period + m_refreshInterval + 4 * kMs;
            m_maxLatencyBudget = 3 * m_period + 60 * kMs;
            if (!wasLocked) {
                m_latencyBudget = m_period + 2 * m_refreshInterval + 10 * kMs;
            }
            m_latencyBudget = std::clamp(m_latencyBudget, m_minLatencyBudget, m_maxLatencyBudget);
        }
    }
}

bool FrameTimeline::onGrid(int64_t delta) const {
    if (delta <= 0) return false;
    const double k = std::round(static_cast<double>(delta) / static_cast<double>(m_period));
    if (k < 1.0) return false;
    const int64_t predicted = static_cast<int64_t>(k * static_cast<double>(m_period));
    return std::llabs(delta - predicted) <= static_cast<int64_t>(kGridTolerance * m_period);
}

int64_t FrameTimeline::regularize(int64_t tsRaw) {
    if (m_frames.empty() || !m_locked) return tsRaw;
    const int64_t delta = tsRaw - m_lastAcceptedTsRaw;
    const double k = std::max(1.0, std::round(static_cast<double>(delta) / static_cast<double>(m_period)));
    const int64_t predicted = m_lastAcceptedTs + static_cast<int64_t>(k * static_cast<double>(m_period));
    const int64_t err = tsRaw - predicted;
    int64_t ts;
    if (std::llabs(err) > static_cast<int64_t>(kGridTolerance * m_period)) {
        ts = tsRaw;                 // phase shift: resync
    } else {
        ts = predicted + err / 10;  // slow phase correction
    }
    return std::max(ts, m_frames.back().ts + m_period / 10);
}

// --------------------------------------------------------------- producer

FrameTimeline::IngestPlan FrameTimeline::onFrameArrived(int64_t tsRaw, int64_t arrival) {
    std::lock_guard<std::mutex> lock(m_mutex);
    IngestPlan plan;
    if (m_lastArrivalTsRaw > 0) observeInterval(tsRaw - m_lastArrivalTsRaw);
    m_lastArrivalTsRaw = tsRaw;

    int64_t lastRef = m_lastAcceptedTsRaw;
    bool haveLast = !m_frames.empty();
    if (m_deferred) {
        // A later frame superseded the held one unless enough time passed
        // for the held frame to have been a real source frame (phase shift).
        if (tsRaw - m_deferredTsRaw >= static_cast<int64_t>(kDeferWindow * m_period)) {
            plan.acceptDeferredFirst = true;
            lastRef = m_deferredTsRaw;
            haveLast = true;
        } else {
            plan.dropDeferred = true;
            ++m_stats.dropped;
        }
        m_deferred = false;
    }

    if (!haveLast || !m_locked) {
        plan.acceptThis = true;
        return plan;
    }

    const int64_t delta = tsRaw - lastRef;
    const int64_t tolerance = static_cast<int64_t>(kGridTolerance * m_period);
    if (delta < tolerance) {
        // Two frames far closer than the cadence: the later one is a newer
        // version of the same source frame (a UI tick landing just after
        // or just before the video update). Replace unless that frame is
        // already on screen.
        if (!plan.acceptDeferredFirst && m_frames.back().index > m_currentF1Index) {
            plan.replaceLast = true;
            ++m_stats.replaced;
        }
        plan.acceptThis = true;
        return plan;
    }
    if (onGrid(delta)) {
        plan.acceptThis = true;
        return plan;
    }
    // Off-grid: hold it. If the next frame follows within the window this
    // one was an intermediate repaint; otherwise it is accepted late.
    plan.deferThis = true;
    plan.deferDeadline = arrival + static_cast<int64_t>(kDeferWindow * m_period);
    m_deferred = true;
    m_deferredTsRaw = tsRaw;
    m_deferredArrival = arrival;
    ++m_stats.deferred;
    return plan;
}

FrameTimeline::IngestPlan FrameTimeline::onDeferTimeout() {
    std::lock_guard<std::mutex> lock(m_mutex);
    IngestPlan plan;
    if (m_deferred) {
        plan.acceptDeferredFirst = true;
        m_deferred = false;
    }
    return plan;
}

bool FrameTimeline::hasDeferred() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_deferred;
}

bool FrameTimeline::lastFrame(TimelineFrame& out) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frames.empty()) return false;
    out = m_frames.back();
    return true;
}

int FrameTimeline::popLast() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frames.empty()) return -1;
    const int slot = m_frames.back().slot;
    m_frames.pop_back();
    if (!m_frames.empty()) {
        m_lastAcceptedTsRaw = m_frames.back().tsRaw;
        m_lastAcceptedTs = m_frames.back().ts;
    } else {
        m_lastAcceptedTsRaw = 0;
        m_lastAcceptedTs = 0;
    }
    return slot;
}

void FrameTimeline::push(TimelineFrame& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    frame.index = m_nextIndex++;
    frame.ts = regularize(frame.tsRaw);

    // A hold that ends with a frame arriving at normal cadence means the
    // frame was late for our budget, not that the source stopped.
    if (m_holdPending) {
        if (!m_frames.empty() && frame.tsRaw - m_frames.back().tsRaw < 2 * m_period) {
            m_latencyBudget = std::min(m_maxLatencyBudget,
                                       m_latencyBudget + std::max(4 * kMs, m_period / 4));
            ++m_stats.starvations;
        }
        m_holdPending = false;
    }

    // Capture offset: arrival - source timestamp. Track the minimum so
    // scheduling jitter on the capture thread never delays the media clock,
    // with a slow rise so a genuine latency increase is followed.
    const int64_t sample = frame.arrival - frame.tsRaw;
    if (!m_captureOffsetValid) {
        m_captureOffset = sample;
        m_captureOffsetValid = true;
    } else {
        m_captureOffset = std::min(m_captureOffset + kMs / 10, sample);
    }

    m_lastAcceptedTsRaw = frame.tsRaw;
    m_lastAcceptedTs = frame.ts;
    m_frames.push_back(frame);
}

// --------------------------------------------------------------- consumer

FrameTimeline::Selection FrameTimeline::select(int64_t displayWallTime) {
    std::lock_guard<std::mutex> lock(m_mutex);
    Selection sel;
    if (m_frames.empty()) return sel;

    // Periodic budget review: shrink while pairs have been ready with
    // comfortable margin.
    if (m_lastBudgetReview == 0) {
        m_lastBudgetReview = displayWallTime;
    } else if (displayWallTime - m_lastBudgetReview > 2 * kSecond) {
        const int64_t threshold = static_cast<int64_t>(0.35 * m_period) + 8 * kMs;
        if (m_minSlack != INT64_MAX && m_minSlack > threshold) {
            // Give back half of the surplus, at most 4 ms per review, so a
            // transient stall's growth decays within seconds while genuine
            // jitter keeps the budget it needed.
            const int64_t give = std::min<int64_t>(4 * kMs, (m_minSlack - threshold) / 2);
            m_latencyBudget = std::max(m_minLatencyBudget, m_latencyBudget - give);
        }
        m_minSlack = INT64_MAX;
        m_lastBudgetReview = displayWallTime;
    }

    const int64_t m = displayWallTime - m_captureOffset - m_latencyBudget;
    sel.valid = true;
    sel.mediaTime = m;

    if (m < m_frames.front().ts) {
        sel.f0 = sel.f1 = m_frames.front();
        sel.startup = true;
        m_currentF1Index = sel.f1.index;
        return sel;
    }
    for (size_t i = 0; i + 1 < m_frames.size(); ++i) {
        const TimelineFrame& a = m_frames[i];
        const TimelineFrame& b = m_frames[i + 1];
        if (m >= a.ts && m < b.ts) {
            sel.f0 = a;
            sel.f1 = b;
            const int64_t duration = std::max<int64_t>(1, b.ts - a.ts);
            int64_t offset = m - a.ts;
            if (m_outputMultiplier > 0) {
                // Quantize to the requested output cadence so 2x/3x/4x show
                // each intermediate position for a whole output interval.
                const int64_t step = std::max<int64_t>(1, m_period / static_cast<int64_t>(m_outputMultiplier));
                offset = (offset / step) * step;
            }
            sel.alpha = std::clamp(static_cast<float>(static_cast<double>(offset) / static_cast<double>(duration)), 0.0f, 1.0f);
            m_currentF1Index = b.index;
            m_holdPending = false;
            return sel;
        }
    }
    // Beyond the newest frame: hold it.
    sel.f0 = sel.f1 = m_frames.back();
    sel.hold = true;
    if (!m_holdPending) m_holdPending = true;
    m_currentF1Index = sel.f1.index;
    return sel;
}

std::vector<int> FrameTimeline::retireBefore(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<int> slots;
    while (m_frames.size() > 1 && m_frames.front().index < frameIndex) {
        slots.push_back(m_frames.front().slot);
        m_frames.pop_front();
    }
    return slots;
}

void FrameTimeline::reportStarvation() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_latencyBudget = std::min(m_maxLatencyBudget,
                               m_latencyBudget + std::max(4 * kMs, m_period / 4));
    ++m_stats.starvations;
}

void FrameTimeline::reportSlack(int64_t slack) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_minSlack = std::min(m_minSlack, slack);
}

FrameTimeline::Stats FrameTimeline::stats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    Stats s = m_stats;
    s.sourcePeriodMs = static_cast<double>(m_period) / kMs;
    s.cadenceLocked = m_locked;
    s.latencyBudgetMs = static_cast<double>(m_latencyBudget) / kMs;
    s.captureOffsetMs = static_cast<double>(m_captureOffset) / kMs;
    s.frames = m_frames.size();
    return s;
}

int64_t FrameTimeline::sourcePeriod() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_period;
}

bool FrameTimeline::cadenceLocked() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_locked;
}
