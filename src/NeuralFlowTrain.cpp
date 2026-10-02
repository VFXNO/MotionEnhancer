#include "NeuralFlow.h"

#include "TrainingVideo.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

using namespace NeuralFlowLayout;

namespace {

constexpr float kLumaScale = 255.0f;

// Same convention as NeuralRound() in MotionSearchCS.hlsl. std::lround
// rounds x.5 away from zero and HLSL's round() rounds it to even, and a
// bilinear predictor doubled into this level's pixels lands on x.5 often
// enough for the disagreement to tip the anchor argmax a whole window.
inline int neuralRound(float v) { return static_cast<int>(std::floor(v + 0.5f)); }

// See kNeuralAnchorBand / NeuralAnchorKey in MotionSearchCS.hlsl. The anchor
// argmax is not differentiated through, so this only changes which window a
// block trains on - and it must be the window inference would pick.
constexpr float kAnchorBand = 0.25f;

inline float anchorKey(float score, float vx, float vy, float rx, float ry) {
    const float dx = vx - rx, dy = vy - ry;
    return std::floor(score / kAnchorBand) + 1.0f / (1.0f + dx * dx + dy * dy);
}

using Gradient = std::array<double, kTotal>;

// ------------------------------------------------------------ example data

// Builds frame1 by sampling frame0 through a known piecewise-constant
// displacement field, so the forward flow of every pixel is exactly the
// field value of the region it belongs to. Blocks that straddle a region
// boundary have no single correct vector and are marked invalid rather than
// supervised toward a value no estimator should produce.
struct Region {
    int x0, y0, x1, y1;        // half-open, in frame1 coordinates
    float vx, vy;              // forward flow of this region, in pixels
};

void renderWarp(const Image& source, const std::vector<Region>& regions,
                Image& frame1, Image& truth, Image& valid) {
    frame1 = Image(source.width, source.height, source.channels);
    const int gridW = (source.width + kBlock - 1) / kBlock;
    const int gridH = (source.height + kBlock - 1) / kBlock;
    truth = Image(gridW, gridH, 2);
    valid = Image(gridW, gridH, 1);

    // Later regions win, so a moving object sits on top of the background.
    std::vector<int> owner(static_cast<size_t>(source.width) * source.height, -1);
    for (int r = 0; r < static_cast<int>(regions.size()); ++r) {
        const Region& g = regions[static_cast<size_t>(r)];
        for (int y = std::max(0, g.y0); y < std::min(source.height, g.y1); ++y)
            for (int x = std::max(0, g.x0); x < std::min(source.width, g.x1); ++x)
                owner[static_cast<size_t>(y) * source.width + x] = r;
    }
    for (int y = 0; y < source.height; ++y)
        for (int x = 0; x < source.width; ++x) {
            const int r = owner[static_cast<size_t>(y) * source.width + x];
            const Region& g = regions[static_cast<size_t>(std::max(0, r))];
            for (int c = 0; c < source.channels; ++c)
                frame1.set(x, y, c, source.sampleBilinear(x - g.vx, y - g.vy, c));
        }

    for (int by = 0; by < gridH; ++by)
        for (int bx = 0; bx < gridW; ++bx) {
            int first = -2;
            bool uniform = true;
            for (int y = by * kBlock; y < (by + 1) * kBlock && uniform; ++y)
                for (int x = bx * kBlock; x < (bx + 1) * kBlock; ++x) {
                    const int cx = std::min(x, source.width - 1);
                    const int cy = std::min(y, source.height - 1);
                    const int r = owner[static_cast<size_t>(cy) * source.width + cx];
                    if (first == -2) first = r;
                    else if (r != first) { uniform = false; break; }
                }
            const Region& g = regions[static_cast<size_t>(std::max(0, first))];
            truth.set(bx, by, 0, g.vx);
            truth.set(bx, by, 1, g.vy);
            valid.set(bx, by, 0, uniform ? 1.0f : 0.0f);
        }
}

} // namespace

std::vector<NeuralExample> makeNeuralExamples(const std::vector<Image>& sources,
                                              int perSource, uint32_t seed) {
    std::mt19937 rng(seed);
    // Crops, not whole frames: four pyramid levels is enough to train weights
    // that are shared by all eight at inference, and the cost of a training
    // step is quadratic in the crop area.
    constexpr int kCrop = 192;
    std::vector<NeuralExample> examples;
    for (size_t s = 0; s < sources.size(); ++s) {
        const Image& source = sources[s];
        if (source.width < kCrop + 64 || source.height < kCrop + 64) continue;
        std::uniform_int_distribution<int> cropX(32, source.width - kCrop - 32);
        std::uniform_int_distribution<int> cropY(32, source.height - kCrop - 32);
        // Up to +-24 px of motion: four levels reach 4*(1+2+4+8) = 60 px, so
        // every generated motion is inside the recursion's range and a
        // failure is the model's, not the schedule's.
        std::uniform_real_distribution<float> globalMotion(-14.0f, 14.0f);
        std::uniform_real_distribution<float> objectMotion(-24.0f, 24.0f);
        std::uniform_int_distribution<int> objectCount(0, 2);
        std::uniform_int_distribution<int> objectSize(24, 80);
        std::uniform_int_distribution<int> objectPos(0, kCrop - 24);
        for (int i = 0; i < perSource; ++i) {
            NeuralExample example;
            const int ox = cropX(rng), oy = cropY(rng);
            example.frame0 = Image(kCrop, kCrop, 3);
            for (int y = 0; y < kCrop; ++y)
                for (int x = 0; x < kCrop; ++x)
                    for (int c = 0; c < 3; ++c)
                        example.frame0.set(x, y, c, source.get(ox + x, oy + y, c));

            std::vector<Region> regions;
            regions.push_back({0, 0, kCrop, kCrop, globalMotion(rng), globalMotion(rng)});
            const int objects = objectCount(rng);
            for (int o = 0; o < objects; ++o) {
                const int w = objectSize(rng), h = objectSize(rng);
                const int x0 = objectPos(rng), y0 = objectPos(rng);
                regions.push_back({x0, y0, x0 + w, y0 + h,
                                   objectMotion(rng), objectMotion(rng)});
            }
            renderWarp(example.frame0, regions, example.frame1, example.truth, example.valid);
            // Deterministic 70/15/15 split by index, so validation and
            // held-out sets stay fixed across runs and across models.
            const size_t index = examples.size();
            example.split = (index % 20) < 14 ? 0 : ((index % 20) < 17 ? 1 : 2);
            examples.push_back(std::move(example));
        }
    }
    return examples;
}

// ======================================================================
// Forward with the intermediates the backward pass needs
// ======================================================================

namespace {

struct LevelCache {
    Image luma;        // 1 channel, 0..255
    Image hidden;      // kHidden channels, post-ReLU
    Image feature;     // kFeatures channels, post-tanh
};

void buildLevel(const NeuralFlowModel& m, const Image& luma, const Image* parent,
                LevelCache& out) {
    out.luma = luma;
    out.hidden = Image(luma.width, luma.height, kHidden);
    for (int y = 0; y < luma.height; ++y)
        for (int x = 0; x < luma.width; ++x)
            for (int k = 0; k < kHidden; ++k) {
                float sum = m.weights[kConv1Bias + k];
                for (int t = 0; t < kTaps; ++t)
                    sum += m.weights[kConv1 + k * kTaps + t] *
                           (luma.get(x + t % 3 - 1, y + t / 3 - 1, 0) / kLumaScale);
                out.hidden.set(x, y, k, std::max(0.0f, sum));
            }
    out.feature = Image(luma.width, luma.height, kFeatures);
    for (int y = 0; y < luma.height; ++y)
        for (int x = 0; x < luma.width; ++x)
            for (int c = 0; c < kFeatures; ++c) {
                float sum = m.weights[kConv2Bias + c];
                for (int k = 0; k < kHidden; ++k)
                    for (int t = 0; t < kTaps; ++t)
                        sum += m.weights[kConv2 + (c * kHidden + k) * kTaps + t] *
                               out.hidden.get(x + t % 3 - 1, y + t / 3 - 1, k);
                if (parent) sum += m.weights[kParentGain + c] * parent->get(x / 2, y / 2, c);
                out.feature.set(x, y, c, std::tanh(sum));
            }
}

// Everything the analytic gradient treats as a constant: the predictor each
// level inherits, and the parent features each level's extractor adds in.
// Both are genuinely weight-dependent, and both are deliberately detached -
// that is what keeps the model non-recurrent and the gradient one level
// deep. Capturing them lets the gradient check differentiate the *same*
// truncated objective; a finite difference that re-runs the whole recursion
// measures a different function and disagrees by 25-60%, which is what it
// did before this existed.
struct DetachedContext {
    std::vector<Image> feature0, feature1;   // per level, the computed features
    std::vector<Image> flow;                 // per level, that level's output
};

std::vector<LevelCache> buildFeatureCache(const NeuralFlowModel& m,
                                          const std::vector<Image>& luma,
                                          const std::vector<Image>* detachedParents) {
    const int levels = static_cast<int>(luma.size());
    std::vector<LevelCache> cache(luma.size());
    for (int level = levels - 1; level >= 0; --level) {
        const Image* parent = nullptr;
        if (level + 1 < levels)
            parent = detachedParents ? &(*detachedParents)[static_cast<size_t>(level + 1)]
                                     : &cache[static_cast<size_t>(level + 1)].feature;
        buildLevel(m, luma[static_cast<size_t>(level)], parent, cache[static_cast<size_t>(level)]);
    }
    return cache;
}

// --- cost volume, duplicated from NeuralFlow.cpp's inference path ---------
// Kept local so the training path cannot drift from a shared helper being
// changed for the inference path's benefit; the GPU/CPU parity test is what
// holds all three in agreement.

// Clamp into the level, then displace, then clamp again - the order the
// shader uses. See the note on samplePixel() in NeuralFlow.cpp: a block on
// a level's right or bottom edge extends past it, and clamping after the
// displacement instead of before samples a different candidate pixel. The
// gradient must distribute to the same pixels the forward pass read, so
// this is shared by both.
inline void samplePixel(const Image& image, int originX, int originY, int k,
                        int dx, int dy, int& px, int& py, int& qx, int& qy) {
    px = std::clamp(originX + 2 * (k % 4) + 1, 0, image.width - 1);
    py = std::clamp(originY + 2 * (k / 4) + 1, 0, image.height - 1);
    qx = std::clamp(px + dx, 0, image.width - 1);
    qy = std::clamp(py + dy, 0, image.height - 1);
}
inline int sampleQuadrant(int k) { return ((k / 4) / 2) * 2 + (k % 4) / 2; }
inline bool sampleIsCentral(int k) {
    const int sx = k % 4, sy = k / 4;
    return (sx == 1 || sx == 2) && (sy == 1 || sy == 2);
}

void costDescriptors(const Image& ref, const Image& cand, int originX, int originY,
                     int dx, int dy, float cost[kCosts]) {
    for (int i = 0; i < kCosts; ++i) cost[i] = 0.0f;
    for (int k = 0; k < kSamples; ++k) {
        int px, py, qx, qy;
        samplePixel(ref, originX, originY, k, dx, dy, px, py, qx, qy);
        float energy = 0.0f;
        for (int c = 0; c < kFeatures; ++c) {
            const float d = ref.get(px, py, c) - cand.get(qx, qy, c);
            energy += 0.5f * d * d;
        }
        cost[0] += energy;
        if (sampleIsCentral(k)) cost[1] += energy;
        cost[2 + sampleQuadrant(k)] += energy;
    }
    cost[0] /= 16.0f;
    for (int i = 1; i < kCosts; ++i) cost[i] *= 0.25f;
}

inline float headScore(const NeuralFlowModel& m, const float cost[kCosts],
                       float unit[kUnits]) {
    float score = m.weights[kScoreBias];
    for (int j = 0; j < kUnits; ++j) {
        float pre = m.weights[kHeadBias + j];
        for (int i = 0; i < kCosts; ++i)
            pre += m.weights[kHeadWeight + j * kCosts + i] * cost[i];
        unit[j] = std::max(0.0f, pre);
        score += unit[j] * m.weights[kOut + j];
    }
    return score;
}

constexpr int kAnchors = 11;
constexpr float kRatio = 0.5f;
constexpr float kCentreShift = (0.25f * (kBlock - 1) - 0.5f * (kBlock - 1)) / kBlock;

std::array<float, 2> parentFlowAt(const Image& parent, float blockX, float blockY) {
    const float px = std::clamp(blockX * kRatio + kCentreShift, 0.0f,
                                static_cast<float>(parent.width - 1));
    const float py = std::clamp(blockY * kRatio + kCentreShift, 0.0f,
                                static_cast<float>(parent.height - 1));
    return {parent.sampleBilinear(px, py, 0) * 2.0f,
            parent.sampleBilinear(px, py, 1) * 2.0f};
}

int blockAnchors(const Image* parentFlow, int bx, int by,
                 std::array<std::array<float, 2>, kAnchors>& out) {
    int count = 0;
    out[count++] = parentFlow
        ? parentFlowAt(*parentFlow, static_cast<float>(bx), static_cast<float>(by))
        : std::array<float, 2>{0.0f, 0.0f};
    if (parentFlow)
        for (int n = 0; n < 8; ++n) {
            const int idx = n < 4 ? n : n + 1;
            out[count++] = parentFlowAt(*parentFlow,
                                        static_cast<float>(bx) + (idx % 3 - 1) * 2.0f,
                                        static_cast<float>(by) + (idx / 3 - 1) * 2.0f);
        }
    out[count++] = {0.0f, 0.0f};
    return count;
}

// Gradient accumulation must clamp exactly as Image::get does, or the border
// pixels every 3x3 tap reaches outside the image would lose their share.
inline void addClamped(Image& image, int x, int y, int c, float value) {
    image.data[image.index(std::clamp(x, 0, image.width - 1),
                           std::clamp(y, 0, image.height - 1), c)] += value;
}

// Truth for one pyramid level: the level-0 block grid decimated by 2^level,
// with vectors scaled into that level's pixels. A coarse block is supervised
// only where every level-0 block it covers is valid and agrees.
void levelTruth(const Image& truth0, const Image& valid0, int level,
                int gridW, int gridH, Image& truth, Image& valid) {
    const int step = 1 << level;
    const float scale = 1.0f / static_cast<float>(step);
    truth = Image(gridW, gridH, 2);
    valid = Image(gridW, gridH, 1);
    for (int by = 0; by < gridH; ++by)
        for (int bx = 0; bx < gridW; ++bx) {
            float vx = 0.0f, vy = 0.0f;
            bool ok = true;
            int count = 0;
            for (int sy = 0; sy < step && ok; ++sy)
                for (int sx = 0; sx < step; ++sx) {
                    const int fx = bx * step + sx, fy = by * step + sy;
                    if (fx >= truth0.width || fy >= truth0.height) continue;
                    if (valid0.get(fx, fy, 0) <= 0.0f) { ok = false; break; }
                    const float tx = truth0.get(fx, fy, 0), ty = truth0.get(fx, fy, 1);
                    if (count && (std::abs(tx - vx / count) > 0.01f ||
                                  std::abs(ty - vy / count) > 0.01f)) { ok = false; break; }
                    vx += tx;
                    vy += ty;
                    ++count;
                }
            if (!ok || count == 0) { valid.set(bx, by, 0, 0.0f); continue; }
            truth.set(bx, by, 0, vx / count * scale);
            truth.set(bx, by, 1, vy / count * scale);
            valid.set(bx, by, 0, 1.0f);
        }
}

struct BlockLoss {
    double loss = 0.0;
    double endPoint = 0.0;
    int large = 0;
    int count = 0;         // every supervised block, for the reported metrics
    double gradLoss = 0.0; // loss of the blocks the gradient was taken over
    int gradCount = 0;     // and how many, which is what normalizes it
};

// Forward one level and, when `grad` is given, accumulate the head and
// feature-map gradients of its supervised loss.
Image runLevel(const NeuralFlowModel& m, const LevelCache& ref, const LevelCache& cand,
               const Image* parentFlow, int level, const Image& truth, const Image& valid,
               int blockStep, Gradient* grad, Image* gradRef, Image* gradCand,
               BlockLoss& stats) {
    const int gridW = (ref.feature.width + kBlock - 1) / kBlock;
    const int gridH = (ref.feature.height + kBlock - 1) / kBlock;
    const int radius = neuralWindowRadius(level, 0);
    const int span = 2 * radius + 1;
    const int candidates = span * span;
    const float temperature = std::exp(m.weights[kLogTemperature]);
    Image flow(gridW, gridH, 4);

    std::vector<float> probability(static_cast<size_t>(candidates));
    std::vector<float> rawScore(static_cast<size_t>(candidates));
    std::vector<std::array<float, kCosts>> costs(static_cast<size_t>(candidates));
    std::vector<std::array<float, kUnits>> units(static_cast<size_t>(candidates));

    for (int by = 0; by < gridH; ++by)
        for (int bx = 0; bx < gridW; ++bx) {
            const int originX = bx * kBlock, originY = by * kBlock;
            std::array<std::array<float, 2>, kAnchors> anchors;
            const int count = blockAnchors(parentFlow, bx, by, anchors);
            int bestAnchor = 0;
            float bestScore = -1e30f;
            for (int a = 0; a < count; ++a) {
                float cost[kCosts], unit[kUnits];
                costDescriptors(ref.feature, cand.feature, originX, originY,
                                neuralRound((anchors[a][0])),
                                neuralRound((anchors[a][1])), cost);
                // Temperature-scaled, matching the shader and the CPU
                // reference: for a near-tied argmax the scale changes the
                // last-bit ordering even though it cannot change the maths.
                const float score = anchorKey(temperature * headScore(m, cost, unit),
                                              anchors[a][0], anchors[a][1],
                                              anchors[0][0], anchors[0][1]);
                if (score > bestScore) { bestScore = score; bestAnchor = a; }
            }
            const int centreX = neuralRound((anchors[bestAnchor][0]));
            const int centreY = neuralRound((anchors[bestAnchor][1]));

            float maxLogit = -1e30f;
            for (int i = 0; i < candidates; ++i) {
                const int dx = centreX + i % span - radius;
                const int dy = centreY + i / span - radius;
                costDescriptors(ref.feature, cand.feature, originX, originY, dx, dy,
                                costs[static_cast<size_t>(i)].data());
                rawScore[static_cast<size_t>(i)] =
                    headScore(m, costs[static_cast<size_t>(i)].data(),
                              units[static_cast<size_t>(i)].data());
                maxLogit = std::max(maxLogit, temperature * rawScore[static_cast<size_t>(i)]);
            }
            double sum = 0.0, vx = 0.0, vy = 0.0;
            for (int i = 0; i < candidates; ++i) {
                const float weight = std::exp(std::max(
                    -80.0f, temperature * rawScore[static_cast<size_t>(i)] - maxLogit));
                probability[static_cast<size_t>(i)] = weight;
                sum += weight;
                vx += weight * (centreX + i % span - radius);
                vy += weight * (centreY + i / span - radius);
            }
            const float fx = static_cast<float>(vx / sum);
            const float fy = static_cast<float>(vy / sum);
            const auto blockConfidence = neuralBlockConfidence(ref.luma, originX, originY);
            flow.set(bx, by, 0, fx);
            flow.set(bx, by, 1, fy);
            flow.set(bx, by, 2, blockConfidence[0]);
            flow.set(bx, by, 3, blockConfidence[1]);
            for (int i = 0; i < candidates; ++i)
                probability[static_cast<size_t>(i)] /= static_cast<float>(sum);

            if (bx >= valid.width || by >= valid.height ||
                valid.get(bx, by, 0) <= 0.0f) continue;
            const float ex = fx - truth.get(bx, by, 0);
            const float ey = fy - truth.get(bx, by, 1);
            // Charbonnier: the gradient stays bounded as the error grows, so
            // one diverged block cannot dominate a batch, and it is still
            // nonzero at the minimum's neighbourhood unlike a hard cap.
            const double norm = std::sqrt(static_cast<double>(ex) * ex +
                                          static_cast<double>(ey) * ey + 0.01);
            stats.loss += norm;
            stats.endPoint += std::sqrt(static_cast<double>(ex) * ex + ey * ey);
            if (std::sqrt(ex * ex + ey * ey) > 8.0f / static_cast<float>(1 << level))
                ++stats.large;
            ++stats.count;
            if (!grad) continue;
            if (blockStep > 1 && ((bx % blockStep) || (by % blockStep))) continue;
            // Only the subsampled blocks contribute to the gradient, so they
            // are what it must be divided by. Normalizing by every
            // supervised block instead silently scaled the step by 1/4.
            stats.gradLoss += norm;
            ++stats.gradCount;

            // ---- backward ------------------------------------------------
            const double dvx = ex / norm, dvy = ey / norm;
            double dLogTemperature = 0.0;
            for (int i = 0; i < candidates; ++i) {
                const double p = probability[static_cast<size_t>(i)];
                if (p < 1e-7) continue;
                const double dx = centreX + i % span - radius;
                const double dy = centreY + i / span - radius;
                // d(soft-argmax)/d(logit_i) = p_i * (d_i - v)
                const double dLogit = p * ((dx - fx) * dvx + (dy - fy) * dvy);
                // logit = exp(logT) * score
                dLogTemperature += dLogit * temperature * rawScore[static_cast<size_t>(i)];
                const double dScore = dLogit * temperature;

                (*grad)[kScoreBias] += dScore;
                double dCost[kCosts] = {};
                for (int j = 0; j < kUnits; ++j) {
                    (*grad)[kOut + j] += dScore * units[static_cast<size_t>(i)][j];
                    if (units[static_cast<size_t>(i)][j] <= 0.0f) continue;   // ReLU
                    const double dPre = dScore * m.weights[kOut + j];
                    (*grad)[kHeadBias + j] += dPre;
                    for (int ci = 0; ci < kCosts; ++ci) {
                        (*grad)[kHeadWeight + j * kCosts + ci] +=
                            dPre * costs[static_cast<size_t>(i)][ci];
                        dCost[ci] += dPre * m.weights[kHeadWeight + j * kCosts + ci];
                    }
                }
                // Each sample's energy reaches the block mean, one quadrant
                // and (for the central four) the centre mean.
                const int cdx = centreX + i % span - radius;
                const int cdy = centreY + i / span - radius;
                for (int k = 0; k < kSamples; ++k) {
                    double dEnergy = dCost[0] / 16.0 + dCost[2 + sampleQuadrant(k)] * 0.25;
                    if (sampleIsCentral(k)) dEnergy += dCost[1] * 0.25;
                    if (dEnergy == 0.0) continue;
                    int px, py, qx, qy;
                    samplePixel(ref.feature, originX, originY, k, cdx, cdy, px, py, qx, qy);
                    for (int c = 0; c < kFeatures; ++c) {
                        const float difference = ref.feature.get(px, py, c) -
                                                 cand.feature.get(qx, qy, c);
                        const float g = static_cast<float>(dEnergy) * difference;
                        addClamped(*gradRef, px, py, c, g);
                        addClamped(*gradCand, qx, qy, c, -g);
                    }
                }
            }
            (*grad)[kLogTemperature] += dLogTemperature;
        }
    return flow;
}

// Backward through one level of the extractor. `parent` is detached: the
// feature pyramid is built coarse-to-fine and the gradient is not propagated
// across feature levels, which keeps the parent gain trainable without
// turning the extractor into a recurrent network.
void backwardExtractor(const NeuralFlowModel& m, const LevelCache& cache,
                       const Image* parent, const Image& gradFeature, Gradient& grad) {
    Image gradHidden(cache.hidden.width, cache.hidden.height, kHidden);
    for (int y = 0; y < cache.feature.height; ++y)
        for (int x = 0; x < cache.feature.width; ++x)
            for (int c = 0; c < kFeatures; ++c) {
                const float upstream = gradFeature.get(x, y, c);
                if (upstream == 0.0f) continue;
                const float value = cache.feature.get(x, y, c);
                const double dPre = upstream * (1.0 - static_cast<double>(value) * value);
                grad[kConv2Bias + c] += dPre;
                if (parent) {
                    const float p = parent->get(x / 2, y / 2, c);
                    grad[kParentGain + c] += dPre * p;
                }
                for (int k = 0; k < kHidden; ++k)
                    for (int t = 0; t < kTaps; ++t) {
                        const int sx = x + t % 3 - 1, sy = y + t / 3 - 1;
                        grad[kConv2 + (c * kHidden + k) * kTaps + t] +=
                            dPre * cache.hidden.get(sx, sy, k);
                        addClamped(gradHidden, sx, sy, k,
                                   static_cast<float>(dPre) *
                                       m.weights[kConv2 + (c * kHidden + k) * kTaps + t]);
                    }
            }
    for (int y = 0; y < cache.hidden.height; ++y)
        for (int x = 0; x < cache.hidden.width; ++x)
            for (int k = 0; k < kHidden; ++k) {
                const float upstream = gradHidden.get(x, y, k);
                if (upstream == 0.0f || cache.hidden.get(x, y, k) <= 0.0f) continue;  // ReLU
                grad[kConv1Bias + k] += upstream;
                for (int t = 0; t < kTaps; ++t)
                    grad[kConv1 + k * kTaps + t] +=
                        upstream * (cache.luma.get(x + t % 3 - 1, y + t / 3 - 1, 0) / kLumaScale);
            }
}

// Luma pyramids do not depend on the weights, so they are built once and
// reused for every step of every epoch - this is most of a step's work.
struct PreparedExample {
    std::vector<Image> luma0, luma1;
    std::vector<Image> truth, valid;     // per level
    uint8_t split = 0;
};

std::vector<PreparedExample> prepare(const std::vector<NeuralExample>& examples, int levels) {
    std::vector<PreparedExample> prepared(examples.size());
    for (size_t e = 0; e < examples.size(); ++e) {
        const NeuralExample& example = examples[e];
        PreparedExample& out = prepared[e];
        out.split = example.split;
        out.luma0 = neuralLumaPyramid(example.frame0, levels);
        out.luma1 = neuralLumaPyramid(example.frame1, levels);
        const size_t count = out.luma0.size();
        out.truth.resize(count);
        out.valid.resize(count);
        for (size_t l = 0; l < count; ++l) {
            const int gridW = (out.luma0[l].width + kBlock - 1) / kBlock;
            const int gridH = (out.luma0[l].height + kBlock - 1) / kBlock;
            levelTruth(example.truth, example.valid, static_cast<int>(l), gridW, gridH,
                       out.truth[l], out.valid[l]);
        }
    }
    return prepared;
}

// One example: forward the whole recursion, and with `grad` also accumulate
// the gradient of every level's supervised loss.
BlockLoss runExample(const NeuralFlowModel& m, const PreparedExample& example,
                     int blockStep, Gradient* grad,
                     DetachedContext* capture = nullptr,
                     const DetachedContext* replay = nullptr) {
    const auto cache0 = buildFeatureCache(m, example.luma0, replay ? &replay->feature0 : nullptr);
    const auto cache1 = buildFeatureCache(m, example.luma1, replay ? &replay->feature1 : nullptr);
    const int levels = static_cast<int>(cache0.size());
    if (capture) {
        capture->feature0.resize(static_cast<size_t>(levels));
        capture->feature1.resize(static_cast<size_t>(levels));
        capture->flow.assign(static_cast<size_t>(levels), Image());
        for (int l = 0; l < levels; ++l) {
            capture->feature0[static_cast<size_t>(l)] = cache0[static_cast<size_t>(l)].feature;
            capture->feature1[static_cast<size_t>(l)] = cache1[static_cast<size_t>(l)].feature;
        }
    }
    BlockLoss stats;
    std::vector<Image> gradFeature0(static_cast<size_t>(levels));
    std::vector<Image> gradFeature1(static_cast<size_t>(levels));
    if (grad)
        for (int l = 0; l < levels; ++l) {
            gradFeature0[static_cast<size_t>(l)] =
                Image(cache0[static_cast<size_t>(l)].feature.width,
                      cache0[static_cast<size_t>(l)].feature.height, kFeatures);
            gradFeature1[static_cast<size_t>(l)] =
                Image(cache1[static_cast<size_t>(l)].feature.width,
                      cache1[static_cast<size_t>(l)].feature.height, kFeatures);
        }

    Image flow;
    for (int level = levels - 1; level >= 0; --level) {
        const size_t l = static_cast<size_t>(level);
        // Every level is supervised against its own truth with the coarser
        // level's output fed in as a constant. The recursion is never
        // differentiated through, but each level trains on exactly the
        // predictor distribution it meets at inference.
        const Image* parentFlow = nullptr;
        if (level + 1 < levels) parentFlow = replay ? &replay->flow[l + 1] : &flow;
        // Filled, because that is what the next level inherits at inference
        // and what the finished field is. Training on an unfilled predictor
        // would fit the weights to a recursion the GPU does not run.
        // The fill is not differentiated through - it is a fixed function of
        // the level's own output, and the predictor is detached anyway.
        flow = neuralFillFlow(runLevel(m, cache0[l], cache1[l], parentFlow,
                                       level, example.truth[l], example.valid[l], blockStep, grad,
                                       grad ? &gradFeature0[l] : nullptr,
                                       grad ? &gradFeature1[l] : nullptr, stats));
        if (capture) capture->flow[l] = flow;
    }
    if (grad)
        for (int level = 0; level < levels; ++level) {
            const size_t l = static_cast<size_t>(level);
            const Image* parent0 = level + 1 < levels ? &cache0[l + 1].feature : nullptr;
            const Image* parent1 = level + 1 < levels ? &cache1[l + 1].feature : nullptr;
            backwardExtractor(m, cache0[l], parent0, gradFeature0[l], *grad);
            backwardExtractor(m, cache1[l], parent1, gradFeature1[l], *grad);
        }
    return stats;
}

NeuralMetrics summarize(const BlockLoss& stats) {
    NeuralMetrics metrics;
    metrics.blocks = static_cast<size_t>(stats.count);
    if (stats.count) {
        metrics.loss = static_cast<float>(stats.loss / stats.count);
        metrics.endPointError = static_cast<float>(stats.endPoint / stats.count);
        metrics.largeError = static_cast<float>(stats.large) / static_cast<float>(stats.count);
    }
    return metrics;
}

unsigned workerCount() {
    const unsigned hardware = std::thread::hardware_concurrency();
    return std::max(1u, hardware ? hardware : 4u);
}

// Runs `body(index)` over [0, count) on every core.
template <typename Body>
void parallelFor(size_t count, Body body) {
    const unsigned workers = std::min<unsigned>(workerCount(), static_cast<unsigned>(count));
    if (workers <= 1) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned w = 0; w < workers; ++w)
        threads.emplace_back([&] {
            for (size_t i = next++; i < count; i = next++) body(i);
        });
    for (auto& thread : threads) thread.join();
}

} // namespace

namespace {

// A single deterministic example small enough to differentiate by hand:
// noise with structure (so the filters see gradients everywhere) plus one
// global motion and one moving rectangle.
NeuralExample gradientCheckExample(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> value(20.0f, 230.0f);
    constexpr int kSize = 64;
    Image source(kSize + 64, kSize + 64, 3);
    for (int y = 0; y < source.height; ++y)
        for (int x = 0; x < source.width; ++x) {
            // Smooth-ish structure rather than white noise: a pure noise
            // field has no consistent match, so the softmax is flat and the
            // gradient it produces is uninformatively small.
            const float base = 128.0f + 60.0f * std::sin(x * 0.21f) * std::cos(y * 0.17f);
            for (int c = 0; c < 3; ++c)
                source.set(x, y, c, std::clamp(base + 0.15f * value(rng), 0.0f, 255.0f));
        }
    NeuralExample example;
    example.frame0 = Image(kSize, kSize, 3);
    for (int y = 0; y < kSize; ++y)
        for (int x = 0; x < kSize; ++x)
            for (int c = 0; c < 3; ++c)
                example.frame0.set(x, y, c, source.get(x + 32, y + 32, c));
    const std::vector<Region> regions = {{0, 0, kSize, kSize, 2.0f, -3.0f},
                                         {20, 16, 48, 44, -5.0f, 4.0f}};
    renderWarp(example.frame0, regions, example.frame1, example.truth, example.valid);
    example.split = 0;
    return example;
}

} // namespace

bool checkNeuralGradient(uint32_t seed, bool verbose) {
    NeuralFlowModel model;
    model.initialize(seed);
    const std::vector<NeuralExample> examples = {gradientCheckExample(seed)};
    const auto prepared = prepare(examples, 3);

    Gradient analytic{};
    // Capture the detached quantities from the unperturbed run and replay
    // them into both finite-difference evaluations, so the numeric side
    // differentiates the same truncated objective the analytic side does.
    DetachedContext detached;
    const BlockLoss base = runExample(model, prepared[0], 1, &analytic, &detached);
    if (base.gradCount == 0) {
        std::cerr << "Gradient check: the example supervised no blocks\n";
        return false;
    }

    // One representative weight from every group, plus the scalars. Checking
    // all 238 would cost 476 full forward passes for no extra information:
    // a wrong derivative is wrong for the whole group.
    struct Probe { const char* name; int index; };
    const Probe probes[] = {
        {"conv1 tap (identity ch)", kConv1 + 4},
        {"conv1 tap (d/dx ch)", kConv1 + kTaps + 3},
        {"conv1 tap (laplacian ch)", kConv1 + 3 * kTaps + 1},
        {"conv1 bias", kConv1Bias + 2},
        {"conv2 centre tap", kConv2 + 4},
        {"conv2 off tap", kConv2 + (1 * kHidden + 3) * kTaps + 7},
        {"conv2 bias", kConv2Bias + 1},
        {"parent gain", kParentGain + 0},
        {"head weight (block mean)", kHeadWeight + 0},
        {"head weight (quadrant)", kHeadWeight + 2 * kCosts + 4},
        {"head bias", kHeadBias + 5},
        {"head output", kOut + 3},
        {"score bias", kScoreBias},
        {"log temperature", kLogTemperature},
    };

    bool ok = true;
    if (verbose)
        std::cout << "Gradient check (" << base.gradCount << " supervised blocks, loss "
                  << std::fixed << std::setprecision(6)
                  << base.gradLoss / base.gradCount << ")\n";
    for (const Probe& probe : probes) {
        const float original = model.weights[probe.index];
        // The forward pass contains two argmaxes (the anchor choice and the
        // softmax's max subtraction), so too small a step measures rounding
        // and too large a step can flip an anchor and measure a
        // discontinuity. 1e-3 sits between the two for these scales.
        const float h = 1e-3f;
        model.weights[probe.index] = original + h;
        const double plus = runExample(model, prepared[0], 1, nullptr, nullptr, &detached).loss;
        model.weights[probe.index] = original - h;
        const double minus = runExample(model, prepared[0], 1, nullptr, nullptr, &detached).loss;
        model.weights[probe.index] = original;

        const double numeric = (plus - minus) / (2.0 * h);
        const double exact = analytic[probe.index];
        const double scale = std::max({std::abs(numeric), std::abs(exact), 1e-4});
        const double relative = std::abs(numeric - exact) / scale;
        // 2% relative, or 1e-3 absolute. The absolute term matters: some
        // derivatives are exactly zero by construction (the score bias
        // shifts every logit equally, which a softmax ignores), and a
        // relative test against a true zero only ever measures the numeric
        // side's own cancellation noise.
        const bool pass = relative < 0.02 || std::abs(numeric - exact) < 1e-3;
        ok = ok && pass;
        if (verbose || !pass)
            std::cout << "  " << std::left << std::setw(26) << probe.name << std::right
                      << " analytic " << std::setw(12) << std::setprecision(5) << exact
                      << "  numeric " << std::setw(12) << numeric
                      << "  rel " << std::setw(9) << std::setprecision(5) << relative
                      << (pass ? "  ok" : "  MISMATCH") << "\n";
    }
    std::cout << (ok ? "Gradient check passed.\n" : "Gradient check FAILED.\n") << std::flush;
    return ok;
}

NeuralMetrics evaluateNeuralFlow(const std::vector<NeuralExample>& examples,
                                 const NeuralFlowModel& model, uint8_t split) {
    const auto prepared = prepare(examples, 4);
    std::vector<BlockLoss> perExample(prepared.size());
    parallelFor(prepared.size(), [&](size_t i) {
        if (prepared[i].split != split) return;
        perExample[i] = runExample(model, prepared[i], 1, nullptr);
    });
    BlockLoss total;
    for (const auto& one : perExample) {
        total.loss += one.loss;
        total.endPoint += one.endPoint;
        total.large += one.large;
        total.count += one.count;
    }
    return summarize(total);
}

bool trainNeuralFlow(const std::vector<NeuralExample>& examples,
                     NeuralFlowModel& model, int epochs, NeuralTrainingReport& report) {
    if (examples.empty()) return false;
    const auto prepared = prepare(examples, 4);
    std::vector<size_t> trainIndex;
    for (size_t i = 0; i < prepared.size(); ++i)
        if (prepared[i].split == 0) trainIndex.push_back(i);
    if (trainIndex.empty()) {
        std::cerr << "Neural training: no usable training examples\n";
        return false;
    }

    // Adam. The gradient is analytic, so a plain first-order method with
    // per-weight scaling is enough; the weights differ in scale by two
    // orders of magnitude (a conv tap against a log temperature), which is
    // exactly what Adam's normalization handles and plain SGD does not.
    //
    // Cosine decay from kLearningRate to a fiftieth of it. At a fixed rate
    // the run reaches roughly its final error within three epochs and then
    // oscillates around it for the rest of the schedule: the step is large
    // enough to keep stepping over the minimum it has already found.
    constexpr double kLearningRate = 3e-3;
    constexpr double kFinalFraction = 0.02;
    constexpr double kBeta1 = 0.9, kBeta2 = 0.999, kEpsilon = 1e-8;
    Gradient momentum{}, velocity{};
    int step = 0;

    NeuralFlowModel best = model;
    best.trained = true;
    NeuralMetrics bestValidation = evaluateNeuralFlow(examples, model, 1);
    const NeuralMetrics initialValidation = bestValidation;
    std::cout << "Neural training: " << trainIndex.size() << " training examples, "
              << "initial validation EPE " << std::fixed << std::setprecision(3)
              << bestValidation.endPointError << " px ("
              << std::setprecision(1) << 100.0f * bestValidation.largeError
              << "% over 8 px)\n" << std::flush;

    std::mt19937 rng(1234);
    constexpr size_t kBatch = 4;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(trainIndex.begin(), trainIndex.end(), rng);
        BlockLoss epochLoss;
        const double progress = epochs > 1
            ? static_cast<double>(epoch) / static_cast<double>(epochs - 1) : 1.0;
        const double learningRate = kLearningRate *
            (kFinalFraction + (1.0 - kFinalFraction) *
                                  0.5 * (1.0 + std::cos(3.14159265358979 * progress)));
        for (size_t start = 0; start < trainIndex.size(); start += kBatch) {
            const size_t end = std::min(start + kBatch, trainIndex.size());
            std::vector<Gradient> perExample(end - start);
            std::vector<BlockLoss> perLoss(end - start);
            parallelFor(end - start, [&](size_t slot) {
                // Stride 2 over level-0 blocks while training: a quarter of
                // the blocks give the same descent direction for a quarter
                // of the backward cost.
                perLoss[slot] = runExample(model, prepared[trainIndex[start + slot]], 2,
                                           &perExample[slot]);
            });
            Gradient grad{};
            int blocks = 0;
            for (size_t slot = 0; slot < perExample.size(); ++slot) {
                for (int w = 0; w < kTotal; ++w) grad[w] += perExample[slot][w];
                blocks += perLoss[slot].gradCount;
                epochLoss.loss += perLoss[slot].loss;
                epochLoss.endPoint += perLoss[slot].endPoint;
                epochLoss.large += perLoss[slot].large;
                epochLoss.count += perLoss[slot].count;
            }
            if (blocks == 0) continue;
            ++step;
            const double scale = 1.0 / blocks;
            const double correction1 = 1.0 - std::pow(kBeta1, step);
            const double correction2 = 1.0 - std::pow(kBeta2, step);
            for (int w = 0; w < kTotal; ++w) {
                const double g = grad[w] * scale;
                if (!std::isfinite(g)) continue;
                momentum[w] = kBeta1 * momentum[w] + (1.0 - kBeta1) * g;
                velocity[w] = kBeta2 * velocity[w] + (1.0 - kBeta2) * g * g;
                const double update = learningRate * (momentum[w] / correction1) /
                                      (std::sqrt(velocity[w] / correction2) + kEpsilon);
                model.weights[w] -= static_cast<float>(update);
            }
            // The temperature is an exponent: left unbounded it can reach a
            // value whose exp() overflows the softmax in one bad batch.
            model.weights[kLogTemperature] =
                std::clamp(model.weights[kLogTemperature], -2.0f, 6.0f);
        }
        const NeuralMetrics validation = evaluateNeuralFlow(examples, model, 1);
        const NeuralMetrics trainMetrics = summarize(epochLoss);
        std::cout << "  epoch " << std::setw(2) << epoch + 1 << "/" << epochs
                  << "  train EPE " << std::fixed << std::setprecision(3)
                  << trainMetrics.endPointError << " px   validation EPE "
                  << validation.endPointError << " px  ("
                  << std::setprecision(1) << 100.0f * validation.largeError
                  << "% over 8 px)" << std::setprecision(3);
        if (validation.endPointError < bestValidation.endPointError) {
            bestValidation = validation;
            best = model;
            best.trained = true;
            std::cout << "  <- best";
        }
        std::cout << "\n" << std::flush;
    }

    model = best;
    model.trained = true;
    report.steps = step;
    report.validation = bestValidation;
    report.heldOut = evaluateNeuralFlow(examples, model, 2);
    report.improved = bestValidation.endPointError < initialValidation.endPointError;
    std::cout << "Neural training finished: validation EPE "
              << std::fixed << std::setprecision(3) << bestValidation.endPointError
              << " px, held-out EPE " << report.heldOut.endPointError << " px ("
              << std::setprecision(1) << 100.0f * report.heldOut.largeError
              << "% over 8 px)\n" << std::flush;
    return true;
}

// ======================================================================
// CLI entry point
// ======================================================================

int trainNeuralFlowModel(const std::vector<std::string>& sourcePaths,
                         const std::string& modelPath, int epochs) {
    if (sourcePaths.empty()) {
        std::cerr << "Neural training needs at least one image or video\n";
        return 1;
    }
    // Decoded frames are only *sources* for the synthetic warps below, so a
    // handful of well-separated frames from each clip is worth more than a
    // long run of near-duplicates.
    constexpr size_t kFramesPerVideo = 12;
    std::vector<Image> sources;
    for (const std::string& path : sourcePaths) {
        std::string extension = std::filesystem::path(path).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        static const char* kVideo[] = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".wmv",
                                       ".webm", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts"};
        const bool video = std::find(std::begin(kVideo), std::end(kVideo), extension) !=
                           std::end(kVideo);
        if (video) {
            std::vector<Image> frames;
            std::string error;
            if (!decodeTrainingVideo(path, kFramesPerVideo, frames, error)) {
                std::cerr << "Could not decode " << path << ": " << error << "\n";
                return 1;
            }
            for (Image& frame : frames) sources.push_back(std::move(frame));
        } else {
            Image image = Image::load(path);
            if (image.empty()) {
                std::cerr << "Could not load " << path << "\n";
                return 1;
            }
            sources.push_back(std::move(image));
        }
    }
    // 192 px crops with a 32 px margin, so anything smaller cannot produce one.
    sources.erase(std::remove_if(sources.begin(), sources.end(), [](const Image& image) {
        return image.width < 256 || image.height < 256;
    }), sources.end());
    if (sources.empty()) {
        std::cerr << "Neural training needs colour sources at least 256x256\n";
        return 1;
    }

    std::cout << "Neural flow training: " << sources.size() << " source frame(s)\n";
    if (!checkNeuralGradient(7, false)) {
        // The trainer would still run, but it would be descending a gradient
        // that does not belong to the loss it reports. Refuse instead.
        std::cerr << "Refusing to train: the analytic gradient disagrees with the "
                     "forward pass.\n";
        return 1;
    }

    const int perSource = std::max(2, static_cast<int>(240 / sources.size()) + 1);
    auto examples = makeNeuralExamples(sources, perSource, 20261002u);
    std::cout << "Built " << examples.size() << " supervised warp examples ("
              << perSource << " per source)\n" << std::flush;
    if (examples.size() < 20) {
        std::cerr << "Too few training examples; use more or larger sources\n";
        return 1;
    }

    NeuralFlowModel model;
    model.initialize(20261002u);
    NeuralTrainingReport report;
    if (!trainNeuralFlow(examples, model, epochs, report)) return 1;
    if (!report.improved) {
        std::cerr << "Model did not improve validation error; not written\n";
        return 1;
    }
    if (!model.save(modelPath)) {
        std::cerr << "Could not write " << modelPath << "\n";
        return 1;
    }
    std::cout << "Model written: " << modelPath << "\n" << std::flush;
    return 0;
}
