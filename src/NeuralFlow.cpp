#include "NeuralFlow.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

using namespace NeuralFlowLayout;

namespace {

// 'MNFN'. Any earlier magic is a different weight layout; load() refuses it
// rather than reinterpreting the bytes as this architecture.
constexpr uint32_t kMagic = 0x4E464E4Du;
constexpr float kLumaScale = 255.0f;

// Same convention as NeuralRound() in MotionSearchCS.hlsl. std::lround
// rounds x.5 away from zero and HLSL's round() rounds it to even, and a
// bilinear predictor doubled into this level's pixels lands on x.5 often
// enough for the disagreement to tip the anchor argmax a whole window.
inline int neuralRound(float v) { return static_cast<int>(std::floor(v + 0.5f)); }

// See kNeuralAnchorBand / NeuralAnchorKey in MotionSearchCS.hlsl. Anchor
// scores within a band of the best are not separable by this measurement,
// and the one nearest the reference wins, so a near-tie resolves stably
// instead of on the last bit of a float.
constexpr float kAnchorBand = 0.25f;

inline float anchorKey(float score, const std::array<float, 2>& vector,
                       const std::array<float, 2>& reference) {
    const float dx = vector[0] - reference[0];
    const float dy = vector[1] - reference[1];
    return std::floor(score / kAnchorBand) + 1.0f / (1.0f + dx * dx + dy * dy);
}


// --------------------------------------------------------------- extractor

// conv1 tap t of hidden channel k, at a pixel of the 0..255 luma image.
// Luma is divided by 255 here and in LearnedFeaturesCS, so the learned
// scales are resolution- and format-independent.
inline float conv1At(const NeuralFlowModel& m, const Image& luma, int x, int y, int k) {
    float sum = m.weights[kConv1Bias + k];
    for (int t = 0; t < kTaps; ++t) {
        const int dx = t % 3 - 1, dy = t / 3 - 1;
        sum += m.weights[kConv1 + k * kTaps + t] * (luma.get(x + dx, y + dy, 0) / kLumaScale);
    }
    return std::max(0.0f, sum);   // ReLU
}

// One level of the feature extractor. `parent` is the next coarser level's
// finished features, or null at the coarsest level.
Image featureLevel(const NeuralFlowModel& m, const Image& luma, const Image* parent) {
    // conv1 over the whole level once, so conv2 reads it instead of
    // recomputing nine times per pixel.
    Image hidden(luma.width, luma.height, kHidden);
    for (int y = 0; y < luma.height; ++y)
        for (int x = 0; x < luma.width; ++x)
            for (int k = 0; k < kHidden; ++k)
                hidden.set(x, y, k, conv1At(m, luma, x, y, k));

    Image features(luma.width, luma.height, kFeatures);
    for (int y = 0; y < luma.height; ++y)
        for (int x = 0; x < luma.width; ++x)
            for (int c = 0; c < kFeatures; ++c) {
                float sum = m.weights[kConv2Bias + c];
                for (int k = 0; k < kHidden; ++k)
                    for (int t = 0; t < kTaps; ++t) {
                        const int dx = t % 3 - 1, dy = t / 3 - 1;
                        sum += m.weights[kConv2 + (c * kHidden + k) * kTaps + t] *
                               hidden.get(x + dx, y + dy, k);
                    }
                if (parent)
                    sum += m.weights[kParentGain + c] * parent->get(x / 2, y / 2, c);
                features.set(x, y, c, std::tanh(sum));
            }
    return features;
}

// -------------------------------------------------------------- cost volume

// Sample k of a block is the pixel (2*(k%4)+1, 2*(k/4)+1) inside it,
// clamped into the level, *then* displaced and clamped again.
//
// The order matters and is not interchangeable with letting Image::get()
// clamp on its own. A block on the right or bottom edge of a level extends
// past it - at the coarsest levels most of the block does - and clamping
// after adding the displacement samples a different candidate pixel than
// clamping before. The shader clamps first, so this does too; relying on
// get()'s clamping instead put the CPU reference 0.09 px away from the GPU
// at the coarsest level, with nothing else in the graph to report it.
inline void samplePixel(const Image& image, int originX, int originY, int k,
                        int dx, int dy, int& px, int& py, int& qx, int& qy) {
    px = std::clamp(originX + 2 * (k % 4) + 1, 0, image.width - 1);
    py = std::clamp(originY + 2 * (k / 4) + 1, 0, image.height - 1);
    qx = std::clamp(px + dx, 0, image.width - 1);
    qy = std::clamp(py + dy, 0, image.height - 1);
}

// Which of the six descriptors sample k contributes to, and with what
// weight. Every sample is in the block mean and in exactly one quadrant;
// the four central samples are also in the centre mean. Quadrants cost no
// extra reads and are what tell the head that a block straddles a motion
// boundary, where one mean cannot separate "matches everywhere badly" from
// "matches half the block perfectly".
inline int sampleQuadrant(int k) {
    const int sx = k % 4, sy = k / 4;
    return (sy / 2) * 2 + sx / 2;
}
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
            const float difference = ref.get(px, py, c) - cand.get(qx, qy, c);
            energy += 0.5f * difference * difference;
        }
        cost[0] += energy;
        if (sampleIsCentral(k)) cost[1] += energy;
        cost[2 + sampleQuadrant(k)] += energy;
    }
    cost[0] /= 16.0f;
    for (int i = 1; i < kCosts; ++i) cost[i] *= 0.25f;
}

// --------------------------------------------------------------- cost head

// Appearance score of a candidate, before the temperature. Intermediates
// are returned because the backward pass needs the post-ReLU activations.
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

// The temperature is applied here, not only in the softmax. For an argmax a
// positive scale is mathematically irrelevant, but it is not numerically
// irrelevant: comparing exp(logT)*a against exp(logT)*b can order two
// near-equal candidates differently from comparing a against b, and the
// shader compares the scaled values. The anchor stage is an argmax over
// eleven candidates that are often near-tied, so a last-bit disagreement
// there moves the result a whole window.
inline float candidateScore(const NeuralFlowModel& m, const Image& ref, const Image& cand,
                            int originX, int originY, int dx, int dy) {
    float cost[kCosts], unit[kUnits];
    costDescriptors(ref, cand, originX, originY, dx, dy, cost);
    return std::exp(m.weights[kLogTemperature]) * headScore(m, cost, unit);
}

// ------------------------------------------------------------------ anchors

// The eleven anchors, in the order the shader's lanes evaluate them so ties
// resolve identically: predictor, eight parent neighbours (EPZS), zero.
// Deliberately the same candidate set as the classical engine's centre
// stage. The previous model also offered a 5x5 grid of +-16 px jumps around
// the predictor, and with a nearly flat cost those were chosen by noise -
// which is where its -35 px vectors on a 10 px motion came from. Reach
// comes from the pyramid, which demonstrably finds a 48 px motion.
constexpr int kAnchors = 11;

// Bilinear parent flow at a child block's *centre*. The constant offset is
// not cosmetic: child block b covers pixels [8b, 8b+7] centred at 8b+3.5,
// and the parent block whose centre holds that same scene point is
// b*0.5 - 0.21875. Sampling at b*0.5 shifts every predictor 3.5 child
// pixels up-left and each level re-applies it. Matches SampleParentFlow.
std::array<float, 2> parentFlowAt(const Image& parent, float blockX, float blockY) {
    constexpr float kRatio = 0.5f;
    constexpr float kCentreShift = (0.25f * (kBlock - 1) - 0.5f * (kBlock - 1)) / kBlock;
    const float px = std::clamp(blockX * kRatio + kCentreShift, 0.0f,
                                static_cast<float>(parent.width - 1));
    const float py = std::clamp(blockY * kRatio + kCentreShift, 0.0f,
                                static_cast<float>(parent.height - 1));
    return {parent.sampleBilinear(px, py, 0) * 2.0f,
            parent.sampleBilinear(px, py, 1) * 2.0f};
}

// Fills `out` with the anchor candidates for one block; returns the count.
int blockAnchors(const Image* parentFlow, int bx, int by,
                 std::array<std::array<float, 2>, kAnchors>& out) {
    int count = 0;
    const std::array<float, 2> predictor = parentFlow
        ? parentFlowAt(*parentFlow, static_cast<float>(bx), static_cast<float>(by))
        : std::array<float, 2>{0.0f, 0.0f};
    out[count++] = predictor;
    if (parentFlow)
        for (int n = 0; n < 8; ++n) {
            const int idx = n < 4 ? n : n + 1;          // skip the centre
            out[count++] = parentFlowAt(*parentFlow,
                                        static_cast<float>(bx) + (idx % 3 - 1) * 2.0f,
                                        static_cast<float>(by) + (idx / 3 - 1) * 2.0f);
        }
    out[count++] = {0.0f, 0.0f};
    return count;
}

} // namespace

// ------------------------------------------------------------------ model IO

bool NeuralFlowModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0;
    uint8_t flag = 0;
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != kMagic ||
        !file.read(reinterpret_cast<char*>(weights.data()), sizeof(weights)) ||
        !file.read(reinterpret_cast<char*>(&flag), sizeof(flag))) return false;
    trained = flag == 1;
    for (float w : weights)
        if (!std::isfinite(w) || std::abs(w) > 500.0f) return false;
    return trained && file.peek() == std::char_traits<char>::eof();
}

bool NeuralFlowModel::save(const std::string& path) const {
    if (!trained) return false;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const uint32_t magic = kMagic;
    const uint8_t flag = 1;
    return static_cast<bool>(
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic))
            .write(reinterpret_cast<const char*>(weights.data()), sizeof(weights))
            .write(reinterpret_cast<const char*>(&flag), sizeof(flag)));
}

void NeuralFlowModel::initialize(uint32_t seed) {
    std::mt19937 rng(seed);
    weights.fill(0.0f);

    // conv1: an identity tap plus oriented first differences and a
    // Laplacian, so the first layer starts as a usable filter bank instead
    // of noise, with small random asymmetry to break the symmetry between
    // channels that identical gradients would otherwise preserve.
    const float bank[kHidden][kTaps] = {
        { 0, 0, 0,  0, 1, 0,  0, 0, 0},            // identity
        { 0, 0, 0, -1, 0, 1,  0, 0, 0},            // d/dx
        { 0,-1, 0,  0, 0, 0,  0, 1, 0},            // d/dy
        { 0,-1, 0, -1, 4,-1,  0,-1, 0},            // Laplacian
        {-1, 0, 1,  0, 0, 0,  1, 0,-1},            // diagonal
        { 1, 0,-1,  0, 0, 0, -1, 0, 1},            // anti-diagonal
    };
    std::normal_distribution<float> jitter(0.0f, 0.02f);
    for (int k = 0; k < kHidden; ++k) {
        for (int t = 0; t < kTaps; ++t)
            weights[kConv1 + k * kTaps + t] = bank[k][t] + jitter(rng);
        // A positive bias so the ReLU starts open: a unit whose pre-
        // activation is negative everywhere receives no gradient at all and
        // stays dead for the whole run.
        weights[kConv1Bias + k] = 0.25f;
    }

    // conv2: each output channel reads mainly one half of the bank through
    // its centre tap, so the two features start decorrelated.
    std::normal_distribution<float> small(0.0f, 0.3f);
    for (int c = 0; c < kFeatures; ++c) {
        for (int k = 0; k < kHidden; ++k)
            for (int t = 0; t < kTaps; ++t)
                weights[kConv2 + (c * kHidden + k) * kTaps + t] =
                    (t == 4 && (k % 2) == c) ? 1.0f : small(rng) * 0.1f;
        weights[kConv2Bias + c] = 0.0f;
        weights[kParentGain + c] = 0.25f;
    }

    // Head: start as the negated block-mean cost, which is plain feature
    // SAD and already a working matcher - training improves on a sane
    // baseline rather than searching for one.
    for (int j = 0; j < kUnits; ++j) {
        for (int i = 0; i < kCosts; ++i)
            weights[kHeadWeight + j * kCosts + i] = small(rng) * 0.1f;
        weights[kHeadWeight + j * kCosts + (j % kCosts)] = 1.0f;
        weights[kHeadBias + j] = 0.1f;
        weights[kOut + j] = (j % kCosts) == 0 ? -1.0f : small(rng) * 0.1f;
    }
    weights[kScoreBias] = 0.0f;
    // exp(2.3) ~ 10: a softmax sharp enough that the output follows the cost
    // minimum rather than drifting to the window's centroid.
    weights[kLogTemperature] = 2.3f;
    trained = false;
}

// ------------------------------------------------------------------ pyramid

std::vector<Image> neuralLumaPyramid(const Image& image, int levels) {
    static const float gaussian[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16, 4.0f / 16, 1.0f / 16};
    std::vector<Image> luma;
    luma.push_back(image.channels == 1 ? image : image.toGrayscale());
    while (static_cast<int>(luma.size()) < levels &&
           std::min(luma.back().width, luma.back().height) >= 2 * kBlock)
        luma.push_back(luma.back().convolve5(gaussian).downsample2x());
    return luma;
}

std::vector<Image> neuralFeaturePyramid(const std::vector<Image>& luma,
                                        const NeuralFlowModel& model) {
    std::vector<Image> features(luma.size());
    for (int level = static_cast<int>(luma.size()) - 1; level >= 0; --level) {
        const Image* parent = level + 1 < static_cast<int>(luma.size())
            ? &features[level + 1] : nullptr;
        features[level] = featureLevel(model, luma[level], parent);
    }
    return features;
}

int neuralWindowRadius(int level, int levels) {
    (void)level;
    (void)levels;
    // One dense window of 81 candidates. A strided search over a wider reach
    // was measured 6x cheaper and far less accurate: the learned cost is an
    // edge correlation with sharp minima and a coarse probe steps over them,
    // unlike the smooth luma SAD surface a three-step search assumes.
    return 4;
}

// ---------------------------------------------------------------- inference

namespace {

// Soft-argmax over the window around `anchor`. Returns the vector; fills
// `probability` (span*span) when it is not null, for the backward pass.
std::array<float, 2> softArgmax(const NeuralFlowModel& m, const Image& ref,
                                const Image& cand, int originX, int originY,
                                const std::array<float, 2>& anchor, int radius,
                                std::vector<float>* probability) {
    const int centreX = neuralRound((anchor[0]));
    const int centreY = neuralRound((anchor[1]));
    const int span = 2 * radius + 1;
    const int candidates = span * span;

    // candidateScore() already carries the temperature, so these are the
    // logits as the shader stores them in LearnedLogits.
    std::vector<float> logit(static_cast<size_t>(candidates));
    float best = -1e30f;
    for (int i = 0; i < candidates; ++i) {
        const int dx = centreX + i % span - radius;
        const int dy = centreY + i / span - radius;
        logit[i] = candidateScore(m, ref, cand, originX, originY, dx, dy);
        best = std::max(best, logit[i]);
    }
    double sum = 0.0, vx = 0.0, vy = 0.0;
    for (int i = 0; i < candidates; ++i) {
        const float weight = std::exp(std::max(-80.0f, logit[i] - best));
        logit[i] = weight;
        sum += weight;
        vx += weight * (centreX + i % span - radius);
        vy += weight * (centreY + i / span - radius);
    }
    if (probability) {
        probability->resize(static_cast<size_t>(candidates));
        for (int i = 0; i < candidates; ++i)
            (*probability)[i] = static_cast<float>(logit[i] / sum);
    }
    return {static_cast<float>(vx / sum), static_cast<float>(vy / sum)};
}

// One level of the recursion: pick the anchor by appearance cost, then the
// soft-argmax of the window around it. Four channels out, the last two the
// per-axis confidence the fill needs.
Image flowLevel(const NeuralFlowModel& m, const Image& luma, const Image& ref,
                const Image& cand, const Image* parentFlow, int level, int levels) {
    const int gridW = (ref.width + kBlock - 1) / kBlock;
    const int gridH = (ref.height + kBlock - 1) / kBlock;
    const int radius = neuralWindowRadius(level, levels);
    Image flow(gridW, gridH, 4);
    for (int by = 0; by < gridH; ++by)
        for (int bx = 0; bx < gridW; ++bx) {
            std::array<std::array<float, 2>, kAnchors> anchors;
            const int count = blockAnchors(parentFlow, bx, by, anchors);
            const int originX = bx * kBlock, originY = by * kBlock;
            // Anchor 0 is the predictor, which is the reference a tie falls
            // back to. (There is no temporal chain on this path; the shader
            // uses the previous pair's vector when it has one.)
            const std::array<float, 2> reference = anchors[0];
            int bestAnchor = 0;
            float bestScore = -1e30f;
            for (int a = 0; a < count; ++a) {
                const float score = anchorKey(
                    candidateScore(m, ref, cand, originX, originY,
                                   neuralRound((anchors[a][0])),
                                   neuralRound((anchors[a][1]))),
                    anchors[a], reference);
                if (score > bestScore) { bestScore = score; bestAnchor = a; }
            }
            const auto v = softArgmax(m, ref, cand, originX, originY,
                                      anchors[bestAnchor], radius, nullptr);
            const auto confidence = neuralBlockConfidence(luma, originX, originY);
            flow.set(bx, by, 0, v[0]);
            flow.set(bx, by, 1, v[1]);
            flow.set(bx, by, 2, confidence[0]);
            flow.set(bx, by, 3, confidence[1]);
        }
    return flow;
}

} // namespace

std::array<float, 2> neuralBlockConfidence(const Image& luma, int originX, int originY) {
    // Diagonal of the block's structure tensor, from the same central
    // differences and the same soft normalization as MotionSearchCS. Both
    // sides clamp at the level edge, Image::get here and the shader's tile
    // load there.
    constexpr float kStructureHalf = 25.0f;
    double sumGx2 = 0.0, sumGy2 = 0.0;
    for (int y = 0; y < kBlock; ++y)
        for (int x = 0; x < kBlock; ++x) {
            const int px = originX + x, py = originY + y;
            const float gx = luma.get(px + 1, py, 0) - luma.get(px - 1, py, 0);
            const float gy = luma.get(px, py + 1, 0) - luma.get(px, py - 1, 0);
            sumGx2 += static_cast<double>(gx) * gx;
            sumGy2 += static_cast<double>(gy) * gy;
        }
    const float mx = static_cast<float>(sumGx2 / (kBlock * kBlock));
    const float my = static_cast<float>(sumGy2 / (kBlock * kBlock));
    return {mx / (mx + kStructureHalf), my / (my + kStructureHalf)};
}

Image neuralFillFlow(const Image& field) {
    // Mirrors FillFlowCS.hlsl; the constants are duplicated rather than
    // shared because HLSL and C++ cannot share a header, and
    // --neural-parity-check is what holds them equal.
    constexpr int kFillRadius = 3;
    constexpr float kMeasured = 1.2f;
    constexpr float kFlowSigma = 2.0f;

    Image out(field.width, field.height, 4);
    for (int by = 0; by < field.height; ++by)
        for (int bx = 0; bx < field.width; ++bx) {
            const float vx = field.get(bx, by, 0), vy = field.get(bx, by, 1);
            const float cx = std::clamp(field.get(bx, by, 2), 0.0f, 1.0f);
            const float cy = std::clamp(field.get(bx, by, 3), 0.0f, 1.0f);
            const float knownX = std::clamp(cx / kMeasured, 0.0f, 1.0f);
            const float knownY = std::clamp(cy / kMeasured, 0.0f, 1.0f);
            float weightX = 0.0f, weightY = 0.0f, sumX = 0.0f, sumY = 0.0f;
            for (int dy = -kFillRadius; dy <= kFillRadius; ++dy)
                for (int dx = -kFillRadius; dx <= kFillRadius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int px = std::clamp(bx + dx, 0, field.width - 1);
                    const int py = std::clamp(by + dy, 0, field.height - 1);
                    const float ox = field.get(px, py, 0), oy = field.get(px, py, 1);
                    const float ddx = ox - vx, ddy = oy - vy;
                    const float similar = 1.0f / (1.0f + (ddx * ddx + ddy * ddy) /
                                                             (kFlowSigma * kFlowSigma));
                    const float inverse = 1.0f / static_cast<float>(dx * dx + dy * dy);
                    // Gated on this block's own confidence; see FillFlowCS.
                    const float gateX = 1.0f + (similar - 1.0f) * knownX;
                    const float gateY = 1.0f + (similar - 1.0f) * knownY;
                    const float wx = std::clamp(field.get(px, py, 2), 0.0f, 1.0f) * gateX * inverse;
                    const float wy = std::clamp(field.get(px, py, 3), 0.0f, 1.0f) * gateY * inverse;
                    weightX += wx;
                    weightY += wy;
                    sumX += wx * ox;
                    sumY += wy * oy;
                }
            const float neighbourX = sumX / std::max(weightX, 1e-6f);
            const float neighbourY = sumY / std::max(weightY, 1e-6f);
            const float availableX = std::clamp(weightX, 0.0f, 1.0f);
            const float availableY = std::clamp(weightY, 0.0f, 1.0f);
            const float takeX = (1.0f - knownX) * availableX;
            const float takeY = (1.0f - knownY) * availableY;
            out.set(bx, by, 0, vx + (neighbourX - vx) * takeX);
            out.set(bx, by, 1, vy + (neighbourY - vy) * takeY);
            out.set(bx, by, 2, std::max(cx, takeX * availableX));
            out.set(bx, by, 3, std::max(cy, takeY * availableY));
        }
    return out;
}

Image neuralFlowLevel(const Image& referenceLuma, const Image& referenceFeatures,
                      const Image& candidateFeatures, const NeuralFlowModel& model,
                      int level, int levels, const Image* parentFlow) {
    return neuralFillFlow(flowLevel(model, referenceLuma, referenceFeatures,
                                    candidateFeatures, parentFlow, level, levels));
}

Image neuralFlow(const std::vector<Image>& referenceLuma,
                 const std::vector<Image>& referenceFeatures,
                 const std::vector<Image>& candidateFeatures,
                 const NeuralFlowModel& model,
                 std::vector<Image>* perLevel) {
    const int levels = static_cast<int>(referenceFeatures.size());
    if (perLevel) perLevel->assign(static_cast<size_t>(levels), Image());
    Image flow;
    for (int level = levels - 1; level >= 0; --level) {
        const size_t l = static_cast<size_t>(level);
        // Fill after every level, exactly as the GPU does: the filled field
        // is what the next level inherits as its predictor, so leaving it
        // out here would make the reference a different estimator.
        flow = neuralFillFlow(flowLevel(model, referenceLuma[l], referenceFeatures[l],
                                        candidateFeatures[l],
                                        level == levels - 1 ? nullptr : &flow, level, levels));
        if (perLevel) (*perLevel)[l] = flow;
    }
    return flow;
}
