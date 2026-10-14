#pragma once

#include "Image.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// A trained convolutional flow estimator: a two-layer feature extractor, a
// learned cost volume over those features, and a differentiable soft-argmax.
// Trained end to end by analytic backpropagation (see trainNeuralFlow).
//
// This replaces the earlier 120-weight model, which was fitted by random
// perturbation search and failed on cases as simple as a pure global shift
// (a -12 px translation read as +5.6 px). Three things were wrong with it,
// and each is a deliberate decision here:
//
//   * **One linear filter is not a feature extractor.** Its descriptor was a
//     single 3x3 convolution of luma followed by tanh, so the "learned cost"
//     was an edge correlation and nothing more. Here the two channels the
//     matcher samples are the output of a 3x3 -> 6 ReLU layer followed by a
//     3x3 -> 2 layer: a real filter bank with a 5x5 receptive field and a
//     hidden nonlinearity.
//
//   * **Capacity belongs in the extractor, not in the channel count.** The
//     matcher reads 16 samples of both feature maps for every candidate
//     displacement, so a third or fourth channel multiplies the bandwidth of
//     the one pass that is already bandwidth-bound at 1080p. The extractor
//     runs once per frame per pixel and is cached per capture slot, so depth
//     there is nearly free. Two channels in, all the capacity behind them.
//
//   * **The cost head must score appearance only.** The old head took the
//     candidate's offset from the predictor and from the previous frame's
//     vector as two of its ten inputs, which let a learned proximity prior
//     dominate the match cost: a candidate far from the predictor could not
//     win however well it matched. The head here sees six descriptions of
//     the sampled feature differences and nothing else. Geometry is not an
//     input to appearance.
//
// The temperature of the soft-argmax is a learned weight. Fixed at 4, the
// score spread across a window was far smaller than the softmax needed, so
// the weights came out nearly uniform and the output collapsed to the
// window's centroid - which is the anchor, which is why the old estimator
// returned its coarse guess almost unchanged.
namespace NeuralFlowLayout {

// Feature extractor. Both layers are 3x3, giving a 5x5 receptive field.
constexpr int kHidden = 6;      // conv1 output channels
constexpr int kFeatures = 2;    // conv2 output channels: what the matcher samples
constexpr int kTaps = 9;        // 3x3

constexpr int kConv1 = 0;                                   // [kHidden][9]
constexpr int kConv1Bias = kConv1 + kHidden * kTaps;         // [kHidden]
constexpr int kConv2 = kConv1Bias + kHidden;                 // [kFeatures][kHidden][9]
constexpr int kConv2Bias = kConv2 + kFeatures * kHidden * kTaps;  // [kFeatures]
constexpr int kParentGain = kConv2Bias + kFeatures;          // [kFeatures]

// Cost head: six cost descriptors -> kUnits ReLU -> one score.
constexpr int kCosts = 6;       // block mean, centre mean, four quadrant means
constexpr int kUnits = 8;
constexpr int kHeadWeight = kParentGain + kFeatures;         // [kUnits][kCosts]
constexpr int kHeadBias = kHeadWeight + kUnits * kCosts;     // [kUnits]
constexpr int kOut = kHeadBias + kUnits;                     // [kUnits]
constexpr int kScoreBias = kOut + kUnits;                    // scalar
constexpr int kLogTemperature = kScoreBias + 1;              // scalar
constexpr int kTotal = kLogTemperature + 1;                  // 238

// Block geometry, fixed by the matcher: a 4x4 grid of samples at stride 2,
// offset 1, inside each 8x8 block.
constexpr int kBlock = 8;
constexpr int kSamples = 16;

} // namespace NeuralFlowLayout

struct NeuralFlowModel {
    std::array<float, NeuralFlowLayout::kTotal> weights = {};
    bool trained = false;

    // "MNFN" files only; earlier formats carry a different layout and are
    // rejected rather than silently misread as these weights.
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    // Filter bank and head initialization before training: an identity tap
    // and oriented differences, so the untrained extractor already produces
    // something with structure for the optimizer to improve on.
    void initialize(uint32_t seed);
};

// ---------------------------------------------------------------- inference

// Luma pyramid, finest first, 0..255 per channel (Burt-Adelson reduce).
// Matches the GPU's LuminanceCS + PyramidCS chain.
std::vector<Image> neuralLumaPyramid(const Image& image, int levels);

// Two-channel learned descriptors per level, built coarse-to-fine so each
// level carries its parent's context. Matches LearnedFeaturesCS.hlsl.
std::vector<Image> neuralFeaturePyramid(const std::vector<Image>& luma,
                                        const NeuralFlowModel& model);

// Search radius for a level. Matches sadRadius()/kNeuralRadius in FlowEngine.
int neuralWindowRadius(int level, int levels);

// Per-axis confidence of one block, from the diagonal of its luma structure
// tensor: how much the block's content changes when it is displaced along
// each axis, which is how much its match cost can say about that component.
// Mirrors the confidence computation in MotionSearchCS.
std::array<float, 2> neuralBlockConfidence(const Image& luma, int originX, int originY);

// Fills the components a block could not measure from the neighbours that
// could, per axis. Mirrors FillFlowCS; see that file for why it exists.
// `field` is four channels: xy = vector, zw = per-axis confidence.
Image neuralFillFlow(const Image& field);

// The whole coarse-to-fine recursion: one vector per 8x8 block on the
// level-0 block grid. Four channels: xy = vector, zw = per-axis confidence.
// Mirrors the neural branch of MotionSearchCS plus the fill that runs after
// each level - that is the point of having it on the CPU.
// `perLevel`, when given, receives every level's own output indexed by
// level, which is what localizes a CPU/GPU disagreement to the level it
// starts at instead of only showing the accumulated result.
Image neuralFlow(const std::vector<Image>& referenceLuma,
                 const std::vector<Image>& referenceFeatures,
                 const std::vector<Image>& candidateFeatures,
                 const NeuralFlowModel& model,
                 std::vector<Image>* perLevel = nullptr);

// One level of the recursion against a supplied predictor field, rather than
// the one this CPU pass would have produced itself.
//
// This is how the parity check isolates a level. The anchor stage is an
// argmax over ten candidates, so it is discontinuous: the ~0.02 px that
// fp32 `exp` and a wave-order reduction differ by is enough to tip a
// near-tied block onto a different anchor, which moves it a whole window
// and is then inherited by every finer level. Measured on one pair, that
// compounds from 0% of blocks at the coarsest level to 8% at level 0.
// Comparing each level against the *GPU's* own predictor removes the
// accumulation, so what is left is that level's arithmetic alone - which is
// the only thing a CPU reference can be held to.
Image neuralFlowLevel(const Image& referenceLuma, const Image& referenceFeatures,
                      const Image& candidateFeatures, const NeuralFlowModel& model,
                      int level, int levels, const Image* parentFlow);

// ----------------------------------------------------------------- training

// One supervised pair: two frames and the true forward flow of each block.
// Built by warping a real frame with a known piecewise-constant field, so
// the truth is exact wherever a block does not straddle a region boundary.
struct NeuralExample {
    Image frame0, frame1;   // RGB
    Image truth;            // 2 channels on the level-0 block grid, in pixels
    Image valid;            // 1 channel, non-zero where truth is known
    uint8_t split = 0;      // 0 train, 1 validation, 2 held-out
};

struct NeuralMetrics {
    float endPointError = 0.0f;   // mean |v - truth| over valid blocks, px
    float largeError = 0.0f;      // fraction of valid blocks off by > 8 px
    float loss = 0.0f;
    size_t blocks = 0;
};

// Generates supervised examples from source frames: a global translation
// plus a few independently moving rectangles, which are the two cases the
// estimator has to get right (camera motion and a small fast object).
std::vector<NeuralExample> makeNeuralExamples(const std::vector<Image>& sources,
                                              int perSource, uint32_t seed);

NeuralMetrics evaluateNeuralFlow(const std::vector<NeuralExample>& examples,
                                 const NeuralFlowModel& model, uint8_t split);

struct NeuralTrainingReport {
    NeuralMetrics validation;
    NeuralMetrics heldOut;
    int steps = 0;
    bool improved = false;
};

// Adam on the analytic gradient of the end-point error, with every pyramid
// level supervised against the truth it is responsible for and the coarser
// level's predictor fed in detached. The recursion is therefore never
// differentiated through, but every level trains on the predictor
// distribution it actually sees at inference - which is the mismatch that
// sank the previous model (it was fitted on one flat +-8 window and deployed
// as an eight-level recursion).
bool trainNeuralFlow(const std::vector<NeuralExample>& examples,
                     NeuralFlowModel& model, int epochs, NeuralTrainingReport& report);

// Compares the analytic gradient with a central finite difference on a
// small example. Prints every weight group it checks and returns false on a
// mismatch. Run this after touching the forward pass or the backward pass:
// a backward implementation nobody checks against the forward one is a
// silent trainer that optimizes the wrong objective.
bool checkNeuralGradient(uint32_t seed, bool verbose);

// CLI: builds examples from images/videos, trains, and writes the model.
int trainNeuralFlowModel(const std::vector<std::string>& sourcePaths,
                         const std::string& modelPath, int epochs);
