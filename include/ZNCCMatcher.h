#pragma once

#include "Image.h"
#include "Pyramid.h"
#include "MotionVector.h"

enum SubpelMode {
    SUBPEL_NONE = 0,
    SUBPEL_PARABOLIC = 1,
    SUBPEL_HALF = 2,
    SUBPEL_QUARTER = 3
};

struct MatcherParams {
    int pyramidLevels = 8;          // 8-level image pyramid
    int blockSize = 8;              // Nominal block size (e.g. 8x8)
    bool adaptiveBlockSize = true;  // Scale block size across pyramid levels
    int minBlockSize = 6;           // Block size at finest level (L0)
    int maxBlockSize = 14;          // Block size at coarsest level (L7)
    int searchRadiusCoarse = 8;     // Search radius at coarsest level (L7)
    int searchRadiusRefine = 4;     // Refinement search radius at finer levels (L6..L0)
    int gridStep = 2;               // Grid spacing for evaluation (1 = dense, 2 = 2x2 grid)
    SubpelMode subpelMode = SUBPEL_QUARTER; // Sub-pixel precision mode
    bool useSpatialPredictors = true; // Multi-candidate spatial predictors (EPZS)
    bool useGaussianWeights = true;   // Gaussian block weighting (halo suppression)
    float smoothnessWeight = 0.0005f; // Small predictor adherence bias
};

class ZNCCMatcher {
public:
    MatcherParams params;

    explicit ZNCCMatcher(const MatcherParams& p = MatcherParams()) : params(p) {}

    // Estimate forward motion field from I0 to I1 across all 8 pyramid levels
    FlowField estimateFlow(const Image& img0, const Image& img1) const;

    // Estimate motion field between two single-level grayscale images given an initial predictor field
    FlowField estimateLevel(
        const Image& gray0,
        const Image& gray1,
        const FlowField* predictor,
        int searchRadius,
        int levelIndex,
        int totalLevels
    ) const;

    // Fast Single-Pass Weighted ZNCC between template block in gray0 and candidate in gray1
    // (u, v) can be continuous float for sub-pixel bilinear evaluation!
    static float computeWeightedZNCC(
        const Image& gray0,
        const Image& gray1,
        int x, int y,
        float u, float v,
        int halfBlock,
        const float* weights,
        float sumW,
        float meanT,
        float stdT,
        bool isIntegerDisp
    );

    // Build 2D Gaussian weights for a block of size (2*halfBlock + 1)^2
    static void buildGaussianWeights(int halfBlock, std::vector<float>& weights, float& sumW);
};
