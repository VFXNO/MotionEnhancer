#pragma once

#include "Image.h"
#include "MotionVector.h"
#include "ZNCCMatcher.h"

struct InterpolatorParams {
    MatcherParams matcherParams;
    float occlusionThreshold = 2.5f; // Threshold in pixels for forward-backward consistency
    bool bidirectional = true;       // Estimate both forward and backward motion

    // Artifact Refinement Options
    bool enablePhotometricGating = true;   // Soft-suppress occluded source on color discrepancy
    bool enableHoleInpainting = true;      // Bilateral inpainting of disocclusion holes
    bool enableDetailRestoration = true;   // High-frequency Laplacian texture preservation
    bool enableColorClamping = true;       // Local color bounding box clamping
    float detailStrength = 0.65f;          // Detail restoration strength (0.0 to 1.0)
    float photoGateThreshold = 20.0f;      // Color distance threshold to trigger gating
    float photoGateSigma = 25.0f;          // Sensitivity of photometric soft-suppression
};

class FrameInterpolator {
public:
    InterpolatorParams params;

    explicit FrameInterpolator(const InterpolatorParams& p = InterpolatorParams()) : params(p) {}

    // Interpolate intermediate frame at relative time t in [0.0, 1.0] (default t = 0.5)
    Image interpolate(
        const Image& frame0,
        const Image& frame1,
        float t = 0.5f,
        FlowField* outForwardFlow = nullptr,
        FlowField* outBackwardFlow = nullptr,
        Image* outOcclusionMask = nullptr
    ) const;

    // Detect occlusions using forward-backward consistency check
    // Returns mask image: 0.0 = occluded, 1.0 = valid/consistent
    static Image computeOcclusionMask(
        const FlowField& forwardFlow,
        const FlowField& backwardFlow,
        float threshold = 2.5f
    );

    // Inpaint missing / low-confidence motion vectors in disocclusion regions
    static void inpaintMotionHoles(
        std::vector<float>& accumVx,
        std::vector<float>& accumVy,
        std::vector<float>& accumWeight,
        std::vector<float>& accumConf0,
        std::vector<float>& accumConf1,
        const FlowField& fallbackFlow,
        int W, int H,
        int searchRadius = 3
    );

    // Extract high-frequency Laplacian detail layer from an image
    static Image extractDetailLayer(const Image& img);
};
