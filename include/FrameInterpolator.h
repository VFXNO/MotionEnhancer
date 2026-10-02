#pragma once

#include "Image.h"
#include "MotionVector.h"
#include "ZNCCMatcher.h"

// Interpolation is unconditional: there is no occlusion mask, no
// confidence weighting and no gating that could suppress a source frame.
// Every output pixel is the time blend of its two warped samples.
struct InterpolatorParams {
    MatcherParams matcherParams;
    bool bidirectional = true;       // Estimate both forward and backward motion

    // Artifact Refinement Options
    bool enableDetailRestoration = true;   // High-frequency Laplacian texture preservation
    bool enableColorClamping = true;       // Local color bounding box clamping
    float detailStrength = 0.65f;          // Detail restoration strength (0.0 to 1.0)
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
        FlowField* outBackwardFlow = nullptr
    ) const;

    // Extract high-frequency Laplacian detail layer from an image
    static Image extractDetailLayer(const Image& img);
};
