#include "Common.hlsli"
#include "Photometric.hlsli"

// Variational refinement of FidelityFX Optical Flow's field
// (FlowEngine::recordFfxRefine). FFX returns whole-pixel vectors on an 8x8
// grid. Starting from them, a per-pixel field at half resolution is refined
// by minimising
//
//   E(u) = sum  delta * psi(|I0(x+u) - I1(x)|^2)                 brightness
//             + gamma * psi(|grad I0(x+u) - grad I1(x)|^2)        gradients
//             + alpha * psi(|grad u|^2)                           smoothness
//
// with psi the Charbonnier penalty sqrt(s + eps^2) - robust, so occlusions
// and motion edges are not averaged over (Brox et al. 2004; the third stage
// of Kroeger et al.'s DIS). Each outer iteration (FfxRefineIterateCS)
// linearises the data terms around the current field - re-warping frame0 by
// it - freezes the robust weights, solves the linear system for the update
// du by red-black successive over-relaxation and adds du to the field.
//
// A field lives on its reference frame's grid and points into the other
// frame, in half-resolution pixels: the backward field (FFX's own
// convention) with frame1 as the reference, the forward field with frame0.
// "I1" below is the reference and "I0" the frame it points into. Both are
// mapped onto the pair's common brightness first (Photometric.hlsli).

// OpenCV's defaults for the same energy. Measured on the rendered pairs,
// alpha 5-60, delta 0-5 and gamma 0-20 move no pair by more than 0.5 dB.
static const float kRefineAlpha = 20.0f;     // smoothness
static const float kRefineDelta = 5.0f;      // brightness constancy
static const float kRefineGamma = 10.0f;     // gradient constancy
static const float kRefineEpsilon2 = 1e-6f;  // Charbonnier epsilon, squared
static const float kRefineOmega = 1.6f;      // over-relaxation

// d psi / d s for psi(s) = sqrt(s + eps^2): the weight a quadratic term gets
// in the linearised system.
float CharbonnierWeight(float squared)
{
    return 0.5f * rsqrt(squared + kRefineEpsilon2);
}

// FFX's backward field (after FfxRepairCS: 8x8 grid, full-resolution
// pixels) at half-resolution position q, in half-resolution pixels - the
// refinement's start (FfxRefineInitCS) and what FfxRefineToRingCS checks the
// result against. Where the four surrounding block vectors agree within
// kStartSpread, their bilinear blend; where they do not - an object's
// outline - the one of the four that matches the two pictures best around
// q. A blend there is neither motion: between a standing character's head
// (about 0) and the world turning past it (-96 px) it gave -48 px over the
// head's edge, which the refinement, one pixel per iteration, could not
// undo - the head's outline came out as noise and partly as background.
static const float kStartSpread = 2.0f;   // half-resolution pixels
float2 FfxStartAt(Texture2D<float4> vectors, Texture2D<float> frame1Luma, Texture2D<float> frame0Luma, float2 q)
{
    const int2 gridMax = int2((Width + 7) / 8, (Height + 7) / 8) - 1;
    const float2 grid = (2.0f * q + 0.5f - 3.5f) / 8.0f;
    const int2 base = int2(floor(grid));
    const float2 f = frac(grid);
    float2 v[4];
    v[0] = 0.5f * vectors.Load(int3(clamp(base, int2(0, 0), gridMax), 0)).xy;
    v[1] = 0.5f * vectors.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0)).xy;
    v[2] = 0.5f * vectors.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0)).xy;
    v[3] = 0.5f * vectors.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0)).xy;
    const float2 low = min(min(v[0], v[1]), min(v[2], v[3]));
    const float2 high = max(max(v[0], v[1]), max(v[2], v[3]));
    if (max(high.x - low.x, high.y - low.y) <= kStartSpread)
        return lerp(lerp(v[0], v[1], f.x), lerp(v[2], v[3], f.x), f.y);
    const int2 limit = int2(LevelWidth, LevelHeight) - 1;
    const float2 inverseSize = rcp(float2(LevelWidth, LevelHeight));
    const int2 centre = clamp(int2(round(q)), int2(0, 0), limit);
    float2 best = v[0];
    float bestError = 1e30f;
    [unroll]
    for (int k = 0; k < 4; ++k)
    {
        float error = 0.0f;
        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int2 at = clamp(centre + int2(dx, dy), int2(0, 0), limit);
                error += abs(frame1Luma.Load(int3(at, 0)) -
                             frame0Luma.SampleLevel(LinearClamp, (float2(at) + v[k] + 0.5f) * inverseSize, 0));
            }
        if (error < bestError) { bestError = error; best = v[k]; }
    }
    return best;
}
