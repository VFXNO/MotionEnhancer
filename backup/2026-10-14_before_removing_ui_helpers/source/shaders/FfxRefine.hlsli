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
