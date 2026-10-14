#include "FfxRefine.hlsli"

// The starting field for one direction of the refinement (FfxRefine.hlsli),
// from FFX's 8x8 grid of whole-pixel vectors (current -> previous,
// full-resolution pixels), in half-resolution pixels:
//
//   FlowScale = +1  the backward field B on frame1's grid: FFX's vectors,
//                   bilinearly blended between block centres, or at an
//                   outline the block vector that fits there (FfxStartAt);
//   FlowScale = -1  the forward field F on frame0's grid: B turned around.
//                   The point p of frame0 lands at p + F(p) in frame1, and
//                   the vector there points back: F(p) = -B(p + F(p)), a
//                   fixed point solved by three iterations from -B(p). Where
//                   p is hidden in frame1 there is no such point and this
//                   lands on the occluder; the refinement then follows the
//                   pictures, so the two fields disagree exactly there -
//                   which is what InterpolateCS's occlusion weighting reads.
//
// Constants: Width/Height = full resolution, LevelWidth/Height = the
// half-resolution field.
Texture2D<float4> FfxVectors : register(t0);   // FFX's field after FfxRepairCS
Texture2D<float> Frame1Luma : register(t1);    // half resolution (pyramid level 1)
Texture2D<float> Frame0Luma : register(t2);
RWTexture2D<float2> Field : register(u0);

// FFX's backward field at half-resolution position q (FfxStartAt).
float2 Backward(float2 q)
{
    return FfxStartAt(FfxVectors, Frame1Luma, Frame0Luma, q);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const float2 p = float2(id.xy);
    float2 v = Backward(p);
    if (FlowScale < 0.0f)
    {
        v = -v;
        [unroll]
        for (int k = 0; k < 3; ++k) v = -Backward(p + v);
    }
    Field[id.xy] = v;
}
