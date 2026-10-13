#include "FfxRefine.hlsli"

// The starting field for one direction of the refinement (FfxRefine.hlsli),
// from FFX's 8x8 grid of whole-pixel vectors (current -> previous,
// full-resolution pixels), in half-resolution pixels:
//
//   FlowScale = +1  the backward field B on frame1's grid: FFX's vectors,
//                   bilinearly blended between block centres;
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
Texture2D<int2> FfxVectors : register(t0);
RWTexture2D<float2> Field : register(u0);

// FFX's backward field at half-resolution position q, in half-resolution
// pixels. Half-resolution pixel i covers full-resolution pixels 2i and
// 2i + 1; block b's centre is at 8b + 3.5.
float2 Backward(float2 q)
{
    const int2 gridMax = int2((Width + 7) / 8, (Height + 7) / 8) - 1;
    const float2 grid = (2.0f * q + 0.5f - 3.5f) / 8.0f;
    const int2 base = int2(floor(grid));
    const float2 f = frac(grid);
    const float2 v00 = float2(FfxVectors.Load(int3(clamp(base, int2(0, 0), gridMax), 0)));
    const float2 v10 = float2(FfxVectors.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0)));
    const float2 v01 = float2(FfxVectors.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0)));
    const float2 v11 = float2(FfxVectors.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0)));
    return 0.5f * lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
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
