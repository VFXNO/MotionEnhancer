#include "FfxRefine.hlsli"

// One refined half-resolution field (FfxRefine.hlsli) into a flow-ring
// entry: a 2x2-pixel grid at full resolution is exactly the half-resolution
// pixel grid, so each ring texel is its pixel's vector in full-resolution
// pixels. The field already has its direction's convention (forward on
// frame0's grid, backward on frame1's); FlowScale = -1 negates it, which the
// single-field mode uses to write the forward entry from the backward field.
Texture2D<float2> Field : register(t0);       // refined field (the backward one), half-resolution pixels
Texture2D<float4> Start : register(t1);       // FFX's field after FfxRepairCS (8x8 grid, full-resolution pixels)
Texture2D<float> RefLuma : register(t2);      // the field's reference frame, half resolution, 0..255
Texture2D<float> CandLuma : register(t3);     // the frame it points into
Texture2D<float2> RefStats : register(t4);
Texture2D<float2> CandStats : register(t5);
RWTexture2D<float2> RingFlow : register(u0);

// The refined vector is kept only where it matches the pictures at least as
// well as the vector the refinement started from, over a kCheckRadius
// neighbourhood: where it does not, the refinement has made the field
// worse - real game content with fine detail and motion blur, where the
// linearisation does not hold - and FFX's own vector stands (as DIS keeps
// its start when Lucas-Kanade ends worse). The refinement then cannot make
// any region worse than FFX itself.
static const int kCheckRadius = 2;

// FFX's backward field at half-resolution position q, in half-resolution
// pixels (FfxRefineInitCS's start).
float2 StartAt(float2 q)
{
    const int2 gridMax = int2((Width + 7) / 8, (Height + 7) / 8) - 1;
    const float2 grid = (2.0f * q + 0.5f - 3.5f) / 8.0f;
    const int2 base = int2(floor(grid));
    const float2 f = frac(grid);
    const float2 v00 = Start.Load(int3(clamp(base, int2(0, 0), gridMax), 0)).xy;
    const float2 v10 = Start.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0)).xy;
    const float2 v01 = Start.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0)).xy;
    const float2 v11 = Start.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0)).xy;
    return 0.5f * lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

float MatchError(int2 p, float2 v, float refGain, float refOffset, float candGain, float candOffset)
{
    const int2 limit = int2(LevelWidth, LevelHeight) - 1;
    const float2 inverseSize = rcp(float2(LevelWidth, LevelHeight));
    float sum = 0.0f;
    [unroll]
    for (int dy = -kCheckRadius; dy <= kCheckRadius; ++dy)
        [unroll]
        for (int dx = -kCheckRadius; dx <= kCheckRadius; ++dx)
        {
            const int2 q = clamp(p + int2(dx, dy), int2(0, 0), limit);
            const float here = refGain * RefLuma.Load(int3(q, 0)) + refOffset;
            const float there = candGain * CandLuma.SampleLevel(LinearClamp, (float2(q) + v + 0.5f) * inverseSize, 0) + candOffset;
            sum += abs(here - there);
        }
    return sum;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, 2);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    const int2 p = min(int2(id.xy), levelMax);
    float2 v = Field.Load(int3(p, 0));
    float refGain, refOffset, candGain, candOffset;
    const float2 refStats = RefStats.Load(int3(0, 0, 0));
    const float2 candStats = CandStats.Load(int3(0, 0, 0));
    PairMapping(refStats, candStats, refGain, refOffset);
    PairMapping(candStats, refStats, candGain, candOffset);
    const float2 start = StartAt(float2(p));
    if (MatchError(p, start, refGain, refOffset, candGain, candOffset) < MatchError(p, v, refGain, refOffset, candGain, candOffset))
        v = start;
    RingFlow[id.xy] = FlowScale * 2.0f * v;
}
