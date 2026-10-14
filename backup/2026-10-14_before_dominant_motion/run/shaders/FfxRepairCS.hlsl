#include "Common.hlsli"

// Repair of FidelityFX Optical Flow's 8x8 field before anything else reads
// it (FlowEngine::recordFfx).
//
// FFX regularises its field internally, and at a large motion that spreads
// one surface's vector over its neighbours: with the camera turning
// 96 px/frame around a still character, the character's near-zero motion
// covered about 60 px of the background beside it, which then barely moved
// - a smear, and in single-field mode a second copy of the character beside
// the first. Each block here tries, besides its own vector, zero motion and
// the vectors of the blocks 1, 2, 4 and 8 blocks away in eight directions,
// and keeps the one under which its pixels match the other frame best: the
// background blocks get the background's motion back from beyond the spread,
// and the character keeps its own. A candidate must beat the block's own
// vector clearly (kRepairMargin), so a field FFX got right is left as it is.
//
// FFX's convention: on the current frame's grid, pointing into the previous
// frame, whole pixels. Output R32G32_FLOAT on the same grid, unscaled.
// --no-ffx-repair turns it off.
Texture2D<int2> FidelityFxFlow : register(t0);
Texture2D<float4> Current : register(t1);
Texture2D<float4> Previous : register(t2);
RWTexture2D<float2> Repaired : register(u0);

static const float kRepairMargin = 0.8f;   // a candidate's error relative to the own vector's
static const float kRepairFloor = 2.0f;    // and below it by at least this many luma levels
static const float kSearchAbove = 6.0f;    // mean luma error above which the near-standstill search runs

float Luma(float3 c)
{
    return 255.0f * dot(c, float3(0.299f, 0.587f, 0.114f));
}

// Mean luma difference between the block's pixels (a 4x4 subsample) and the
// previous frame displaced by v.
float BlockError(int2 block, float2 v)
{
    const int2 limit = int2(Width, Height) - 1;
    const int2 shift = int2(round(v));
    float sum = 0.0f;
    [unroll]
    for (int s = 0; s < 16; ++s)
    {
        const int2 p = clamp(block * 8 + int2(s & 3, s >> 2) * 2 + 1, int2(0, 0), limit);
        sum += abs(Luma(Current.Load(int3(p, 0)).rgb) - Luma(Previous.Load(int3(clamp(p + shift, int2(0, 0), limit), 0)).rgb));
    }
    return sum / 16.0f;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const int2 grid = int2((Width + 7) / 8, (Height + 7) / 8);
    if (any(int2(id.xy) >= grid)) return;
    const int2 block = int2(id.xy);
    const float2 own = float2(FidelityFxFlow.Load(int3(block, 0)));
    if (SearchRadius == 0)
    {
        Repaired[id.xy] = own;     // --no-ffx-repair
        return;
    }
    const float ownError = BlockError(block, own);
    float2 best = own;
    float bestError = ownError;
    const float zeroError = BlockError(block, float2(0.0f, 0.0f));
    if (zeroError < bestError) { best = float2(0.0f, 0.0f); bestError = zeroError; }
    const int2 directions[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
                                int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
    const int distances[4] = {1, 2, 4, 8};
    [loop]
    for (int d = 0; d < 4; ++d)
    {
        [loop]
        for (int k = 0; k < 8; ++k)
        {
            const int2 n = clamp(block + directions[k] * distances[d], int2(0, 0), grid - 1);
            const float2 v = float2(FidelityFxFlow.Load(int3(n, 0)));
            if (all(v == best)) continue;
            const float e = BlockError(block, v);
            if (e < bestError) { best = v; bestError = e; }
        }
    }
    // Near standstill. A character the camera orbits stays put on screen
    // while the world turns past it, and FFX gave its whole body the
    // world's motion: on a 96 px/frame turn the head read -96 px against a
    // true (0, +-8), was dragged across every generated frame and came out
    // as two or three noisy copies - the head "disappeared" (head 14 dB
    // with FFX, 45 dB with DIS). Neither the head's neighbours nor zero
    // itself (the head bobs 8 px) offered its motion. A 4 px grid within
    // 8 px of standstill does, to within 2 px, which the refinement then
    // polishes; then a 2 px step around whatever is best. Only for a block
    // that nothing so far fits (kSearchAbove): one that already matches
    // cannot gain the margin, and the search is most of this pass's cost.
    if (bestError > kSearchAbove)
    {
        [loop]
        for (int s = 0; s < 25; ++s)
        {
            const float2 v = float2(s % 5 - 2, s / 5 - 2) * 4.0f;
            if (all(v == float2(0.0f, 0.0f)) || all(v == best)) continue;
            const float e = BlockError(block, v);
            if (e < bestError) { best = v; bestError = e; }
        }
        // 2 px then 1 px around the best: without the refinement (plain
        // FFX) nothing else polishes it.
        [loop]
        for (int step = 2; step >= 1; --step)
        {
            const float2 centre = best;
            [loop]
            for (int k = 0; k < 8; ++k)
            {
                const float2 v = centre + step * float2(directions[k]);
                const float e = BlockError(block, v);
                if (e < bestError) { best = v; bestError = e; }
            }
        }
    }
    const bool clearlyBetter = bestError < kRepairMargin * ownError && bestError < ownError - kRepairFloor;
    Repaired[id.xy] = clearlyBetter ? best : own;
}
