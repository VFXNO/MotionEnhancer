#include "Common.hlsli"
#include "Photometric.hlsli"

// DIS step 2 of 2 at one pyramid level: a flow vector for every pixel, from
// the overlapping patches that cover it (DisPatchCS). Each patch's vector is
// weighted by how well it explains *this pixel*,
//     w = 1 / max(1, |frame1(x + u) - frame0(x)|),
// so on a motion boundary a pixel takes the vector of the patch whose
// motion actually matches it rather than the average of both sides - which
// is what lets an object edge sit between two block centres instead of on
// the block grid.
//
// Both frames go through the pair's photometric mapping first, as in
// DisPatchCS.
//
// Constants: LevelWidth/Height = this level, BlockSize = patch stride.
Texture2D<float> RefLuma : register(t0);
Texture2D<float> CandLuma : register(t1);
Texture2D<float2> PatchFlow : register(t2);
Texture2D<float2> RefStats : register(t3);
Texture2D<float2> CandStats : register(t4);
RWTexture2D<float2> DenseFlow : register(u0);

static const int kPatch = 8;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const int2 levelSize = int2(LevelWidth, LevelHeight);
    const int2 p = int2(id.xy);
    uint pw, ph;
    PatchFlow.GetDimensions(pw, ph);
    const int2 lastOrigin = max(int2(0, 0), levelSize - kPatch);
    float refGain, refOffset, candGain, candOffset;
    const float2 refStats = RefStats.Load(int3(0, 0, 0));
    const float2 candStats = CandStats.Load(int3(0, 0, 0));
    PairMapping(refStats, candStats, refGain, refOffset);
    PairMapping(candStats, refStats, candGain, candOffset);
    const float reference = refGain * RefLuma.Load(int3(p, 0)) + refOffset;
    const float2 inverseSize = rcp(float2(levelSize));

    // Patches whose (clamped) origin covers p; the clamp pulls the last
    // row/column inward, so test coverage rather than trusting the index.
    const int2 first = max(int2(0, 0), (p - (kPatch - 1)) / BlockSize - 1);
    const int2 last = min(int2(pw, ph) - 1, p / BlockSize + 1);
    float2 sum = float2(0.0f, 0.0f);
    float weights = 0.0f;
    float2 nearest = float2(0.0f, 0.0f);
    for (int iy = first.y; iy <= last.y; ++iy)
    {
        for (int ix = first.x; ix <= last.x; ++ix)
        {
            const int2 origin = min(int2(ix, iy) * BlockSize, lastOrigin);
            if (any(p < origin) || any(p >= origin + kPatch)) continue;
            const float2 u = PatchFlow.Load(int3(ix, iy, 0));
            const float candidate = candGain * CandLuma.SampleLevel(LinearClamp, (float2(p) + u + 0.5f) * inverseSize, 0) + candOffset;
            const float w = rcp(max(1.0f, abs(candidate - reference)));
            sum += w * u;
            weights += w;
            nearest = u;
        }
    }
    DenseFlow[id.xy] = weights > 0.0f ? sum / weights : nearest;
}
