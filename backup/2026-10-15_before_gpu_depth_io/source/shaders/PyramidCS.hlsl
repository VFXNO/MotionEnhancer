#include "Common.hlsli"

// One pyramid level -> the next coarser one (LevelWidth x LevelHeight is the
// destination size). Burt-Adelson reduce: separable [1 4 6 4 1]/16 Gaussian
// centred on the even source pixel (2x, 2y), so coarse pixel i sits on top
// of fine pixel 2i and a coarse flow vector scales to the fine level by an
// exact factor of 2 for coarse-to-fine flow propagation.
//
// Explicit loads rather than a bilinear-tap trick: it does not depend on the
// sampler, on the exact source/destination ratio, or on filtering support
// for the luma format, so it cannot drift on odd-sized levels.
Texture2D<float>   InputLevel  : register(t0);
RWTexture2D<float> OutputLevel : register(u0);

static const float kWeights[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;

    uint sourceW, sourceH;
    InputLevel.GetDimensions(sourceW, sourceH);
    int2 sourceMax = int2(sourceW, sourceH) - 1;
    int2 center = int2(id.xy) * 2;

    float sum = 0.0f;
    [unroll]
    for (int ky = 0; ky < 5; ++ky)
    {
        float rowSum = 0.0f;
        [unroll]
        for (int kx = 0; kx < 5; ++kx)
        {
            int2 p = clamp(center + int2(kx - 2, ky - 2), int2(0, 0), sourceMax);
            rowSum += InputLevel.Load(int3(p, 0)) * kWeights[kx];
        }
        sum += rowSum * kWeights[ky];
    }
    OutputLevel[id.xy] = sum;
}
