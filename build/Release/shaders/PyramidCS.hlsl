#include "Common.hlsli"

Texture2D<float>   InputLevel  : register(t0);
RWTexture2D<float> OutputLevel : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= LevelWidth || DTid.y >= LevelHeight)
        return;

    float2 sourceDims = float2(LevelWidth * 2, LevelHeight * 2);
    float2 center = float2(DTid.xy * 2) + 0.5f;
    static const float offsets[3] = { -1.2f, 0.0f, 1.2f };
    static const float weights[3] = { 5.0f / 16.0f, 6.0f / 16.0f, 5.0f / 16.0f };

    // Bilinear filtering combines each outer Gaussian tap pair, reducing the
    // separable 5x5 kernel from 25 explicit loads to nine samples.
    float sum = 0.0f;
    [unroll]
    for (int y = 0; y < 3; ++y)
    {
        [unroll]
        for (int x = 0; x < 3; ++x)
        {
            float2 uv = (center + float2(offsets[x], offsets[y])) / sourceDims;
            sum += InputLevel.SampleLevel(LinearClamp, uv, 0) * weights[x] * weights[y];
        }
    }
    OutputLevel[DTid.xy] = sum;
}
