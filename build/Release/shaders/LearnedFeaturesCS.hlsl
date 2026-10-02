#include "Common.hlsli"

// Trainable local feature extraction from frame luma. Matching uses these
// features rather than pixelwise SAD or MSAD.
Texture2D<float> Luma : register(t0);
Texture2D<float2> ParentFeatures : register(t1);
RWTexture2D<float2> Features : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    float2 value = FeatureControl.xy;
    [unroll]
    for (int y = -1; y <= 1; ++y)
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const int tap = (y + 1) * 3 + x + 1;
            const int2 pixel = clamp(int2(id.xy) + int2(x, y), int2(0, 0),
                                     int2(LevelWidth, LevelHeight) - 1);
            const float luma = Luma.Load(int3(pixel, 0)) / kLumaScale;
            value.x += FeatureWeights[tap / 4][tap % 4] * luma;
            value.y += FeatureWeights[3 + tap / 4][tap % 4] * luma;
        }
    if (LevelIndex < TotalLevels - 1)
    {
        uint parentWidth, parentHeight;
        ParentFeatures.GetDimensions(parentWidth, parentHeight);
        const int2 parent = clamp(int2(id.xy) / 2, int2(0, 0),
                                  int2(parentWidth, parentHeight) - 1);
        value += ParentFeatures.Load(int3(parent, 0)) * FeatureControl.zw;
    }
    Features[id.xy] = tanh(value);
}
