#include "Common.hlsli"

// Packs a luma pyramid level (R16_FLOAT, 0..255) into 4 quantized bytes per
// texel (R32_UINT, width / 4) for the msad4 matcher: one load per word
// instead of four. Pixels beyond the right edge replicate the last pixel,
// matching the clamp the matcher applies elsewhere.
Texture2D<float> Luma : register(t0);
RWTexture2D<uint> Packed : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint packedWidth = (LevelWidth + 3) / 4;
    if (id.x >= packedWidth || id.y >= LevelHeight) return;
    const int maxX = (int)LevelWidth - 1;
    uint word = 0u;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        const int x = min((int)(id.x * 4) + i, maxX);
        const float v = Luma.Load(int3(x, (int)id.y, 0));
        const uint b = min(255u, (uint)(clamp(v, 0.0f, kLumaScale) + 0.5f));
        word |= b << (8 * i);
    }
    Packed[id.xy] = word;
}
