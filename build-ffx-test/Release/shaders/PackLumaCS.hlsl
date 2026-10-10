#include "Common.hlsli"
#include "Photometric.hlsli"

// Packs a luma pyramid level (R16_FLOAT, 0..255) into 4 bytes per texel
// (R32_UINT, width / 4) for the SAD matcher. Each byte is compared
// individually. Pixels beyond the right edge replicate the last pixel,
// matching the clamp the matcher applies elsewhere.
//
// The bytes are this frame's luma mapped onto the brightness the pair has in
// common (Photometric.hlsli): OwnStats and OtherStats are the two frames'
// whole-frame statistics. Packed per pair, not per frame, for that reason.
//
// Local transforms (luma minus a local mean, or divided by local contrast)
// were measured too: they fix the brightness case as well, but they make a
// pixel's byte depend on its neighbours, which differ across a moving
// object's edge between the two frames, and cost 0.3-1.5 dB on ordinary
// content.
Texture2D<float> Luma : register(t0);
Texture2D<float2> OwnStats : register(t1);
Texture2D<float2> OtherStats : register(t2);
RWTexture2D<uint> Packed : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint packedWidth = (LevelWidth + 3) / 4;
    if (id.x >= packedWidth || id.y >= LevelHeight) return;
    float gain, offset;
    PairMapping(OwnStats.Load(int3(0, 0, 0)), OtherStats.Load(int3(0, 0, 0)), gain, offset);
    const int maxX = (int)LevelWidth - 1;
    uint word = 0u;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        const int x = min((int)(id.x * 4) + i, maxX);
        const float v = gain * Luma.Load(int3(x, (int)id.y, 0)) + offset;
        const uint b = min(255u, (uint)(clamp(v, 0.0f, kLumaScale) + 0.5f));
        word |= b << (8 * i);
    }
    Packed[id.xy] = word;
}
