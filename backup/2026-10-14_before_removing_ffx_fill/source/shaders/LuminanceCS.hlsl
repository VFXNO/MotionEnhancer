#include "Common.hlsli"

// Full-resolution RGBA frame -> 0..255 luma (pyramid level 0).
Texture2D<float4> Source : register(t0);
RWTexture2D<float> Destination : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    Destination[id.xy] = Luminance(Source.Load(int3(id.xy, 0)).rgb) * kLumaScale;
}
