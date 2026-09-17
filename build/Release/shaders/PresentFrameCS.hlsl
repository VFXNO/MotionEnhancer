#include "Common.hlsli"

Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Destination : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    Destination[id.xy] = float4(Source.Load(int3(id.xy, 0)).rgb, 1.0f);
}
