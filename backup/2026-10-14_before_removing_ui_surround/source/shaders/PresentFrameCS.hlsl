#include "Common.hlsli"

// Pass-through copy of a source frame into the output surface (used when no
// interpolation is needed, e.g. the interpolation factor is 0 or 1).
Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Destination : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    float3 color = Source.Load(int3(id.xy, 0)).rgb;
    // No motion, mask or trust here: a debug view only greys the picture
    // and marks it as shown without interpolation.
    if (DebugView() > 0) color = DebugMarker(id.xy, DebugGrey(color), float3(1.0f, 0.85f, 0.0f));
    Destination[id.xy] = float4(color, 1.0f);
}
