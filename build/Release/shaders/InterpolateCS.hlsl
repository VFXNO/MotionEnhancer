#include "Common.hlsli"

Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<int2> ForwardFlow : register(t2);
Texture2D<int2> BackwardFlow : register(t3);
RWTexture2D<float4> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    float2 pixel = float2(id.xy);
    float2 inverseSize = rcp(float2(Width, Height));
    uint2 flowCoord = min(id.xy / 8, (uint2(Width, Height) + 7) / 8 - 1);

    // Use exactly one motion-compensated source. This avoids crossfading,
    // double images, and ghost trails while retaining continuous motion.
    if (TimeT < 0.5f)
    {
        float2 motion = float2(ForwardFlow.Load(int3(flowCoord, 0)));
        float2 uv = (pixel - motion * TimeT + 0.5f) * inverseSize;
        Output[id.xy] = float4(Frame0.SampleLevel(LinearClamp, uv, 0).rgb, 1.0f);
    }
    else
    {
        float2 motion = float2(BackwardFlow.Load(int3(flowCoord, 0)));
        float2 uv = (pixel - motion * (1.0f - TimeT) + 0.5f) * inverseSize;
        Output[id.xy] = float4(Frame1.SampleLevel(LinearClamp, uv, 0).rgb, 1.0f);
    }
}
