#include "Common.hlsli"

// Interpolation consumes the estimator's forward and backward vector fields
// directly. No per-pixel matching, vector reselection, or refining pass.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float2> ForwardFlow : register(t2);
Texture2D<float2> BackwardFlow : register(t3);
RWTexture2D<float4> Output : register(u0);

float2 SampleFlow(Texture2D<float2> flow, float2 pixel)
{
    const int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
    const float2 position = (pixel - (BlockSize - 1) * 0.5f) / BlockSize;
    const int2 base = int2(floor(position));
    const float2 fraction = frac(position);
    const float2 v00 = flow.Load(int3(clamp(base, int2(0, 0), gridMax), 0));
    const float2 v10 = flow.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0));
    const float2 v01 = flow.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0));
    const float2 v11 = flow.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    const float2 forward = SampleFlow(ForwardFlow, pixel);
    const float2 backward = SampleFlow(BackwardFlow, pixel);
    const float3 c0 = Frame0.SampleLevel(LinearClamp, (pixel - t * forward + 0.5f) * inverseSize, 0).rgb;
    const float3 c1 = Frame1.SampleLevel(LinearClamp, (pixel - (1.0f - t) * backward + 0.5f) * inverseSize, 0).rgb;
    Output[id.xy] = float4(lerp(c0, c1, t), 1.0f);
}
