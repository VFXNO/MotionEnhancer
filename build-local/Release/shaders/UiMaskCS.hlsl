#include "Synthesis.hlsli"

// Static-UI evidence for one frame pair (see the "static UI" notes in
// Synthesis.hlsli). Runs on the flow queue right after the pair's flow, at
// full resolution, and writes the updated count into the pair's ring entry;
// the next pair reads it back as PreviousUi while the frame chain is
// unbroken (unbound otherwise, which reads 0 and starts from nothing).
//
// Constants are the interpolation constants for this pair at t = 0.5, the
// midpoint, where a warp moves a pixel furthest from both frames.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> PreviousUi : register(t4);
RWTexture2D<float> Ui : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const int3 texel = int3(id.xy, 0);
    const float3 p0 = Frame0.Load(texel).rgb;
    const float3 p1 = Frame1.Load(texel).rgb;
    float count = DecodeUiCount(PreviousUi.Load(texel));

    if (MaxComponent(abs(p1 - p0)) * kLumaScale > kUiStaticTolerance)
    {
        count -= kUiChangePenalty;
    }
    else
    {
        const float t = saturate(TimeT);
        const float2 inverseSize = rcp(float2(Width, Height));
        float2 from0, from1;
        WarpSources(ForwardFlow, BackwardFlow, float2(id.xy), t, from0, from1);
        const float3 c0 = Frame0.SampleLevel(LinearClamp, (from0 + 0.5f) * inverseSize, 0).rgb;
        const float3 c1 = Frame1.SampleLevel(LinearClamp, (from1 + 0.5f) * inverseSize, 0).rgb;
        const float3 warped = lerp(c0, c1, t);
        const float3 still = lerp(p0, p1, t);
        if (MaxComponent(abs(warped - still)) * kLumaScale > kUiDamageTolerance) count += 1.0f;
    }
    Ui[id.xy] = EncodeUiCount(count);
}
