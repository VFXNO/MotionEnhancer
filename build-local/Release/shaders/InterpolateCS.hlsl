#include "Synthesis.hlsli"

// Interpolation consumes the estimator's forward and backward vector fields
// directly. No per-pixel matching, vector reselection, or refining pass.
// The warp itself (field conventions, trajectory solve) is WarpSources in
// Synthesis.hlsli.
//
// The one exception is screen-fixed UI, and only where UiMaskCS has
// confirmed it over several frame pairs (unbound, it reads 0 and this pass
// is exactly the plain warp):
//   * a UI pixel is shown where it is, not warped: the time blend of the
//     two frames at that same pixel, which for an unchanged pixel is the
//     pixel itself;
//   * a warped sample that lands on UI is not the moving content it is
//     supposed to fetch - the content is underneath the UI in that frame -
//     so it is dropped and the other frame supplies the pixel. Without
//     this, background moving past a HUD drags copies of the HUD with it.
// This is not a per-pixel guess between candidate vectors: the mask is a
// temporally confirmed fact about which pixels do not move, and it changes
// nothing elsewhere.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> UiMask : register(t4);
RWTexture2D<float4> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    float2 from0, from1;
    WarpSources(ForwardFlow, BackwardFlow, pixel, t, from0, from1);
    const float2 uv0 = (from0 + 0.5f) * inverseSize;
    const float2 uv1 = (from1 + 0.5f) * inverseSize;
    const float3 c0 = Frame0.SampleLevel(LinearClamp, uv0, 0).rgb;
    const float3 c1 = Frame1.SampleLevel(LinearClamp, uv1, 0).rgb;
    float3 color = lerp(c0, c1, t);

    const float ui0 = UiCoverage(UiMask.SampleLevel(LinearClamp, uv0, 0));
    const float ui1 = UiCoverage(UiMask.SampleLevel(LinearClamp, uv1, 0));
    if (ui0 + ui1 > 0.0f)
    {
        const float w0 = (1.0f - t) * (1.0f - ui0);
        const float w1 = t * (1.0f - ui1);
        // Both samples on UI: nothing better to fetch, keep the plain blend.
        if (w0 + w1 > 1e-4f) color = (w0 * c0 + w1 * c1) / (w0 + w1);
    }
    const float uiHere = UiCoverage(UiMask.Load(int3(id.xy, 0)));
    if (uiHere > 0.0f)
    {
        const float3 still = lerp(Frame0.Load(int3(id.xy, 0)).rgb, Frame1.Load(int3(id.xy, 0)).rgb, t);
        color = lerp(color, still, uiHere);
    }
    Output[id.xy] = float4(color, 1.0f);
}
