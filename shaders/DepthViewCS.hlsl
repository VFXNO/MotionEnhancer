#include "Common.hlsli"

// Debug view 5 (Ctrl+Alt+F3, MOTION_ENHANCER_DEBUG_VIEW=5): the AI depth
// (DepthService) of what is on screen, drawn over the frame InterpolateCS or
// PresentFrameCS just wrote. Nearness 0 (far) .. 1 (near) as a colour ramp:
// dark blue far, through cyan, green and yellow, to bright red near. A
// generated frame shows its two frames' depth blended at its time t, as the
// picture is; a real frame shows its own (both inputs the same texture).
// Where no depth is available (AI depth off, or this frame's not finished)
// the pass does not run and the view stays grey.
Texture2D<float> Depth0 : register(t0);
Texture2D<float> Depth1 : register(t1);
RWTexture2D<float4> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float2 uv = (float2(id.xy) + 0.5f) / float2(Width, Height);
    const float t = saturate(TimeT);
    const float near = saturate(lerp(Depth0.SampleLevel(LinearClamp, uv, 0), Depth1.SampleLevel(LinearClamp, uv, 0), t));
    float3 color = DebugHue(0.66f * (1.0f - near)) * (0.3f + 0.7f * near);
    const bool real = t < 0.01f || t > 0.99f;
    color = DebugMarker(id.xy, color, real ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 1.0f));
    Output[id.xy] = float4(color, 1.0f);
}
