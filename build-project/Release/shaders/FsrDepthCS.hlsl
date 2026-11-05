#include "Common.hlsli"

// AMD FSR frame generation's depth input (FsrFrameGen / FsrRealtime). There
// is no game depth buffer. With the AI depth of the frame (DepthService's
// "nearness", 0 far .. 1 near, kSize x kSize over the whole frame) bound and
// TimeT > 0, the depth is FlowScale + TimeT * (0.5 - nearness): nearer
// surfaces get smaller depth (0 = near plane, 1 = far plane, not inverted),
// spread over TimeT around FlowScale. Without it (TimeT = 0, or no AI depth
// for this frame) every pixel gets FlowScale: one flat surface, all motion
// from the motion vectors.
Texture2D<float> Nearness : register(t0);
RWTexture2D<float> Depth : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    float depth = FlowScale;
    if (TimeT > 0.0f)
    {
        const float2 uv = (float2(id.xy) + 0.5f) / float2(LevelWidth, LevelHeight);
        depth = saturate(FlowScale + TimeT * (0.5f - Nearness.SampleLevel(LinearClamp, uv, 0)));
    }
    Depth[id.xy] = depth;
}
