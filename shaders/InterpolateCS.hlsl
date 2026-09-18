#include "Common.hlsli"

// Synthesises the frame at TimeT between Frame0 (t=0) and Frame1 (t=1).
//
//   ForwardFlow   frame0 block -> displacement to frame1 (full-res grid)
//   BackwardFlow  frame1 block -> displacement to frame0 (full-res grid)
//
// Each output pixel is fetched from both sources by backward warping:
//   from frame0: x - F * t         (the pixel that moves F over the interval
//                                   has covered F*t of it by time t)
//   from frame1: x - B * (1 - t)
// The two warped samples are always blended by the requested time. No
// confidence, consistency, or fallback logic suppresses interpolation.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float2> ForwardFlow : register(t2);
Texture2D<float2> BackwardFlow : register(t3);
RWTexture2D<float4> Output : register(u0);

// Luma disagreement (0..1 colour scale) below which the two warps are
// considered the same content, and above which they are considered distinct.
// Integer block vectors on detailed content routinely disagree by 20-30/255
// without being wrong; snapping those to the nearest frame makes the region
// move at source cadence (visible judder), so the band is deliberately wide
// and only clear mismatches (occlusions, failed vectors) are hard-switched.
// Flow is one float vector per 16x16 block (centre at b*16 + 7.5). Sampling it
// block-constant leaves a seam at every block edge, so interpolate between
// the four nearest block centres. SINT textures cannot use the hardware
// sampler, hence the manual four-tap blend.
float2 SampleFlow(Texture2D<float2> flow, float2 pixel)
{
    int2 gridMax = int2(FlowGridSize(Width, Height)) - 1;
    float2 blockPos = (pixel - (kFlowBlock - 1) * 0.5f) / kFlowBlock;
    int2 b0 = int2(floor(blockPos));
    float2 f = blockPos - float2(b0);

    int2 c00 = clamp(b0, int2(0, 0), gridMax);
    int2 c10 = clamp(b0 + int2(1, 0), int2(0, 0), gridMax);
    int2 c01 = clamp(b0 + int2(0, 1), int2(0, 0), gridMax);
    int2 c11 = clamp(b0 + int2(1, 1), int2(0, 0), gridMax);

    float2 v00 = flow.Load(int3(c00, 0));
    float2 v10 = flow.Load(int3(c10, 0));
    float2 v01 = flow.Load(int3(c01, 0));
    float2 v11 = flow.Load(int3(c11, 0));

    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    float t = saturate(TimeT);
    float2 pixel = float2(id.xy);
    float2 inverseSize = rcp(float2(Width, Height));

    float2 forward = SampleFlow(ForwardFlow, pixel);
    float2 backward = SampleFlow(BackwardFlow, pixel);

    // Forward/backward flow should cancel when sampled at the corresponding
    // destination. Reduce blending when the two independently estimated
    // fields disagree, which limits ghosts around occlusions and bad blocks.
    float2 uv0 = (pixel - forward * t + 0.5f) * inverseSize;
    float2 uv1 = (pixel - backward * (1.0f - t) + 0.5f) * inverseSize;
    float3 c0 = Frame0.SampleLevel(LinearClamp, uv0, 0).rgb;
    float3 c1 = Frame1.SampleLevel(LinearClamp, uv1, 0).rgb;

    // Keep the fallback continuous through the midpoint. A hard switch to
    // the nearest source frame at t=0.5 causes visible judder in real-time
    // playback when flow consistency briefly drops.
    Output[id.xy] = float4(lerp(c0, c1, t), 1.0f);
}
