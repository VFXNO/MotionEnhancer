#include "Common.hlsli"

// Synthesises the frame at TimeT between Frame0 (t=0) and Frame1 (t=1).
//
//   ForwardFlow   frame0 block -> displacement to frame1 (BlockSize grid)
//   BackwardFlow  frame1 block -> displacement to frame0 (BlockSize grid)
//
// Flow is one float vector per finest-level block. Sampling it
// block-constant leaves a seam at every block edge, so the default vector
// interpolates between the four nearest block centres. SINT textures cannot
// use the hardware sampler, hence the manual four-tap blend.
//
// Around a moving object the bilinear blend ramps between "object motion"
// and "background motion" over an ~8 px band, and pixels in that band are
// warped with a vector that belongs to neither side (halo / stair-stepping).
// Per-pixel candidate selection fixes this while keeping the 8x8 grid: every
// output pixel scores the displacement vectors of its surrounding blocks
// (both fields, negated where needed so all candidates are frame0->frame1
// "through-p" displacements) with a symmetric photometric check, and keeps
// the bilinear default unless a block vector beats it by a clear margin.
// Background pixels then follow the background vector, object pixels the
// object's, and the boundary follows the silhouette instead of the block
// grid. The chosen vector warps both source samples, so the pair is
// consistent by construction and interpolation is never suppressed: when no
// candidate matches (occlusion) the best-scoring one is still the output,
// and in flat or noisy areas the hysteresis margin pins the blend to the
// bilinear default, so nothing flickers.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float2> ForwardFlow : register(t2);
Texture2D<float2> BackwardFlow : register(t3);
RWTexture2D<float4> Output : register(u0);

// Photometric taps for the candidate check: centre plus the four
// neighbours, bilinear sampled.
static const float2 kErrorTaps[5] = {
    float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(-1.0f, 0.0f),
    float2(0.0f, 1.0f), float2(0.0f, -1.0f)
};

// Hysteresis margin for switching away from the bilinear default:
// ~4 luma levels per channel and tap (4/255 * 3 channels * 5 taps).
static const float kSelectMargin = 4.0f * 3.0f * 5.0f / 255.0f;

// Candidates closer together than this (px) are the same vector; scoring
// one representative is enough. Uniform regions collapse the candidate set
// to one or two entries this way.
static const float kDedupeDist = 0.25f;

// Loads the four nearest block vectors around `pixel` (same indices the
// bilinear blend uses, including the border clamp).
void LoadFlowQuad(Texture2D<float2> flow, float2 pixel, out float2 v[4])
{
    int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
    float2 blockPos = (pixel - (BlockSize - 1) * 0.5f) / BlockSize;
    int2 b0 = int2(floor(blockPos));

    v[0] = flow.Load(int3(clamp(b0, int2(0, 0), gridMax), 0));
    v[1] = flow.Load(int3(clamp(b0 + int2(1, 0), int2(0, 0), gridMax), 0));
    v[2] = flow.Load(int3(clamp(b0 + int2(0, 1), int2(0, 0), gridMax), 0));
    v[3] = flow.Load(int3(clamp(b0 + int2(1, 1), int2(0, 0), gridMax), 0));
}

float2 SampleFlow(Texture2D<float2> flow, float2 pixel)
{
    float2 v[4];
    LoadFlowQuad(flow, pixel, v);
    float2 f = frac((pixel - (BlockSize - 1) * 0.5f) / BlockSize);
    return lerp(lerp(v[0], v[1], f.x), lerp(v[2], v[3], f.x), f.y);
}

// Symmetric photometric error of one "through-p" displacement v: the pixels
// on its trajectory (p - v*t in frame0, p + v*(1-t) in frame1) must show the
// same content. Sum of rgb differences over the five taps; the centre pair
// comes back out so the selected candidate can be blended without
// resampling.
float PairError(float2 pixel, float2 v, float t, float2 inverseSize,
                out float3 centre0, out float3 centre1)
{
    float err = 0.0f;
    centre0 = float3(0.0f, 0.0f, 0.0f);
    centre1 = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int k = 0; k < 5; ++k)
    {
        float2 p = pixel + kErrorTaps[k];
        float3 c0 = Frame0.SampleLevel(LinearClamp, (p - v * t + 0.5f) * inverseSize, 0).rgb;
        float3 c1 = Frame1.SampleLevel(LinearClamp, (p + v * (1.0f - t) + 0.5f) * inverseSize, 0).rgb;
        err += dot(abs(c0 - c1), float3(1.0f, 1.0f, 1.0f));
        if (k == 0)
        {
            centre0 = c0;
            centre1 = c1;
        }
    }
    return err;
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

    // MOTION_ENHANCER_PIXEL_SELECT=0 dispatches with FlowScale <= 0 and
    // takes the historical path: the bilinear forward and backward warps are
    // blended by time without per-pixel selection.
    if (FlowScale <= 0.0f)
    {
        float2 uv0 = (pixel - forward * t + 0.5f) * inverseSize;
        float2 uv1 = (pixel - backward * (1.0f - t) + 0.5f) * inverseSize;
        float3 c0 = Frame0.SampleLevel(LinearClamp, uv0, 0).rgb;
        float3 c1 = Frame1.SampleLevel(LinearClamp, uv1, 0).rgb;
        Output[id.xy] = float4(lerp(c0, c1, t), 1.0f);
        return;
    }

    // Candidates in frame0->frame1 sense. 0/1 are the defaults: the
    // bilinear forward vector keeps the historical frame0 warp, the negated
    // bilinear backward vector keeps the historical frame1 warp. 2..5 are
    // the nearest forward block vectors, 6..9 the negated backward ones.
    float2 cands[10];
    cands[0] = forward;
    cands[1] = -backward;
    float2 quad[4];
    LoadFlowQuad(ForwardFlow, pixel, quad);
    [unroll]
    for (int i = 0; i < 4; ++i) cands[2 + i] = quad[i];
    LoadFlowQuad(BackwardFlow, pixel, quad);
    [unroll]
    for (int i = 0; i < 4; ++i) cands[6 + i] = -quad[i];

    // Defaults are scored first and kept unless a block vector clears the
    // hysteresis margin; ties therefore always go to the default.
    float defaultErr = 1e30f;
    float bestErr = 1e30f;
    float3 bestC0 = float3(0.0f, 0.0f, 0.0f);
    float3 bestC1 = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int i = 0; i < 10; ++i)
    {
        bool dup = false;
        [unroll]
        for (int j = 0; j < i; ++j)
            dup = dup || distance(cands[i], cands[j]) < kDedupeDist;
        if (dup) continue;

        float3 s0, s1;
        float err = PairError(pixel, cands[i], t, inverseSize, s0, s1);
        if (i < 2)
        {
            defaultErr = min(defaultErr, err);
            if (err < bestErr)
            {
                bestErr = err;
                bestC0 = s0;
                bestC1 = s1;
            }
        }
        else if (err < defaultErr - kSelectMargin && err < bestErr)
        {
            bestErr = err;
            bestC0 = s0;
            bestC1 = s1;
        }
    }

    Output[id.xy] = float4(lerp(bestC0, bestC1, t), 1.0f);
}
