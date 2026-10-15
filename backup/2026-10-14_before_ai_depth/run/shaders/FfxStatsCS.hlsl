#include "Common.hlsli"

// The brightness mapping for one captured frame before FidelityFX Optical
// Flow sees it (FfxNormalizeCS), and the running reference it is mapped to.
//
// One 256-thread group measures the frame's luma mean and spread (0..255)
// on a 160x90 grid of samples. State, a 4x1 R32_FLOAT texture:
//   0, 1  this frame's gain and offset (luma levels), FfxNormalizeCS reads them;
//   2, 3  the reference mean and spread: the brightness FFX has been seeing.
// A frame within kStepMean levels and kStepSpread of the reference passes
// unchanged and becomes the reference - steady scenes and ordinary pans,
// whose statistics move a little with their content, are not touched (a
// fixed target for every frame cost a 41 px pan 15 dB). A frame that jumps
// - exposure adaptation, a flash, a light switching on - is mapped back onto
// the reference, which then follows it by kFollow per frame, so FFX never
// sees a pair with the whole step between them. SearchRadius = 1 restarts
// the reference at this frame (the start of a new FFX chain).
Texture2D<float4> Frame : register(t0);
RWTexture2D<float> State : register(u0);

static const uint kThreads = 256;
static const uint2 kGrid = uint2(160, 90);
static const float kStepMean = 12.0f;     // levels
static const float kStepSpread = 0.15f;   // relative
static const float kFollow = 0.3f;
groupshared float2 Partial[kThreads];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID)
{
    float sum = 0.0f, sumSq = 0.0f;
    const uint count = kGrid.x * kGrid.y;
    for (uint i = tid.x; i < count; i += kThreads)
    {
        const uint2 at = min(uint2((float2(i % kGrid.x, i / kGrid.x) + 0.5f) / float2(kGrid) * float2(Width, Height)),
                             uint2(Width, Height) - 1u);
        const float v = 255.0f * dot(Frame.Load(int3(at, 0)).rgb, float3(0.299f, 0.587f, 0.114f));
        sum += v;
        sumSq += v * v;
    }
    Partial[tid.x] = float2(sum, sumSq);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (tid.x < stride) Partial[tid.x] += Partial[tid.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0)
    {
        const float mean = Partial[0].x / (float)count;
        const float spread = max(sqrt(max(0.0f, Partial[0].y / (float)count - mean * mean)), 1.0f);
        float referenceMean = State[uint2(2, 0)];
        float referenceSpread = State[uint2(3, 0)];
        if (SearchRadius == 1 || !(referenceSpread >= 1.0f))
        {
            referenceMean = mean;
            referenceSpread = spread;
        }
        float gain = 1.0f, offset = 0.0f;
        const float ratio = spread / referenceSpread;
        if (abs(mean - referenceMean) <= kStepMean && abs(ratio - 1.0f) <= kStepSpread)
        {
            referenceMean = mean;
            referenceSpread = spread;
        }
        else
        {
            gain = clamp(referenceSpread / spread, 0.5f, 2.0f);
            offset = referenceMean - gain * mean;
            referenceMean = lerp(referenceMean, mean, kFollow);
            referenceSpread = lerp(referenceSpread, spread, kFollow);
        }
        State[uint2(0, 0)] = gain;
        State[uint2(1, 0)] = offset;
        State[uint2(2, 0)] = referenceMean;
        State[uint2(3, 0)] = referenceSpread;
    }
}
