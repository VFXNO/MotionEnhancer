#include "Synthesis.hlsli"

// The pair's overall motion for AMD FSR frame generation's camera
// (FsrRealtime, --fsr-camera): one group samples the backward field (each
// pixel of the current frame -> where it was in the previous one, full
// resolution pixels) on a kGridX x kGridY grid and writes a robust mean to
// Result[0, 0]: xy = the motion, z = the share of samples that agree with it,
// w = the number of samples. Robust: the plain mean first, then twice the
// mean of only the samples within a radius of the estimate, so a character
// moving across a panning scene does not pull the camera along.
Texture2D<float4> BackwardFlow : register(t0);
RWTexture2D<float4> Result : register(u0);

static const uint kThreads = 1024;
static const uint kGridX = 64;
static const uint kGridY = 36;
static const uint kSamples = kGridX * kGridY;
static const uint kPerThread = (kSamples + kThreads - 1) / kThreads;

groupshared float3 Partial[kThreads];

float2 SampleAt(uint i)
{
    // Centres of the grid cells, kept off the outermost 5% where the flow
    // is least reliable (content entering or leaving the frame).
    const float2 cell = (float2(i % kGridX, i / kGridX) + 0.5f) / float2(kGridX, kGridY);
    const float2 position = (0.05f + 0.9f * cell) * float2(Width, Height) - 0.5f;
    return SampleFlow(BackwardFlow, position);
}

float3 Reduce(uint li, float3 value)
{
    Partial[li] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (li < stride) Partial[li] += Partial[li + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float3 total = Partial[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(1024, 1, 1)]
void CSMain(uint li : SV_GroupIndex)
{
    float2 samples[kPerThread];
    float3 sum = 0.0f;
    [unroll]
    for (uint k = 0; k < kPerThread; ++k)
    {
        const uint i = li + k * kThreads;
        samples[k] = i < kSamples ? SampleAt(i) : 0.0f;
        if (i < kSamples) sum += float3(samples[k], 1.0f);
    }
    float3 total = Reduce(li, sum);
    float2 estimate = total.xy / max(total.z, 1.0f);
    float kept = total.z;
    [unroll]
    for (uint pass = 0; pass < 2; ++pass)
    {
        const float radius = 4.0f + 0.25f * length(estimate);
        sum = 0.0f;
        [unroll]
        for (uint k = 0; k < kPerThread; ++k)
        {
            const uint i = li + k * kThreads;
            if (i < kSamples && length(samples[k] - estimate) <= radius) sum += float3(samples[k], 1.0f);
        }
        total = Reduce(li, sum);
        if (total.z >= 1.0f)
        {
            estimate = total.xy / total.z;
            kept = total.z;
        }
    }
    if (li == 0) Result[uint2(0, 0)] = float4(estimate, kept / float(kSamples), float(kSamples));
}
