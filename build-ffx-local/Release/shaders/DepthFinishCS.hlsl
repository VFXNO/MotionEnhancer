#include "Common.hlsli"

// The network's raw output (relative inverse depth, larger = nearer, at an
// unknown scale and shift) mapped to "nearness" 0 (far) .. 1 (near): its 2nd
// percentile becomes 0 and its 98th 1, measured on every 4th value (a
// 128 x 128 grid), as DepthEstimator::estimate does on the CPU. One group of
// 1024 threads: the range, a 1024-bin histogram over it, the two
// percentiles, then every texel.
Texture2D<float> Raw : register(t0);
RWTexture2D<float> Nearness : register(u0);

static const uint kSize = 256;
static const uint kThreads = 1024;
static const uint kBins = 1024;
static const uint kSamples = (kSize / 2) * (kSize / 2);   // 16384

groupshared float Low[kThreads];
groupshared float High[kThreads];
groupshared uint Histogram[kBins];
groupshared float2 Range;

float SampleAt(uint i)
{
    return Raw.Load(int3((i % (kSize / 2)) * 2, (i / (kSize / 2)) * 2, 0));
}

[numthreads(1024, 1, 1)]
void CSMain(uint li : SV_GroupIndex)
{
    float lo = 1e30f, hi = -1e30f;
    for (uint i = li; i < kSamples; i += kThreads)
    {
        const float v = SampleAt(i);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    Low[li] = lo;
    High[li] = hi;
    Histogram[li] = 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (li < stride)
        {
            Low[li] = min(Low[li], Low[li + stride]);
            High[li] = max(High[li], High[li + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float minimum = Low[0];
    const float width = max(High[0] - minimum, 1e-12f) / (float)kBins;
    for (uint j = li; j < kSamples; j += kThreads)
    {
        const uint bin = min((uint)((SampleAt(j) - minimum) / width), kBins - 1u);
        InterlockedAdd(Histogram[bin], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0)
    {
        const uint lowRank = kSamples / 50, highRank = kSamples - 1u - kSamples / 50;
        uint seen = 0u;
        float low = minimum, high = minimum + width * kBins;
        bool lowFound = false;
        for (uint b = 0; b < kBins; ++b)
        {
            seen += Histogram[b];
            if (!lowFound && seen > lowRank) { low = minimum + (b + 0.5f) * width; lowFound = true; }
            if (seen > highRank) { high = minimum + (b + 0.5f) * width; break; }
        }
        Range = float2(low, high > low ? 1.0f / (high - low) : 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();
    const float2 range = Range;
    for (uint k = li; k < kSize * kSize; k += kThreads)
    {
        const uint2 p = uint2(k % kSize, k / kSize);
        Nearness[p] = saturate((Raw.Load(int3(p, 0)) - range.x) * range.y);
    }
}
