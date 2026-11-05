#include "Common.hlsli"

// Mean and standard deviation of one frame's luma, measured on a small
// pyramid level (LevelWidth x LevelHeight is that level). One 256-thread
// group; the result is a single R32G32_FLOAT texel that PackLumaCS reads to
// match the brightness of a frame pair (see there). A coarse level is
// plenty: the statistics describe the whole frame, and they are the same at
// any level for the global gain/offset changes they exist to cancel.
Texture2D<float> Luma : register(t0);
RWTexture2D<float2> Stats : register(u0);

static const uint kThreads = 256;
groupshared float2 Partial[kThreads];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID)
{
    const uint count = LevelWidth * LevelHeight;
    float sum = 0.0f, sumSq = 0.0f;
    for (uint i = tid.x; i < count; i += kThreads)
    {
        const float v = Luma.Load(int3(i % LevelWidth, i / LevelWidth, 0));
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
        const float variance = max(0.0f, Partial[0].y / (float)count - mean * mean);
        Stats[uint2(0, 0)] = float2(mean, sqrt(variance));
    }
}
