#include "Common.hlsli"

// The finest level's dense DIS flow, reduced to the flow ring's block grid
// (BlockSize x BlockSize pixels per texel) as the consumers expect:
// xy = the mean vector over the block, zw = full confidence (DIS has no
// separate measured/inherited distinction; FillFlowCS is a SAD-path pass).
Texture2D<float2> DenseFlow : register(t0);
RWTexture2D<float4> RingFlow : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 maxP = int2(Width, Height) - 1;
    const int2 origin = int2(id.xy) * BlockSize;
    float2 sum = float2(0.0f, 0.0f);
    for (int y = 0; y < BlockSize; ++y)
        for (int x = 0; x < BlockSize; ++x)
            sum += DenseFlow.Load(int3(min(origin + int2(x, y), maxP), 0));
    RingFlow[id.xy] = float4(sum / (float)(BlockSize * BlockSize), 1.0f, 1.0f);
}
