#include "Common.hlsli"

// The AI pyramid flow network's input (FlowEngine::recordAiFlow): both
// frames of the pair (Width x Height) averaged down to the network
// resolution (LevelWidth x LevelHeight), RGB 0..1 as the trainer saw them,
// laid out planar - frame0 R, G, B, then frame1 R, G, B, each LevelHeight
// rows - in a LevelWidth x 6*LevelHeight R32_FLOAT texture whose rows,
// copied into a buffer, are exactly the [1, 6, H, W] tensor.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
RWTexture2D<float> Planar : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const uint2 source = uint2(Width, Height);
    const uint2 size = uint2(LevelWidth, LevelHeight);
    const uint2 from = id.xy * source / size;
    const uint2 to = max(from + 1u, (id.xy + 1u) * source / size);
    float3 sum0 = float3(0.0f, 0.0f, 0.0f);
    float3 sum1 = float3(0.0f, 0.0f, 0.0f);
    for (uint y = from.y; y < to.y; ++y)
        for (uint x = from.x; x < to.x; ++x)
        {
            sum0 += Frame0.Load(int3(x, y, 0)).rgb;
            sum1 += Frame1.Load(int3(x, y, 0)).rgb;
        }
    const float count = float((to.x - from.x) * (to.y - from.y));
    const float3 a = sum0 / count;
    const float3 b = sum1 / count;
    Planar[uint2(id.x, id.y)] = a.r;
    Planar[uint2(id.x, id.y + LevelHeight)] = a.g;
    Planar[uint2(id.x, id.y + 2u * LevelHeight)] = a.b;
    Planar[uint2(id.x, id.y + 3u * LevelHeight)] = b.r;
    Planar[uint2(id.x, id.y + 4u * LevelHeight)] = b.g;
    Planar[uint2(id.x, id.y + 5u * LevelHeight)] = b.b;
}
