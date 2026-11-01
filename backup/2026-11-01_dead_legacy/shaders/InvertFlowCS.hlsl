#include "Common.hlsli"

Texture2D<int2> ForwardFlow : register(t0);
RWTexture2D<int2> BackwardFlow : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (LevelWidth + 7) / 8;
    uint flowHeight = (LevelHeight + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;

    int2 backward = -ForwardFlow.Load(int3(id.xy, 0));
    float2 blockCenter = float2(id.xy) * 8.0f + 4.0f;
    [unroll]
    for (uint iteration = 0; iteration < 3; ++iteration)
    {
        int2 sourceBlock = clamp(int2((blockCenter + float2(backward)) / 8.0f), int2(0, 0),
                                 int2(flowWidth, flowHeight) - 1);
        backward = -ForwardFlow.Load(int3(sourceBlock, 0));
    }
    BackwardFlow[id.xy] = backward;
}
