#include "Common.hlsli"

Texture2D<int2> InputFlow : register(t0);
RWTexture2D<int2> OutputFlow : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (LevelWidth + 7) / 8;
    uint flowHeight = (LevelHeight + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;

    int2 vectors[9];
    uint index = 0;
    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            int2 coord = clamp(int2(id.xy) + int2(x, y), int2(0, 0),
                               int2(flowWidth, flowHeight) - 1);
            vectors[index++] = InputFlow.Load(int3(coord, 0));
        }
    }

    uint bestDistance = 0xffffffffu;
    int2 bestVector = vectors[4];
    [unroll]
    for (uint candidate = 0; candidate < 9; ++candidate)
    {
        uint distance = 0;
        [unroll]
        for (uint neighbor = 0; neighbor < 9; ++neighbor)
        {
            int2 delta = vectors[candidate] - vectors[neighbor];
            distance += delta.x * delta.x + delta.y * delta.y;
        }
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestVector = vectors[candidate];
        }
    }
    OutputFlow[id.xy] = bestVector;
}
