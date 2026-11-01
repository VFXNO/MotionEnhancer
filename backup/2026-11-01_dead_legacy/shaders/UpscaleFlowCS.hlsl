#include "Common.hlsli"

Texture2D<int2> InputFlow : register(t0);
Texture2D<float> FirstImage : register(t1);
Texture2D<float> SecondImage : register(t2);
RWTexture2D<int2> OutputFlow : register(u0);

float BlockSad4x4(int2 origin, int2 motion)
{
    float sad = 0.0f;
    [unroll]
    for (int y = 0; y < 4; ++y)
    {
        [unroll]
        for (int x = 0; x < 4; ++x)
        {
            int2 reference = clamp(origin + int2(x, y), int2(0, 0), int2(LevelWidth, LevelHeight) - 1);
            int2 candidate = clamp(reference + motion, int2(0, 0), int2(LevelWidth, LevelHeight) - 1);
            sad += abs(FirstImage.Load(int3(reference, 0)) - SecondImage.Load(int3(candidate, 0)));
        }
    }
    return sad;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (LevelWidth + 7) / 8;
    uint flowHeight = (LevelHeight + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;

    uint coarseWidth;
    uint coarseHeight;
    InputFlow.GetDimensions(coarseWidth, coarseHeight);
    int2 baseCoord = int2(id.xy) / 2;
    int2 parity = int2(id.xy) & 1;
    int2 offsets[4] = {
        int2(parity.x - 1, parity.y - 1),
        int2(parity.x, parity.y - 1),
        int2(parity.x - 1, parity.y),
        int2(parity.x, parity.y)
    };

    float bestSad = 1e30f;
    int2 bestVector = int2(0, 0);
    int2 lumaOrigin = int2(id.xy) * 4;
    [unroll]
    for (uint candidateIndex = 0; candidateIndex < 4; ++candidateIndex)
    {
        int2 source = clamp(baseCoord + offsets[candidateIndex], int2(0, 0),
                            int2(coarseWidth, coarseHeight) - 1);
        int2 coarseVector = InputFlow.Load(int3(source, 0));
        float sad = BlockSad4x4(lumaOrigin, coarseVector);
        if (sad < bestSad)
        {
            bestSad = sad;
            bestVector = coarseVector * 2;
        }
    }
    OutputFlow[id.xy] = bestVector;
}
