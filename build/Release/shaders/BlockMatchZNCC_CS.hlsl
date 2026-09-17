#include "Common.hlsli"

Texture2D<float>  RefImage        : register(t0);
Texture2D<float>  CandImage       : register(t1);
Texture2D<float2> CoarsePredictor : register(t2);

RWTexture2D<float2> OutputFlow : register(u0);
RWTexture2D<float>  OutputZNCC : register(u1);

// A 16x16 group shares its reference pixels plus a one-pixel matching halo.
groupshared float RefTile[18][18];

[numthreads(16, 16, 1)]
void CSMain(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    int2 groupBase = int2(groupId.xy) * 16;
    for (uint tileY = groupThreadId.y; tileY < 18; tileY += 16)
    {
        for (uint tileX = groupThreadId.x; tileX < 18; tileX += 16)
        {
            int2 source = clamp(
                groupBase + int2(tileX, tileY) - 1,
                int2(0, 0),
                int2(LevelWidth, LevelHeight) - 1);
            RefTile[tileY][tileX] = RefImage.Load(int3(source, 0));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (dispatchThreadId.x >= LevelWidth || dispatchThreadId.y >= LevelHeight)
        return;

    int2 coord = int2(dispatchThreadId.xy);
    int2 local = int2(groupThreadId.xy) + 1;
    float2 normCoord = (float2(coord) + 0.5f) / float2(LevelWidth, LevelHeight);

    int2 predictor = int2(0, 0);
    if (LevelIndex < TotalLevels - 1)
    {
        predictor = int2(round(CoarsePredictor.SampleLevel(
            LinearClamp, normCoord, 0).xy));
    }

    float templateCenter = RefTile[local.y][local.x];
    float bestCost = 1e30f;
    int2 bestDisp = predictor;

    // Census/SAD is robust to moderate brightness changes without the costly
    // means, variances, square roots, and divisions required by ZNCC.
    for (int searchY = -SearchRadius; searchY <= SearchRadius; ++searchY)
    {
        for (int searchX = -SearchRadius; searchX <= SearchRadius; ++searchX)
        {
            int2 displacement = predictor + int2(searchX, searchY);
            int2 candidateCenterCoord = clamp(
                coord + displacement,
                int2(0, 0),
                int2(LevelWidth, LevelHeight) - 1);
            float candidateCenter = CandImage.Load(int3(candidateCenterCoord, 0));
            float cost = 0.0f;

            [unroll]
            for (int blockY = -1; blockY <= 1; ++blockY)
            {
                [unroll]
                for (int blockX = -1; blockX <= 1; ++blockX)
                {
                    float reference = RefTile[local.y + blockY][local.x + blockX];
                    int2 candidateCoord = clamp(
                        coord + displacement + int2(blockX, blockY),
                        int2(0, 0),
                        int2(LevelWidth, LevelHeight) - 1);
                    float candidate = CandImage.Load(int3(candidateCoord, 0));

                    float weight = (blockX == 0 && blockY == 0) ? 4.0f :
                                   ((blockX == 0 || blockY == 0) ? 2.0f : 1.0f);
                    float censusMismatch =
                        ((reference >= templateCenter) != (candidate >= candidateCenter)) ? 1.0f : 0.0f;
                    cost += weight * (abs(reference - candidate) + censusMismatch * 6.0f);
                }
            }

            int2 deviation = displacement - predictor;
            cost += SmoothnessWeight * 255.0f *
                    float(deviation.x * deviation.x + deviation.y * deviation.y);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestDisp = displacement;
            }
        }
    }

    // Eight inexpensive bilinear checks retain sub-pixel stability.
    float2 finalDisp = float2(bestDisp);
    float2 invDims = 1.0f / float2(LevelWidth, LevelHeight);
    [unroll]
    for (int subY = -1; subY <= 1; ++subY)
    {
        [unroll]
        for (int subX = -1; subX <= 1; ++subX)
        {
            if (subX == 0 && subY == 0) continue;
            float2 candidateDisp = float2(bestDisp) + float2(subX, subY) * 0.5f;
            float cost = 0.0f;

            [unroll]
            for (int blockY = -1; blockY <= 1; ++blockY)
            {
                [unroll]
                for (int blockX = -1; blockX <= 1; ++blockX)
                {
                    float reference = RefTile[local.y + blockY][local.x + blockX];
                    float2 sampleCoord = float2(coord + int2(blockX, blockY)) + candidateDisp;
                    float candidate = CandImage.SampleLevel(
                        LinearClamp, (sampleCoord + 0.5f) * invDims, 0);
                    float weight = (blockX == 0 && blockY == 0) ? 4.0f :
                                   ((blockX == 0 || blockY == 0) ? 2.0f : 1.0f);
                    cost += weight * abs(reference - candidate);
                }
            }

            if (cost < bestCost)
            {
                bestCost = cost;
                finalDisp = candidateDisp;
            }
        }
    }

    OutputFlow[coord] = finalDisp;
    OutputZNCC[coord] = saturate(1.0f - bestCost / (16.0f * 255.0f));
}
