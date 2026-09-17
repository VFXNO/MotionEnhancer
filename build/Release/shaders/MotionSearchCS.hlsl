#include "Common.hlsli"

Texture2D<float> RefImage : register(t0);
Texture2D<float> CandImage : register(t1);
Texture2D<int2> CoarsePredictor : register(t2);
RWTexture2D<int2> OutputFlow : register(u0);

// D3D11/SM5 port of FidelityFX Optical Flow v5's 8x8 block search.
// Each 64-thread group evaluates 256 offsets for four 8x8 blocks.
groupshared uint RefRows[8][2];
groupshared uint SearchRows[144];
groupshared uint CandidateMin[64];
groupshared int2 SharedPredictor;

uint PackLuma(int2 position)
{
    int y = clamp(position.y, 0, int(LevelHeight) - 1);
    uint packed = 0;
    [unroll]
    for (int x = 0; x < 4; ++x)
    {
        int px = clamp(position.x + x, 0, int(LevelWidth) - 1);
        uint value = max(1u, (uint)round(RefImage.Load(int3(px, y, 0))));
        packed |= min(value, 255u) << (x * 8);
    }
    return packed;
}

uint PackCandidateLuma(int2 position)
{
    int y = clamp(position.y, 0, int(LevelHeight) - 1);
    uint packed = 0;
    [unroll]
    for (int x = 0; x < 4; ++x)
    {
        int px = clamp(position.x + x, 0, int(LevelWidth) - 1);
        uint value = max(1u, (uint)round(CandImage.Load(int3(px, y, 0))));
        packed |= min(value, 255u) << (x * 8);
    }
    return packed;
}

uint EncodeCandidate(uint cost, int2 offset, uint searchX, uint searchY)
{
    uint centerDistance = (uint)(abs(offset.x) + abs(offset.y));
    return (min(cost, 65535u) << 16) | (centerDistance << 8) |
           (searchY << 4) | searchX;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    uint flowWidth = (LevelWidth + 7) / 8;
    uint flowHeight = (LevelHeight + 7) / 8;
    uint searchGroupX = localIndex & 3;
    uint searchY = localIndex >> 2;

    [unroll]
    for (uint blockY = 0; blockY < 2; ++blockY)
    {
        [unroll]
        for (uint blockX = 0; blockX < 2; ++blockX)
        {
            uint2 flowCoord = groupId.xy * 2 + uint2(blockX, blockY);
            bool validBlock = flowCoord.x < flowWidth && flowCoord.y < flowHeight;
            int2 blockOrigin = int2(flowCoord) * 8;

            if (localIndex == 0)
            {
                SharedPredictor = (validBlock && LevelIndex < TotalLevels - 1)
                    ? CoarsePredictor.Load(int3(flowCoord, 0))
                    : int2(0, 0);
            }
            GroupMemoryBarrierWithGroupSync();

            if (localIndex < 16)
            {
                uint row = localIndex >> 1;
                uint column = localIndex & 1;
                RefRows[row][column] = PackLuma(blockOrigin + int2(column * 4, row));
            }

            int2 searchOrigin = blockOrigin + SharedPredictor - int2(8, 8);
            for (uint index = localIndex; index < 144; index += 64)
            {
                uint packedX = index % 6;
                uint row = index / 6;
                SearchRows[index] = PackCandidateLuma(
                    searchOrigin + int2(packedX * 4, row));
            }
            GroupMemoryBarrierWithGroupSync();

            uint4 costs = uint4(0, 0, 0, 0);
            [unroll]
            for (uint row = 0; row < 8; ++row)
            {
                uint sourceIndex = (searchY + row) * 6 + searchGroupX;
                costs = msad4(
                    RefRows[row][0],
                    uint2(SearchRows[sourceIndex], SearchRows[sourceIndex + 1]),
                    costs);
                costs = msad4(
                    RefRows[row][1],
                    uint2(SearchRows[sourceIndex + 1], SearchRows[sourceIndex + 2]),
                    costs);
            }

            uint best = 0xffffffffu;
            [unroll]
            for (uint lane = 0; lane < 4; ++lane)
            {
                uint sx = searchGroupX * 4 + lane;
                int2 offset = int2(sx, searchY) - int2(8, 8);
                bool enabled = abs(offset.x) <= SearchRadius && abs(offset.y) <= SearchRadius;
                uint cost = enabled ? costs[lane] : 65535u;
                uint smoothness = (uint)(SmoothnessWeight * 255.0f * dot(offset, offset));
                best = min(best, EncodeCandidate(cost + smoothness, offset, sx, searchY));
            }
            CandidateMin[localIndex] = best;
            GroupMemoryBarrierWithGroupSync();

            [unroll]
            for (uint stride = 32; stride > 0; stride >>= 1)
            {
                if (localIndex < stride)
                    CandidateMin[localIndex] = min(CandidateMin[localIndex], CandidateMin[localIndex + stride]);
                GroupMemoryBarrierWithGroupSync();
            }

            if (localIndex == 0 && validBlock)
            {
                uint encoded = CandidateMin[0];
                int2 correction = int2(encoded & 15u, (encoded >> 4) & 15u) - int2(8, 8);
                int2 result = SharedPredictor + correction;

                OutputFlow[flowCoord] = result;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
