#include "Common.hlsli"

// Spatial cleanup of the flow grid at one level.
//
//   InputFlow    raw output of MotionSearchCS for this level
//   FirstImage   frame0 luma at this level
//   SecondImage  frame1 luma at this level
//   OutputFlow   cleaned flow
//
// Each block considers its own vector and its eight neighbours' vectors. A
// local consensus pulls isolated unstable vectors toward surrounding motion
// without rescanning the image patch.
Texture2D<float2> InputFlow : register(t0);
Texture2D<float> FirstImage : register(t1);
Texture2D<float> SecondImage : register(t2);
RWTexture2D<float2> OutputFlow : register(u0);

// A neighbour's vector must beat the block's own by this much; ties and
// noise-level differences keep the block's own measurement.
static const float kConsensusRadius = 3.5f;
static const float kStaticScoreMargin = 0.005f;
static const float kStaticMotionThreshold = 1.0f;
static const float kStaticConsensusThreshold = 0.75f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(LevelWidth, LevelHeight, BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    int2 gridMax = int2(grid) - 1;
    int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    float2 own = InputFlow.Load(int3(id.xy, 0));

    // Only the finest level needs the image-based static test. This catches
    // locally consistent but incorrect HUD vectors without multiplying the
    // expensive block score across every pyramid level.
    if (LevelIndex == 0 && length(own) > kStaticMotionThreshold)
    {
        RefArea ref = ChooseReferenceArea(FirstImage, int2(id.xy) * BlockSize, levelMax, BlockSize);
        float ownScore = ScoreCandidate(FirstImage, SecondImage, ref,
                                        int2(round(own)), levelMax);
        float zeroScore = ScoreCandidate(FirstImage, SecondImage, ref,
                                         int2(0, 0), levelMax);
        if (zeroScore >= ownScore - kStaticScoreMargin)
        {
            OutputFlow[id.xy] = float2(0.0f, 0.0f);
            return;
        }
    }

    float2 vectors[9];

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        int2 offset = int2(i % 3 - 1, i / 3 - 1);
        int2 coord = clamp(int2(id.xy) + offset, int2(0, 0), gridMax);
        vectors[i] = InputFlow.Load(int3(coord, 0));
    }

    float bestClusterCost = 1e30f;
    float2 consensus = own;
    [unroll]
    for (int candidateIndex = 0; candidateIndex < 9; ++candidateIndex)
    {
        float clusterCost = 0.0f;
        [unroll]
        for (int memberIndex = 0; memberIndex < 9; ++memberIndex)
            clusterCost += length(vectors[candidateIndex] - vectors[memberIndex]);
        if (clusterCost < bestClusterCost)
        {
            bestClusterCost = clusterCost;
            consensus = vectors[candidateIndex];
        }
    }

    float disagreement = length(own - consensus);
    if (disagreement > kConsensusRadius)
    {
        float2 stable = length(consensus) < kStaticConsensusThreshold
            ? float2(0.0f, 0.0f) : consensus;
        OutputFlow[id.xy] = stable;
    }
    else
    {
        OutputFlow[id.xy] = own;
    }
}
