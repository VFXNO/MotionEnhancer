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
// without rescanning the image patch. There is deliberately no image-based
// "is it really static" test here any more: MotionSearchCS already scores
// a zero-centred window at every level and only prefers it when it is
// strictly better, so a vector that survives the search is kept as motion.
Texture2D<float2> InputFlow : register(t0);
Texture2D<float> FirstImage : register(t1);
Texture2D<float> SecondImage : register(t2);
RWTexture2D<float2> OutputFlow : register(u0);

// A neighbour's vector must beat the block's own by this much; ties and
// noise-level differences keep the block's own measurement. The consensus
// vector replaces the outlier as-is; only sub-half-pixel clusters are
// treated as measurement noise and snapped to zero. Real slow motion
// (>= 0.5 px per pair, ~15 px/s at 30 fps) is never suppressed.
static const float kConsensusRadius = 3.5f;
static const float kNoiseClusterThreshold = 0.4f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(LevelWidth, LevelHeight, BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    int2 gridMax = int2(grid) - 1;
    float2 own = InputFlow.Load(int3(id.xy, 0));

    // MotionSearchCS diagnostics mode (SmoothnessWeight < 0): pass through.
    if (SmoothnessWeight < 0.0f)
    {
        OutputFlow[id.xy] = own;
        return;
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
        OutputFlow[id.xy] = length(consensus) < kNoiseClusterThreshold
            ? float2(0.0f, 0.0f) : consensus;
    }
    else
    {
        OutputFlow[id.xy] = own;
    }
}
