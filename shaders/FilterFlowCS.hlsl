#include "Common.hlsli"

// Spatial cleanup of the flow grid at one level.
//
//   InputFlow    raw output of MotionSearchCS for this level
//   FirstImage   frame0 luma at this level
//   SecondImage  frame1 luma at this level
//   OutputFlow   cleaned flow
//
// Each block considers its own vector and its eight neighbours' vectors. A
// robust local consensus is used to pull isolated unstable vectors toward the
// surrounding motion while preserving coherent motion boundaries.
//
// It deliberately is not a vector median; each candidate is selected by its
// measured match score.
Texture2D<float2> InputFlow : register(t0);
Texture2D<float> FirstImage : register(t1);
Texture2D<float> SecondImage : register(t2);
RWTexture2D<float2> OutputFlow : register(u0);

// A neighbour's vector must beat the block's own by this much; ties and
// noise-level differences keep the block's own measurement.
static const float kConsensusRadius = 2.0f;
static const float kConsensusBlend = 0.75f;
static const float kStaticScoreMargin = 0.02f;
static const float kStaticMotionThreshold = 1.5f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(LevelWidth, LevelHeight);
    if (id.x >= grid.x || id.y >= grid.y) return;
    int2 gridMax = int2(grid) - 1;
    int2 levelMax = int2(LevelWidth, LevelHeight) - 1;

    float2 own = InputFlow.Load(int3(id.xy, 0));
    RefArea ref = ChooseReferenceArea(FirstImage, int2(id.xy) * kFlowBlock, levelMax);

    float ownScore = ScoreCandidate(FirstImage, SecondImage, ref, int2(round(own)), levelMax);
    float bestNeighborScore = -1e30f;
    float2 bestNeighbor = own;

    // Keep the best matching neighbour as a motion candidate and collect a
    // local consensus at the same time. The consensus is based on nearby
    // vectors, not image content, so one bad match cannot dominate it.
    float2 seen[9];
    seen[0] = own;
    int seenCount = 1;
    float2 consensusSum = own;
    int consensusCount = 1;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        if (i == 4) continue;
        int2 coord = clamp(int2(id.xy) + int2(i % 3 - 1, i / 3 - 1), int2(0, 0), gridMax);
        float2 candidate = InputFlow.Load(int3(coord, 0));
        consensusSum += candidate;
        consensusCount++;

        bool duplicate = false;
        [loop]
        for (int s = 0; s < seenCount; ++s)
        {
            if (all(abs(candidate - seen[s]) < 1e-4f)) { duplicate = true; break; }
        }
        if (duplicate) continue;
        seen[seenCount++] = candidate;

        float score = ScoreCandidate(FirstImage, SecondImage, ref, int2(round(candidate)), levelMax);
        if (score > bestNeighborScore)
        {
            bestNeighborScore = score;
            bestNeighbor = candidate;
        }
    }

    float2 consensus = consensusSum / max((float)consensusCount, 1.0f);
    float disagreement = length(own - consensus);
    if (disagreement > kConsensusRadius)
    {
        float zeroScore = ScoreCandidate(FirstImage, SecondImage, ref, int2(0, 0), levelMax);
        bool staticContent = length(own) > kStaticMotionThreshold &&
                             zeroScore >= ownScore - kStaticScoreMargin;

        // Prefer a matching neighbour when available, then blend it toward
        // the local consensus so isolated vectors follow stable motion rather
        // than creating a hard vector discontinuity.
        // Static content is forced to zero motion to prevent HUD/text warp.
        float2 stable = staticContent ? float2(0.0f, 0.0f) :
            (bestNeighborScore >= ownScore - 0.05f)
            ? bestNeighbor : consensus;
        OutputFlow[id.xy] = staticContent ? stable : lerp(own, stable, kConsensusBlend);
    }
    else
    {
        OutputFlow[id.xy] = own;
    }
}
