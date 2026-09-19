#include "Common.hlsli"

// Flow at level l+1 -> initial flow (predictor) at level l.
//
//   InputFlow    filtered flow of the coarser level (l+1), coarse pixels
//   FirstImage   frame0 luma at THIS level (l)
//   SecondImage  frame1 luma at THIS level (l)
//   LevelWidth/Height = size of THIS level (l)
//
// Fine blocks test the four nearest coarse vectors against the actual fine
// block after scaling to fine-level units.
Texture2D<int2> InputFlow : register(t0);
Texture2D<float> FirstImage : register(t1);
Texture2D<float> SecondImage : register(t2);
RWTexture2D<int2> OutputFlow : register(u0);

// A neighbouring coarse vector must beat the block's own coarse vector by
// this much. On an edge the neighbours differ by a pixel along the edge and
// score within noise of each other; taking the max there scatters the
// vectors along an edge, while preserving clear score improvements.
static const float kNeighbourMargin = 0.01f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(LevelWidth, LevelHeight);
    if (id.x >= grid.x || id.y >= grid.y) return;
    int2 levelMax = int2(LevelWidth, LevelHeight) - 1;

    uint coarseW, coarseH;
    InputFlow.GetDimensions(coarseW, coarseH);
    int2 coarseMax = int2(coarseW, coarseH) - 1;

    int2 baseCoord = int2(id.xy) / 2;
    int2 parity = int2(id.xy) & 1;
    // Nearest 2x2 coarse neighbourhood: own block plus the neighbour on the
    // side this fine block lies on.
    int2 side = parity * 2 - 1; // -1 or +1
    int2 offsets[3] = {
        int2(side.x, 0),
        int2(0, side.y),
        int2(side.x, side.y)
    };

    RefArea ref = ChooseReferenceArea(FirstImage, int2(id.xy) * kFlowBlock, levelMax);

    int2 ownVector = InputFlow.Load(int3(baseCoord, 0)) * 2;
    int2 bestVector = ownVector;
    float bestScore = ScoreCandidate(FirstImage, SecondImage, ref, bestVector, levelMax);
    int2 seen[4];
    seen[0] = ownVector;
    int seenCount = 1;
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        int2 source = clamp(baseCoord + offsets[i], int2(0, 0), coarseMax);
        int2 fineVector = InputFlow.Load(int3(source, 0)) * 2;
        bool duplicate = false;
        [loop]
        for (int s = 0; s < seenCount; ++s)
        {
            if (all(fineVector == seen[s])) { duplicate = true; break; }
        }
        if (duplicate) continue;
        seen[seenCount++] = fineVector;
        float score = ScoreCandidate(FirstImage, SecondImage, ref, fineVector, levelMax);
        if (score > bestScore + kNeighbourMargin)
        {
            bestScore = score;
            bestVector = fineVector;
        }
    }

    OutputFlow[id.xy] = bestVector;
}
