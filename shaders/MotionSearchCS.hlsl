#include "Common.hlsli"

// ZNCC block matcher. One threadgroup per 16x16 flow block: the C++ side
// dispatches exactly (flowWidth, flowHeight, 1) groups, so groupId.xy is the
// flow block coordinate at this level.
//
//   RefImage        frame0 luma at this level
//   CandImage       frame1 luma at this level
//   CoarsePredictor flow for this level, already upscaled from the coarser
//                   level (ignored at the coarsest level, where it is unbound)
//   OutputFlow      best vector, in this level's pixels
//
// Two search windows are evaluated: +-SearchRadius around the coarse
// predictor, and +-SearchRadius around the zero vector. Without the second
// one a wrong vector picked at a coarse level (where a block is hundreds of
// screen pixels and a moving video can dominate it) is unrecoverable, because
// every finer level can only move +-SearchRadius from what it inherited.
// Static content -- most of a desktop -- then gets warped by hundreds of
// pixels. The zero window gives every level a way back to "did not move".
//
// Every 16x16 flow block is matched using a fixed 16x16 support region.
Texture2D<float> RefImage : register(t0);
Texture2D<float> CandImage : register(t1);
Texture2D<float2> CoarsePredictor : register(t2);
RWTexture2D<float2> OutputFlow : register(u0);

static const int kMaxRadius = 8;                          // GPUInterpolator clamps SearchRadius to 0..8
static const int kMaxWindow = kContext + 2 * kMaxRadius; // kContext: Common.hlsli
static const int kThreads = 64;

// Near-tie margin between the predictor and zero windows; see the final
// selection for which side wins a near-tie.
// Deterministic tie-break toward the smaller offset within a window,
// independent of the user-controlled SmoothnessWeight (which may be zero).
static const float kTieBreak = 1e-5f;

groupshared float RefTile[kContext][kContext];
groupshared float CandWindow[kMaxWindow][kMaxWindow];
groupshared float RefMean;
groupshared float RefStd;
groupshared int RefSize;      // fixed 16
groupshared float2 Predictor;
groupshared float PredictorScore; // score of the predictor window at offset (0,0)
groupshared float BestScore[2][kThreads];
groupshared float2 BestVector[2][kThreads];
groupshared float2 RefinementBase;
groupshared float RefinementBaseScore;

float2 SampleParentFlow(Texture2D<float2> parent, uint2 childBlock)
{
    uint parentWidth, parentHeight;
    parent.GetDimensions(parentWidth, parentHeight);

    // Flow samples live at block-grid indices, not pixel centers. Two child
    // blocks correspond to one parent block, so child indices map directly
    // through a factor of 0.5. The first two child blocks must both use the
    // first parent vector rather than extrapolating around it.
    float2 p = float2(childBlock) * 0.5f;
    int2 p0 = int2(floor(p));
    int2 maxCoord = int2(parentWidth, parentHeight) - 1;
    p = clamp(p, 0.0f, float2(maxCoord));
    p0 = int2(floor(p));
    float2 f = p - float2(p0);
    int2 c00 = clamp(p0, int2(0, 0), maxCoord);
    int2 c10 = clamp(p0 + int2(1, 0), int2(0, 0), maxCoord);
    int2 c01 = clamp(p0 + int2(0, 1), int2(0, 0), maxCoord);
    int2 c11 = clamp(p0 + int2(1, 1), int2(0, 0), maxCoord);

    float2 v00 = parent.Load(int3(c00, 0));
    float2 v10 = parent.Load(int3(c10, 0));
    float2 v01 = parent.Load(int3(c01, 0));
    float2 v11 = parent.Load(int3(c11, 0));
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y) * 2.0f;
}

void RefStats(int size, int offset, out float mean, out float std)
{
    float sum = 0.0f, sumSq = 0.0f;
    [loop]
    for (int y = 0; y < size; ++y)
        [loop]
        for (int x = 0; x < size; ++x)
        {
            float v = RefTile[offset + y][offset + x];
            sum += v;
            sumSq += v * v;
        }
    float n = (float)(size * size);
    mean = sum / n;
    float var = max(sumSq / n - mean * mean, 0.0f);
    std = sqrt(var);
}

void LoadWindow(float2 base, int radius, int size, int2 refOrigin, int2 levelMax, uint li)
{
    int windowSize = size + 2 * radius;
    int2 windowOrigin = refOrigin + int2(round(base)) - int2(radius, radius);
    uint windowTexels = (uint)(windowSize * windowSize);
    [loop]
    for (uint idx = li; idx < windowTexels; idx += kThreads)
    {
        int wx = (int)(idx % (uint)windowSize);
        int wy = (int)(idx / (uint)windowSize);
        int2 p = clamp(windowOrigin + int2(wx, wy), int2(0, 0), levelMax);
        CandWindow[wy][wx] = CandImage.Load(int3(p, 0));
    }
}

// Scores this thread's strided slice of the offsets in the loaded window.
// Returns the best (score, absolute vector) pair for this thread.
void SearchWindow(float2 base, int radius, int size, int refOffset,
                  float refMean, float refStd, uint li, bool recordBase,
                  out float bestScore, out float2 bestVector)
{
    bool refFlat = refStd < kFlatStd;
    float n = (float)(size * size);
    int span = 2 * radius + 1;
    uint candidateCount = (uint)(span * span);
    bestScore = -1e30f;
    bestVector = base;

    [loop]
    for (uint c = li; c < candidateCount; c += kThreads)
    {
        int ox = (int)(c % (uint)span) - radius;
        int oy = (int)(c / (uint)span) - radius;
        int2 w = int2(radius + ox, radius + oy); // window-local top-left of this candidate

        float sumC = 0.0f, sumC2 = 0.0f, sumRC = 0.0f;
        [loop]
        for (int y = 0; y < size; ++y)
        {
            [loop]
            for (int x = 0; x < size; ++x)
            {
                float cv = CandWindow[w.y + y][w.x + x];
                sumC += cv;
                sumC2 += cv * cv;
                sumRC += RefTile[refOffset + y][refOffset + x] * cv;
            }
        }

        float meanC = sumC / n;
        float stdC = sqrt(max(sumC2 / n - meanC * meanC, 0.0f));
        bool candFlat = stdC < kFlatStd;

        float score;
        if (refFlat && candFlat)
        {
            float diff = abs(refMean - meanC);
            score = (diff < 5.0f) ? (1.0f - diff * 0.05f) : -1.0f;
        }
        else if (refFlat || candFlat)
        {
            score = -1.0f;
        }
        else
        {
            float cov = sumRC / n - refMean * meanC;
            score = clamp(cov / (refStd * stdC + 1e-6f), -1.0f, 1.0f);
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestVector = base + float2(ox, oy);
        }
    }
}

[numthreads(8, 8, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    int radius = clamp(SearchRadius, 0, kMaxRadius);
    int2 blockOrigin = int2(groupId.xy) * kFlowBlock;
    int2 contextOrigin = blockOrigin - int2(kContextPad, kContextPad);

    // --- Stage 1: predictor + 16x16 reference context into shared memory --
    if (li == 0)
    {
        // Pyramid dimensions halve each level. Propagate the parent flow
        // directly as the fine-level predictor; the separate upscale shader
        // is unnecessary for the block-grid search.
        Predictor = (LevelIndex < TotalLevels - 1)
            ? SampleParentFlow(CoarsePredictor, groupId.xy)
            : float2(0.0f, 0.0f);
    }
    [unroll]
    for (int j = 0; j < 2; ++j)
        [unroll]
        for (int i = 0; i < 2; ++i)
        {
            int2 local = int2(tid.xy) + int2(i, j) * 8;
            int2 p = clamp(contextOrigin + local, int2(0, 0), levelMax);
            RefTile[local.y][local.x] = RefImage.Load(int3(p, 0));
        }
    GroupMemoryBarrierWithGroupSync();

    // --- Stage 2: reference statistics for the actual flow block ------------
    if (li == 0)
    {
        float mean, std;
        RefStats(kContext, 0, mean, std);
        RefMean = mean;
        RefStd = std;
        RefSize = kContext;
        PredictorScore = -1e30f;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 predictor = Predictor;
    bool zeroDistinct = any(abs(predictor) > 1e-4f);
    int size = RefSize;
    int refOffset = 0;
    int2 refOrigin = contextOrigin;
    float refMean = RefMean;
    float refStd = RefStd;

    // --- Stage 3: search around the predictor -----------------------------
    LoadWindow(predictor, radius, size, refOrigin, levelMax, li);
    GroupMemoryBarrierWithGroupSync();
    float scoreP;
    float2 vectorP;
    SearchWindow(predictor, radius, size, refOffset, refMean, refStd, li, false,
                 scoreP, vectorP);
    GroupMemoryBarrierWithGroupSync(); // everyone is done reading CandWindow

    // --- Stage 4: search around zero (skipped when it is the same window) --
    float scoreZ = -1e30f;
    float2 vectorZ = float2(0.0f, 0.0f);
    if (zeroDistinct)
    {
        LoadWindow(float2(0.0f, 0.0f), radius, size, refOrigin, levelMax, li);
    }
    GroupMemoryBarrierWithGroupSync();
    if (zeroDistinct)
    {
        SearchWindow(float2(0.0f, 0.0f), radius, size, refOffset, refMean, refStd, li, false,
                     scoreZ, vectorZ);
    }

    BestScore[0][li] = scoreP;
    BestVector[0][li] = vectorP;
    BestScore[1][li] = scoreZ;
    BestVector[1][li] = vectorZ;
    GroupMemoryBarrierWithGroupSync();

    // --- Stage 5: reduce both searches to their group-wide best -----------
    [unroll]
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (li < stride)
        {
            [unroll]
            for (int s = 0; s < 2; ++s)
            {
                if (BestScore[s][li + stride] > BestScore[s][li])
                {
                    BestScore[s][li] = BestScore[s][li + stride];
                    BestVector[s][li] = BestVector[s][li + stride];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (li == 0)
    {
        float bestP = BestScore[0][0];
        float bestZ = zeroDistinct ? BestScore[1][0] : -1e30f;
        bool useZero = zeroDistinct && bestZ > bestP;
        RefinementBase = useZero ? BestVector[1][0] : BestVector[0][0];
        RefinementBaseScore = useZero ? bestZ : bestP;
        OutputFlow[groupId.xy] = RefinementBase;
    }
    GroupMemoryBarrierWithGroupSync();

    // Integer-only local correction around the selected vector. The broad
    // search remains stable while this centered 8x8 pass can correct a match
    // pulled toward a neighboring motion region.
    float fineMean, fineStd;
    RefStats(8, kSmallSupportPad, fineMean, fineStd);
    LoadWindow(RefinementBase, 1, 8, blockOrigin, levelMax, li);
    GroupMemoryBarrierWithGroupSync();
    float fineScore;
    float2 fineVector;
    SearchWindow(RefinementBase, 1, 8, kSmallSupportPad,
                 fineMean, fineStd, li, false, fineScore, fineVector);
    BestScore[0][li] = fineScore;
    BestVector[0][li] = fineVector;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (li < stride && BestScore[0][li + stride] > BestScore[0][li])
        {
            BestScore[0][li] = BestScore[0][li + stride];
            BestVector[0][li] = BestVector[0][li + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (li == 0 && BestScore[0][0] > RefinementBaseScore + 0.02f)
    {
        OutputFlow[groupId.xy] = BestVector[0][0];
    }
}
