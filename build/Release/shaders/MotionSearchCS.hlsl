#include "Common.hlsli"

// MSAD64 block matcher. One threadgroup per 32x32 flow block: the C++ side
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
// Every flow block is matched using 64 samples distributed across its support
// region. The support can span up to 64x64 pixels on coarse levels.
Texture2D<float> RefImage : register(t0);
Texture2D<float> CandImage : register(t1);
Texture2D<float2> CoarsePredictor : register(t2);
RWTexture2D<float2> OutputFlow : register(u0);

static const int kMaxRadius = 8;                          // GPUInterpolator clamps SearchRadius to 0..8
static const int kThreads = 64;

// Near-tie margin between the predictor and zero windows; see the final
// selection for which side wins a near-tie.
// Deterministic tie-break toward the smaller offset within a window,
// independent of the user-controlled SmoothnessWeight (which may be zero).
static const float kTieBreak = 1e-5f;

groupshared float RefTile[kContext][kContext];
groupshared float RefMean;
groupshared float RefStd;
groupshared int RefSize;      // 16, 32, or 64 depending on BlockSize
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
    // Map child-grid coordinates into the parent grid. Pyramid pixels are 2x
    // larger at the parent level, so the level scale is part of this ratio.
    float ratio = (float)BlockSize / max(1.0f, (float)ParentBlockSize * 2.0f);
    float2 p = float2(childBlock) * ratio;
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

// Scores candidate offsets using direct texture loads to keep 64x64 support
// within the D3D11 shared-memory limit.
// Returns the best (score, absolute vector) pair for this thread.
void SearchWindow(float2 base, int radius, int size, int refOffset,
                  float refMean, float refStd, int2 refOrigin, int2 levelMax,
                  uint li, bool recordBase,
                  out float bestScore, out float2 bestVector)
{
    int support = min(size, 8);
    int sampleStep = max(1, size / 8);
    float n = (float)(support * support);
    float refSum = 0.0f;
    [loop]
    for (int y = 0; y < support; ++y)
    {
        [loop]
        for (int x = 0; x < support; ++x)
            refSum += RefTile[refOffset + y * sampleStep][refOffset + x * sampleStep];
    }
    float refMeanLocal = refSum / n;
    int span = 2 * radius + 1;
    uint candidateCount = (uint)(span * span);
    bestScore = -1e30f;
    bestVector = base;

    [loop]
    for (uint c = li; c < candidateCount; c += kThreads)
    {
        int ox = (int)(c % (uint)span) - radius;
        int oy = (int)(c / (uint)span) - radius;
        float candSum = 0.0f;
        [loop]
        for (int y = 0; y < support; ++y)
        {
            [loop]
            for (int x = 0; x < support; ++x)
            {
                int2 p = clamp(refOrigin + int2(round(base)) + int2(ox, oy) +
                               int2(x * sampleStep, y * sampleStep), int2(0, 0), levelMax);
                float cv = CandImage.Load(int3(p, 0));
                candSum += cv;
            }
        }
        float candMean = candSum / n;
        float sad = 0.0f;
        [loop]
        for (int y = 0; y < support; ++y)
        {
            [loop]
            for (int x = 0; x < support; ++x)
            {
                float rv = RefTile[refOffset + y * sampleStep][refOffset + x * sampleStep] - refMeanLocal;
                int2 p = clamp(refOrigin + int2(round(base)) + int2(ox, oy) +
                               int2(x * sampleStep, y * sampleStep), int2(0, 0), levelMax);
                float cv = CandImage.Load(int3(p, 0)) - candMean;
                sad += abs(rv - cv);
            }
        }

        float score = 1.0f - sad / (n * kLumaScale);

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
    int2 blockOrigin = int2(groupId.xy) * BlockSize;
    int matchSize = kContext;
    int contextPad = (matchSize - BlockSize) / 2;
    int2 contextOrigin = blockOrigin - int2(contextPad, contextPad);

    // --- Stage 1: predictor + reference context into shared memory ---------
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
    for (int j = 0; j < kContext / 8; ++j)
        [unroll]
        for (int i = 0; i < kContext / 8; ++i)
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
        RefStats(matchSize, 0, mean, std);
        RefMean = mean;
        RefStd = std;
        RefSize = matchSize;
        PredictorScore = -1e30f;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 predictor = Predictor;
    // A zero-centered recovery search lets a bad coarse predictor return to
    // static motion at finer levels. It is skipped when the predictor is zero.
    bool zeroDistinct = any(abs(predictor) > 1e-4f);
    int size = RefSize;
    int refOffset = 0;
    int2 refOrigin = contextOrigin;
    float refMean = RefMean;
    float refStd = RefStd;

    // --- Stage 3: search around the predictor -----------------------------
    float scoreP;
    float2 vectorP;
    SearchWindow(predictor, radius, size, refOffset, refMean, refStd,
                 refOrigin, levelMax, li, false, scoreP, vectorP);

    // --- Stage 4: search around zero (skipped when it is the same window) --
    float scoreZ = -1e30f;
    float2 vectorZ = float2(0.0f, 0.0f);
    if (zeroDistinct)
    {
        SearchWindow(float2(0.0f, 0.0f), radius, size, refOffset, refMean, refStd,
                     refOrigin, levelMax, li, false, scoreZ, vectorZ);
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

    // Subpixel refinement is disabled; keep the integer MSAD result.
    #if 0
    // Refine the selected integer vector at half-pixel offsets. Each thread
    // evaluates one of the eight surrounding candidates, then the group
    // reduction selects the best float displacement.
    static const float2 kHalfOffsets[8] = {
        float2(-0.5f, -0.5f), float2(0.0f, -0.5f), float2(0.5f, -0.5f),
        float2(-0.5f,  0.0f),                         float2(0.5f,  0.0f),
        float2(-0.5f,  0.5f), float2(0.0f,  0.5f), float2(0.5f,  0.5f)
    };
    float subBestScore = RefinementBaseScore;
    float2 subBestVector = RefinementBase;
    if (li < 8)
    {
        float2 candidate = RefinementBase + kHalfOffsets[li];
        float score = ScoreSubpixel(candidate, refOrigin, levelMax);
        if (score > subBestScore)
        {
            subBestScore = score;
            subBestVector = candidate;
        }
    }
    BestScore[0][li] = subBestScore;
    BestVector[0][li] = subBestVector;
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
    if (li == 0)
        OutputFlow[groupId.xy] = BestVector[0][0];

    // Second iterative pass: refine the selected half-pixel result at
    // quarter-pixel offsets. This narrows the search around the updated best
    // vector instead of repeating the broad integer search.
    if (li == 0)
    {
        RefinementBase = BestVector[0][0];
        RefinementBaseScore = BestScore[0][0];
    }
    GroupMemoryBarrierWithGroupSync();

    static const float2 kQuarterOffsets[8] = {
        float2(-0.25f, -0.25f), float2(0.0f, -0.25f), float2(0.25f, -0.25f),
        float2(-0.25f,  0.0f),                         float2(0.25f,  0.0f),
        float2(-0.25f,  0.25f), float2(0.0f, 0.25f), float2(0.25f, 0.25f)
    };
    float quarterBestScore = RefinementBaseScore;
    float2 quarterBestVector = RefinementBase;
    if (li < 8)
    {
        float2 candidate = RefinementBase + kQuarterOffsets[li];
        float score = ScoreSubpixel(candidate, refOrigin, levelMax);
        if (score > quarterBestScore)
        {
            quarterBestScore = score;
            quarterBestVector = candidate;
        }
    }
    BestScore[0][li] = quarterBestScore;
    BestVector[0][li] = quarterBestVector;
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
    if (li == 0)
        OutputFlow[groupId.xy] = BestVector[0][0];

    // Third iterative pass: refine the quarter-pixel result at 1/8-pixel
    // offsets. This runs at every pyramid level and preserves float vectors.
    if (li == 0)
    {
        RefinementBase = BestVector[0][0];
        RefinementBaseScore = BestScore[0][0];
    }
    GroupMemoryBarrierWithGroupSync();

    static const float2 kEighthOffsets[8] = {
        float2(-0.125f, -0.125f), float2(0.0f, -0.125f), float2(0.125f, -0.125f),
        float2(-0.125f,  0.0f),                          float2(0.125f,  0.0f),
        float2(-0.125f,  0.125f), float2(0.0f, 0.125f), float2(0.125f, 0.125f)
    };
    float eighthBestScore = RefinementBaseScore;
    float2 eighthBestVector = RefinementBase;
    if (li < 8)
    {
        float2 candidate = RefinementBase + kEighthOffsets[li];
        float score = ScoreSubpixel(candidate, refOrigin, levelMax);
        if (score > eighthBestScore)
        {
            eighthBestScore = score;
            eighthBestVector = candidate;
        }
    }
    BestScore[0][li] = eighthBestScore;
    BestVector[0][li] = eighthBestVector;
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
    if (li == 0)
        OutputFlow[groupId.xy] = BestVector[0][0];
    #endif

}
