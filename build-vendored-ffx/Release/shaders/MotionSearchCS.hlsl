#include "Common.hlsli"

// msad4 block matcher with predictive candidate selection (SM 6.0: wave ops).
// One 8x8 thread group per 8x8 flow block; the C++ side dispatches exactly
// (flowWidth, flowHeight, 1) groups, so groupId.xy is the flow block
// coordinate at this level.
//
//   RefImage        frame0 luma at this level (R16_FLOAT, 0..255 scale)
//   CandPacked      frame1 luma at this level, 4 quantized bytes per texel
//                   (PackLumaCS): one load per word
//   CoarsePredictor filtered flow of the coarser level (unbound at the
//                   coarsest level)
//   TemporalFlow    filtered flow of the previous frame pair at THIS level
//                   (same units; bound when FlowScale > 0)
//   OutputFlow      best vector, in this level's pixels
//
// Hardware mapping, after AMD's FidelityFX Optical Flow:
//   * msad4(ref, src, acc) compares a 4-byte reference word with the four
//     4-byte windows of an 8-byte source pair and returns all four masked
//     SADs in one instruction (native on AMD; emulated with byte ops by
//     other vendors' drivers). Each reference quad scores FOUR horizontal
//     candidates at once; an 8x8 block against 4 candidates is 16 msad4.
//   * The candidate window is staged once in shared memory from the packed
//     luma texture (one load per word, word-aligned origin).
//   * "Reference byte 0 is ignored" is used as a free edge weighting: the
//     reference is split into two words per quad, one holding the samples
//     whose gradient is above the block's adaptive threshold (edges, full
//     weight) and one holding the rest (weight kSmoothWeight). A cel-shaded
//     character is matched by its outline, not by the background behind it,
//     while the low-weight remainder still breaks ties along straight edges.
//   * 8 px blocks (FFX's grid): a block straddling an object boundary is
//     four times less contaminated by background than a 16 px one.
//   * Reductions use wave intrinsics; the group is small and the kernel is
//     bound by latency, so barriers and shared-memory round trips are what
//     it must not spend.
//
// Candidates:
//   1. Centre selection: the parent predictor, the temporal predictor and
//      the eight neighbouring parent vectors (EPZS) are each scored once.
//   2. A +-SearchRadius window around the best centre.
//   3. A +-SearchRadius window around zero when zero is outside the first
//      window, so a bad inheritance can always return to "did not move".
//      Zero wins only when it is *strictly* better by kZeroMargin: on
//      ambiguous content ties go to motion, never to a stall.
//   Blocks with few edge samples pick among the centres only.
Texture2D<float> RefImage : register(t0);
Texture2D<uint> CandPacked : register(t1);
Texture2D<float2> CoarsePredictor : register(t2);
Texture2D<float2> TemporalFlow : register(t3);
RWTexture2D<float2> OutputFlow : register(u0);

static const int kBlock = 8;
static const int kQuads = kBlock / 4;                     // 2 reference words per row
static const int kThreads = 64;
static const int kMaxWaves = 4;                           // 64 threads at >= 16 lanes per wave
static const int kMaxRadius = 8;                          // SearchRadius is clamped to 0..8
static const int kSpanMax = 2 * kMaxRadius + 1;           // 17
// The strip origin is rounded down to a word boundary, so the candidate
// columns cover up to span + 3 bytes: 5 groups of 4 for span 17.
static const int kQuadGroupsMax = (kSpanMax + 3 + 3) / 4; // 5
static const int kStripRows = kSpanMax - 1 + kBlock;      // 24
static const int kStripWords = kQuadGroupsMax + kQuads + 1; // 8 packed words per row
static const int kTile = kBlock + 2;                      // reference block with a 1 px halo
static const int kCentres = 10;                           // parent, temporal, 8 neighbours
static const int kMinEdgeSamples = 6;
// One luma level of weighted mean absolute difference (score units).
static const float kZeroMargin = 1.0f / kLumaScale;
static const float kMinGradient = 2.0f;
static const float kGradientFraction = 0.5f;
// Weight of the below-threshold samples relative to edge samples.
static const float kSmoothWeight = 0.1f;
static const uint  kSmoothWeightFixed = 2u;               // x16 fixed point for the atomics
// (1/255) / (0.0005 * 64): the default SmoothnessWeight costs one luma
// level at 8 px: enough to settle ties, never enough to hold motion back.
static const float kSmoothnessScale = 0.1225f;

groupshared float  RefTile[kTile][kTile];                  // luma with halo (gradients)
groupshared uint   RefBytes[kBlock][kBlock];
groupshared uint   EdgeMask[kBlock][kBlock];
groupshared uint   RefWords[kBlock][kQuads];              // edge samples only (others 0)
groupshared uint   SmoothWords[kBlock][kQuads];           // remaining samples only
groupshared uint   Strip[kStripRows][kStripWords];
groupshared uint   CentreSad[kCentres];
groupshared int2   CentreVec[kCentres];
groupshared bool   CentreValid[kCentres];
groupshared float  WaveSum[kMaxWaves];
groupshared uint   WaveSumU[kMaxWaves];
groupshared float  WaveBest[kMaxWaves];
groupshared float2 WaveBestVector[kMaxWaves];
groupshared float2 Predictor;
groupshared float2 Temporal;

uint QuantizeLuma(float v)
{
    return min(255u, (uint)(clamp(v, 0.0f, kLumaScale) + 0.5f));
}

uint PackWord(uint b0, uint b1, uint b2, uint b3)
{
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// Packed word w (pixels 4w..4w+3) of candidate row y, clamped to the level.
uint LoadPackedWord(int w, int y, int2 packedMax)
{
    return CandPacked.Load(int3(clamp(int2(w, y), int2(0, 0), packedMax), 0));
}

// Four consecutive candidate pixels starting at p (any alignment).
uint LoadCandWord(int2 p, int2 packedMax)
{
    const int w = p.x >> 2;              // floor for negatives too
    const uint shift = (uint)(p.x & 3) * 8u;
    const uint lo = LoadPackedWord(w, p.y, packedMax);
    if (shift == 0u) return lo;
    const uint hi = LoadPackedWord(w + 1, p.y, packedMax);
    return (lo >> shift) | (hi << (32u - shift));
}

float2 SampleParentFlow(Texture2D<float2> parent, float2 childBlock)
{
    uint parentWidth, parentHeight;
    parent.GetDimensions(parentWidth, parentHeight);
    // Parent pixels are 2x larger; block sizes may differ per level.
    float ratio = (float)BlockSize / max(1.0f, (float)ParentBlockSize * 2.0f);
    int2 maxCoord = int2(parentWidth, parentHeight) - 1;
    float2 p = clamp(childBlock * ratio, 0.0f, float2(maxCoord));
    int2 p0 = int2(floor(p));
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

// Group-wide sum of a per-thread float (one barrier; none on a single
// wave: 64 threads at wave64 — the common AMD configuration — where the
// wave reduction alone is already group-wide).
float GroupSum(float v, uint wave, uint waveCount)
{
    if (waveCount == 1u) return WaveActiveSum(v);
    const float partial = WaveActiveSum(v);
    if (WaveIsFirstLane()) WaveSum[wave] = partial;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    for (uint i = 0; i < waveCount; ++i) total += WaveSum[i];
    return total;
}

uint GroupSumU(uint v, uint wave, uint waveCount)
{
    if (waveCount == 1u) return WaveActiveSum(v);
    const uint partial = WaveActiveSum(v);
    if (WaveIsFirstLane()) WaveSumU[wave] = partial;
    GroupMemoryBarrierWithGroupSync();
    uint total = 0u;
    for (uint i = 0; i < waveCount; ++i) total += WaveSumU[i];
    return total;
}

// Group-wide best (score, vector); ties go to the lowest lane (one barrier,
// skipped on a single wave).
void GroupBest(float score, float2 vector, uint wave, uint waveCount,
               out float bestScore, out float2 bestVector)
{
    const float waveMax = WaveActiveMax(score);
    const uint lane = WaveGetLaneIndex();
    const uint winner = WaveActiveMin(score == waveMax ? lane : 0xFFFFFFFFu);
    if (waveCount == 1u)
    {
        bestScore = waveMax;
        bestVector = WaveReadLaneAt(vector, winner);
        return;
    }
    const float2 waveVector = WaveReadLaneAt(vector, winner);
    if (WaveIsFirstLane())
    {
        WaveBest[wave] = waveMax;
        WaveBestVector[wave] = waveVector;
    }
    GroupMemoryBarrierWithGroupSync();
    bestScore = WaveBest[0];
    bestVector = WaveBestVector[0];
    for (uint i = 1; i < waveCount; ++i)
    {
        if (WaveBest[i] > bestScore)
        {
            bestScore = WaveBest[i];
            bestVector = WaveBestVector[i];
        }
    }
}

// Scores the +-radius window around `base`: stages the candidate region as
// packed words, then each task (candidate row, group of 4 horizontal
// offsets) accumulates msad4 over the 8 reference rows. Returns the
// group-wide best through the out parameters (uniform).
void SearchWindow(float2 base, int radius, int2 blockOrigin, int2 packedMax, uint li,
                  uint wave, uint waveCount, float keptScale,
                  out float bestScore, out float2 bestVector)
{
    const int2 ib = (int2)round(base);
    const int span = 2 * radius + 1;
    const int rows = span - 1 + kBlock;
    // Word-aligned strip: the first candidate sits `shift` bytes into it.
    const int2 windowOrigin = blockOrigin + ib - int2(radius, radius);
    const int firstWord = windowOrigin.x >> 2;
    const int shift = windowOrigin.x & 3;
    const int quadGroups = (span + shift + 3) / 4;
    const int words = quadGroups + kQuads;

    GroupMemoryBarrierWithGroupSync();   // previous readers of Strip are done
    [loop]
    for (int t = (int)li; t < rows * words; t += kThreads)
    {
        const int row = t / words;
        const int col = t - row * words;
        Strip[row][col] = LoadPackedWord(firstWord + col, windowOrigin.y + row, packedMax);
    }
    GroupMemoryBarrierWithGroupSync();

    float myBest = -1e30f;
    float2 myVector = base;
    const int tasks = span * quadGroups;
    [loop]
    for (int task = (int)li; task < tasks; task += kThreads)
    {
        const int oy = task / quadGroups;
        const int qx = task - oy * quadGroups;      // strip byte columns 4qx .. 4qx+3
        uint4 sadEdge = uint4(0u, 0u, 0u, 0u);
        uint4 sadSmooth = uint4(0u, 0u, 0u, 0u);
        [loop]
        for (int r = 0; r < kBlock; ++r)
        {
            [unroll]
            for (int q = 0; q < kQuads; ++q)
            {
                const uint2 src = uint2(Strip[oy + r][q + qx], Strip[oy + r][q + qx + 1]);
                sadEdge = msad4(RefWords[r][q], src, sadEdge);
                sadSmooth = msad4(SmoothWords[r][q], src, sadSmooth);
            }
        }
        [unroll]
        for (int m = 0; m < 4; ++m)
        {
            const int ox = qx * 4 + m - shift;      // candidate x offset within the window
            if (ox < 0 || ox >= span) continue;
            const float2 v = float2(ib) + float2((float)(ox - radius), (float)(oy - radius));
            const float2 dv = v - Predictor;
            const float weighted = (float)sadEdge[m] + kSmoothWeight * (float)sadSmooth[m];
            const float score = 1.0f - weighted * keptScale
                              - max(SmoothnessWeight, 0.0f) * dot(dv, dv) * kSmoothnessScale;
            // Deterministic tie-break toward the earlier (smaller) offset.
            if (score > myBest)
            {
                myBest = score;
                myVector = v;
            }
        }
    }
    GroupBest(myBest, myVector, wave, waveCount, bestScore, bestVector);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    const int2 packedMax = int2((int)((LevelWidth + 3) / 4) - 1, (int)LevelHeight - 1);
    const int radius = clamp(SearchRadius, 0, kMaxRadius);
    const int2 blockOrigin = int2(groupId.xy) * kBlock;
    const bool haveTemporal = FlowScale > 0.0f;
    const uint waveLanes = WaveGetLaneCount();
    const uint wave = li / waveLanes;
    const uint waveCount = (kThreads + waveLanes - 1) / waveLanes;

    // --- Stage 1: predictors, reference bytes, edge mask -------------------
    if (li == 0)
    {
        Predictor = (LevelIndex < TotalLevels - 1)
            ? SampleParentFlow(CoarsePredictor, float2(groupId.xy))
            : float2(0.0f, 0.0f);
        Temporal = haveTemporal ? TemporalFlow.Load(int3(groupId.xy, 0)) : float2(0.0f, 0.0f);
    }
    // Reference block with a 1 px halo, for the gradients.
    [loop]
    for (int t = (int)li; t < kTile * kTile; t += kThreads)
    {
        const int2 tp = int2(t % kTile, t / kTile);
        const int2 p = clamp(blockOrigin + tp - 1, int2(0, 0), levelMax);
        RefTile[tp.y][tp.x] = RefImage.Load(int3(p, 0));
    }
    GroupMemoryBarrierWithGroupSync();
    const int2 tp = int2(tid.xy) + 1;
    const float refValue = RefTile[tp.y][tp.x];
    const float gx = RefTile[tp.y][tp.x + 1] - RefTile[tp.y][tp.x - 1];
    const float gy = RefTile[tp.y + 1][tp.x] - RefTile[tp.y - 1][tp.x];
    const float gradient = abs(gx) + abs(gy);
    const float gradientMean = GroupSum(gradient, wave, waveCount) / (float)kThreads;

    // Adaptive edge threshold: half the block's mean gradient. In a block
    // holding a cel edge over smooth texture the edge stays, the texture
    // drops out; in a fully textured block most samples stay.
    const float threshold = max(kMinGradient, kGradientFraction * gradientMean);
    const bool edge = gradient >= threshold;
    // Bytes are forced nonzero: a zero reference byte is the msad4 mask.
    RefBytes[tid.y][tid.x] = max(1u, QuantizeLuma(refValue));
    EdgeMask[tid.y][tid.x] = edge ? 1u : 0u;
    const uint edgeCount = GroupSumU(edge ? 1u : 0u, wave, waveCount);   // barrier: bytes visible
    if (li < (uint)(kBlock * kQuads))
    {
        const int row = (int)li / kQuads;
        const int quad = (int)li % kQuads;
        uint e[4], s[4];
        [unroll]
        for (int m = 0; m < 4; ++m)
        {
            const uint b = RefBytes[row][quad * 4 + m];
            const bool isEdge = EdgeMask[row][quad * 4 + m] != 0u;
            e[m] = isEdge ? b : 0u;
            s[m] = isEdge ? 0u : b;
        }
        RefWords[row][quad] = PackWord(e[0], e[1], e[2], e[3]);
        SmoothWords[row][quad] = PackWord(s[0], s[1], s[2], s[3]);
    }
    GroupMemoryBarrierWithGroupSync();

    const float2 predictor = Predictor;
    const float2 temporal = Temporal;

    // Diagnostics (MOTION_ENHANCER_MSAD_DEBUG=n sets SmoothnessWeight = -n):
    //   1: report the edge count and threshold instead of a vector
    //   3: level 0 reports the coarser level's result (parent predictor)
    //   5: skip the window search   6: skip the centre scoring   (timing)
    if (SmoothnessWeight < 0.0f && SmoothnessWeight > -1.5f)
    {
        if (li == 0) OutputFlow[groupId.xy] = float2((float)edgeCount, threshold);
        return;
    }
    if (SmoothnessWeight < -2.5f && SmoothnessWeight > -3.5f && LevelIndex == 0)
    {
        if (li == 0) OutputFlow[groupId.xy] = predictor;
        return;
    }
    const bool debugNoWindow = SmoothnessWeight < -4.5f && SmoothnessWeight > -5.5f;
    const bool debugNoCentres = SmoothnessWeight < -5.5f && SmoothnessWeight > -6.5f;

    // Blocks with few edge samples still choose among the predictor
    // candidates below (a sliver of outline is enough to tell the object's
    // motion from the background's), but skip the window search, where
    // flat content only finds chance matches. A fully flat block ties and
    // keeps the parent predictor; the consensus filter reconciles it with
    // its neighbours.
    const bool searchWindow = edgeCount >= (uint)kMinEdgeSamples && !debugNoWindow;
    const float weightSum = (float)edgeCount + kSmoothWeight * (float)(kBlock * kBlock - edgeCount);
    const float keptScale = 1.0f / (weightSum * kLumaScale);

    // --- Stage 2: centre candidates ------------------------------------------
    if (li < (uint)kCentres)
    {
        float2 candidate = predictor;
        bool valid = true;
        if (li == 1)
        {
            candidate = temporal;
            valid = haveTemporal;
        }
        else if (li >= 2)
        {
            // Neighbouring parent blocks (EPZS): an object that straddles
            // blocks at the coarse level is often carried by a neighbour.
            const int n = (int)li - 2;
            const int idx = n < 4 ? n : n + 1;                 // skip the centre
            const int2 offset = int2(idx % 3 - 1, idx / 3 - 1);
            valid = LevelIndex < TotalLevels - 1;
            if (valid)
                candidate = SampleParentFlow(CoarsePredictor, float2(groupId.xy) + float2(offset) * 2.0f);
        }
        CentreVec[li] = (int2)round(candidate);
        CentreValid[li] = valid && !debugNoCentres;
        CentreSad[li] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    // 80 tasks: (candidate, reference row); each packs the candidate's
    // 8 bytes on that row and runs 2 msad4 per class.
    [loop]
    for (int t = (int)li; t < kCentres * kBlock; t += kThreads)
    {
        const int c = t / kBlock;
        const int row = t % kBlock;
        if (!CentreValid[c]) continue;
        const int2 origin = blockOrigin + CentreVec[c] + int2(0, row);
        uint sadEdge = 0u;
        uint sadSmooth = 0u;
        [unroll]
        for (int q = 0; q < kQuads; ++q)
        {
            const uint2 src = uint2(LoadCandWord(origin + int2(q * 4, 0), packedMax), 0u);
            sadEdge += msad4(RefWords[row][q], src, uint4(0u, 0u, 0u, 0u)).x;
            sadSmooth += msad4(SmoothWords[row][q], src, uint4(0u, 0u, 0u, 0u)).x;
        }
        InterlockedAdd(CentreSad[c], sadEdge * 16u + kSmoothWeightFixed * sadSmooth);
    }
    GroupMemoryBarrierWithGroupSync();
    float2 centre = predictor;
    {
        uint best = 0xFFFFFFFFu;
        [unroll]
        for (int i = 0; i < kCentres; ++i)
        {
            if (CentreValid[i] && CentreSad[i] < best)
            {
                best = CentreSad[i];
                centre = float2(CentreVec[i]);
            }
        }
    }
    if (!searchWindow)
    {
        if (li == 0) OutputFlow[groupId.xy] = centre;
        return;
    }

    // --- Stage 3: window around the best centre ---------------------------
    float scoreP;
    float2 vectorP;
    SearchWindow(centre, radius, blockOrigin, packedMax, li, wave, waveCount, keptScale, scoreP, vectorP);

    // --- Stage 4: zero-centred recovery window ---------------------------
    float scoreZ = -1e30f;
    float2 vectorZ = float2(0.0f, 0.0f);
    const bool zeroDistinct = any(abs(round(centre)) > (float)radius);
    if (zeroDistinct)
        SearchWindow(float2(0.0f, 0.0f), radius, blockOrigin, packedMax, li, wave, waveCount, keptScale, scoreZ, vectorZ);

    if (li == 0)
    {
        OutputFlow[groupId.xy] = (zeroDistinct && scoreZ > scoreP + kZeroMargin) ? vectorZ : vectorP;
    }
}
