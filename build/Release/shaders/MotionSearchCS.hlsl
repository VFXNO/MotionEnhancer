#include "Common.hlsli"

// Scalar 8x8 SAD block matcher (SM 6.0: wave reductions).
// One 8x8 thread group per 8x8 flow block; the C++ side dispatches exactly
// (flowWidth, flowHeight, 1) groups, so groupId.xy is the flow block
// coordinate at this level.
//
//   RefImage        frame0 luma at this level (R16_FLOAT, 0..255 scale)
//   CandPacked      frame1 luma at this level, 4 quantized bytes per texel
//                   (PackLumaCS): one load per word
//   CoarsePredictor estimated flow of the coarser level (unbound at the
//                   coarsest level)
//   TemporalFlow    estimated flow of the previous frame pair at THIS level
//                   (same units; bound when FlowScale > 0)
//   OutputFlow      best vector, in this level's pixels
//
//   * Match cost is the ordinary sum of |reference - candidate| over all
//     64 quantized luma pixels, with equal weight for every pixel.
//   * Packed luma is only a storage/read optimization: bytes are unpacked
//     and compared individually, with no msad4 instruction.
//   * Edge-only SAD is an additional neural feature, never a weighting of
//     the actual matching cost.
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
Texture2D<float2> RefFeature : register(t4);
Texture2D<float2> CandFeature : register(t5);
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
// Repeating or dense textures produce several near-equal SAD minima (alias
// peaks); a block on such content measures several "equally right" motions.
// While the temporal chain is bound (FlowScale > 0), candidates within this
// band of the best score count as ambiguous and the one closest to the
// previous pair's vector wins: the decision stays stable across pairs, so
// the warp interpolates smoothly instead of popping between alias peaks.
// Vectors are never suppressed by ambiguity - only disambiguated. Without
// a temporal chain there is nothing to disambiguate with, and the plain
// minimum is kept.
static const float kAmbiguityBand = 1.0f / kLumaScale;
static const int kMinEdgeSamples = 6;
// One luma level of weighted mean absolute difference (score units).
static const float kZeroMargin = 1.0f / kLumaScale;
static const float kMinGradient = 2.0f;
static const float kGradientFraction = 0.5f;
// (1/255) / (0.0005 * 64): the default SmoothnessWeight costs one luma
// level at 8 px: enough to settle ties, never enough to hold motion back.
static const float kSmoothnessScale = 0.1225f;

groupshared float  RefTile[kTile][kTile];                  // luma with halo (gradients)
groupshared uint   RefBytes[kBlock][kBlock];
groupshared uint   EdgeMask[kBlock][kBlock];
groupshared uint   RefWords[kBlock][kQuads];              // all reference luma bytes
groupshared uint   EdgeWords[kBlock][kQuads];             // 0xff at edge pixels, 0 otherwise
groupshared uint   Strip[kStripRows][kStripWords];
groupshared float LearnedLogits[kSpanMax * kSpanMax];
groupshared uint   CentreSad[kCentres];
groupshared uint   CentreEdgeSad[kCentres];
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

// Ordinary scalar SAD of four individual luma pixels. Edge-only SAD is
// computed alongside it for the neural feature; zero luma remains a valid
// pixel rather than the masked-byte sentinel used by msad4.
uint2 SadWord(uint reference, uint edgeMask, uint candidate)
{
    uint2 result = 0u;
    [unroll]
    for (int pixel = 0; pixel < 4; ++pixel)
    {
        const uint shift = (uint)pixel * 8u;
        const int a = (int)((reference >> shift) & 255u);
        const int b = (int)((candidate >> shift) & 255u);
        const uint difference = (uint)abs(a - b);
        result.x += difference;
        result.y += difference * ((edgeMask >> shift) & 1u);
    }
    return result;
}

void SadFour(uint reference, uint edgeMask, uint2 source,
             inout uint4 total, inout uint4 edges)
{
    [unroll]
    for (int offset = 0; offset < 4; ++offset)
    {
        const uint shift = (uint)offset * 8u;
        uint candidate = source.x >> shift;
        if (offset != 0) candidate |= source.y << (32u - shift);
        const uint2 costs = SadWord(reference, edgeMask, candidate);
        total[offset] += costs.x;
        edges[offset] += costs.y;
    }
}

// Packed word w (pixels 4w..4w+3) of candidate row y, clamped to the level.
uint LoadPackedWord(int w, int y, int2 packedMax)
{
    const int row = clamp(y, 0, packedMax.y);
    if (w < 0)
    {
        const uint first = CandPacked.Load(int3(0, row, 0)) & 255u;
        return first * 0x01010101u;
    }
    if (w > packedMax.x)
    {
        const uint word = CandPacked.Load(int3(packedMax.x, row, 0));
        const uint last = (word >> (((LevelWidth - 1u) & 3u) * 8u)) & 255u;
        return last * 0x01010101u;
    }
    return CandPacked.Load(int3(w, row, 0));
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

// Match descriptors produced by the trainable 3x3 image-filter shader.
// No raw luma SAD, hand-weighted edge cost, or msad4 is used here.
// Six descriptions of the same sixteen sampled differences: block mean,
// centre mean, and one mean per quadrant. The quadrants are free - the
// samples are already loaded - and they are what tells the head that a block
// straddles a motion boundary rather than simply matching badly everywhere.
struct LearnedCost
{
    float all;
    float centre;
    float4 quadrant;
};

LearnedCost LearnedCosts(float2 motion, int2 blockOrigin, int2 levelMax)
{
    const int2 shift = (int2)round(motion);
    LearnedCost cost;
    cost.all = 0.0f;
    cost.centre = 0.0f;
    cost.quadrant = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll]
    for (int sy = 0; sy < 4; ++sy)
        [unroll]
        for (int sx = 0; sx < 4; ++sx)
        {
            const int2 p = clamp(blockOrigin + int2(2 * sx + 1, 2 * sy + 1),
                                 int2(0, 0), levelMax);
            const int2 q = clamp(p + shift, int2(0, 0), levelMax);
            const float2 difference = RefFeature.Load(int3(p, 0)) - CandFeature.Load(int3(q, 0));
            const float squared = 0.5f * dot(difference, difference);
            cost.all += squared;
            if ((sx == 1 || sx == 2) && (sy == 1 || sy == 2)) cost.centre += squared;
            cost.quadrant[(sy / 2) * 2 + sx / 2] += squared;
        }
    cost.all *= 1.0f / 16.0f;
    cost.centre *= 0.25f;
    cost.quadrant *= 0.25f;
    return cost;
}

// `centre` is the displacement the search window is built around. In training
// (RealtimeSession, NativeFlowExample) the window was always the +-8 box
// around the predictor and feature 2/3 was the offset *from that box centre*,
// never a distance to the pyramid predictor as such. Passing the predictor
// here while the window sits on a different anchor feeds the head offsets far
// outside [-0.5, 0.5], where its ReLU units extrapolate linearly and the
// learned proximity prior swamps the feature cost - which is exactly the case
// of a large motion, whose anchor is by definition far from the predictor.
float LearnedScore(float2 motion, float2 centre, float2 temporal,
                   int2 blockOrigin, int2 levelMax)
{
    const LearnedCost cost = LearnedCosts(motion, blockOrigin, levelMax);
    const float2 dp = (motion - centre) / 16.0f;
    // The temporal offset is absolute (candidate minus previous pair's
    // vector), so unlike dp it can leave the trained range whenever the anchor
    // is far from the previous vector. Saturate it at the 16 px the training
    // window spanned rather than letting the linear tail dominate the score.
    const float2 dt = clamp((motion - temporal) / 16.0f, -1.0f, 1.0f);
    // Ten inputs per unit, three constant rows each: [all centre q0 q1],
    // [q2 q3 dp.x dp.y], [dt.x dt.y . .].
    const float4 f0 = float4(cost.all, cost.centre, cost.quadrant.x, cost.quadrant.y);
    const float4 f1 = float4(cost.quadrant.z, cost.quadrant.w, dp);
    float total = NetControl.x;
    [unroll]
    for (int j = 0; j < 8; ++j)
    {
        const float pre = NetBias[j / 4][j % 4] + dot(f0, NetWeights[j * 3])
            + dot(f1, NetWeights[j * 3 + 1]) + dot(dt, NetWeights[j * 3 + 2].xy);
        total += max(0.0f, pre) * NetOut[j / 4][j % 4];
    }
    return 4.0f * (total + (FlowScale > 0.0f ? NetControl.z * dot(dt, dt) : 0.0f));
}

void LearnedWindow(float2 base, int radius, float2 temporal,
                   int2 blockOrigin, int2 levelMax, uint li, uint wave, uint waveCount,
                   out float bestScore, out float2 vector)
{
    const int2 center = (int2)round(base);
    const float2 centre = float2(center);
    // Dense, every displacement scored. A strided three-step search over the
    // same reach was 6x cheaper and far less accurate: it works on the smooth
    // luma SAD surface, but the learned cost is an edge correlation with
    // sharp minima, and a stride-4 probe steps straight over them. Cost is
    // controlled by the window's radius instead - see kNeuralRadius.
    const int span = 2 * radius + 1;
    const int candidates = span * span;
    float localMax = -1e30f;
    GroupMemoryBarrierWithGroupSync();   // previous readers of LearnedLogits are done
    [loop]
    for (int task = (int)li; task < candidates; task += kThreads)
    {
        const float2 displacement = float2(center + int2(task % span - radius,
                                                          task / span - radius));
        const float score = LearnedScore(displacement, centre, temporal,
                                         blockOrigin, levelMax);
        LearnedLogits[task] = score;
        localMax = max(localMax, score);
    }
    float2 unused;
    GroupBest(localMax, base, wave, waveCount, bestScore, unused);
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f, x = 0.0f, y = 0.0f;
    [loop]
    for (int task = (int)li; task < candidates; task += kThreads)
    {
        const float weight = exp(clamp(LearnedLogits[task] - bestScore, -80.0f, 0.0f));
        const float2 motion = float2(center + int2(task % span - radius,
                                                   task / span - radius));
        total += weight;
        x += weight * motion.x;
        y += weight * motion.y;
    }
    const float sum = GroupSum(total, wave, waveCount);
    vector = float2(GroupSum(x, wave, waveCount), GroupSum(y, wave, waveCount)) /
             max(sum, 1e-10f);
}

// Scores the +-radius window around `base`: stages the candidate region as
// packed words, then each task (candidate row, group of 4 horizontal
// offsets) accumulates scalar per-pixel SAD over the 8 reference rows. Returns the
// group-wide best through the out parameters (uniform).
void SearchWindow(float2 base, int radius, int2 blockOrigin, int2 packedMax, uint li,
                  uint wave, uint waveCount, float keptScale, uint edgeCount,
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
    // Best candidate within the ambiguity band of the running maximum,
    // measured against the previous pair's vector (temporal chain only):
    // what an alias tie resolves to.
    const bool coherent = FlowScale > 0.0f;
    float myCoherentScore = -1e30f;
    float2 myCoherentVector = base;
    float myCoherentDist2 = 1e30f;
    const int tasks = span * quadGroups;
    [loop]
    for (int task = (int)li; task < tasks; task += kThreads)
    {
        const int oy = task / quadGroups;
        const int qx = task - oy * quadGroups;      // strip byte columns 4qx .. 4qx+3
        uint4 sadTotal = uint4(0u, 0u, 0u, 0u);
        uint4 sadEdge = uint4(0u, 0u, 0u, 0u);
        [loop]
        for (int r = 0; r < kBlock; ++r)
        {
            [unroll]
            for (int q = 0; q < kQuads; ++q)
            {
                const uint2 src = uint2(Strip[oy + r][q + qx], Strip[oy + r][q + qx + 1]);
                SadFour(RefWords[r][q], EdgeWords[r][q], src, sadTotal, sadEdge);
            }
        }
        [unroll]
        for (int m = 0; m < 4; ++m)
        {
            const int ox = qx * 4 + m - shift;      // candidate x offset within the window
            if (ox < 0 || ox >= span) continue;
            const float2 v = float2(ib) + float2((float)(ox - radius), (float)(oy - radius));
            const float2 dv = v - Predictor;
            const float cost = (float)sadTotal[m];
            const float score = 1.0f - cost * keptScale
                              - max(SmoothnessWeight, 0.0f) * dot(dv, dv) * kSmoothnessScale;
            if (!coherent)
            {
                // Deterministic tie-break toward the earlier (smaller) offset.
                if (score > myBest)
                {
                    myBest = score;
                    myVector = v;
                }
                continue;
            }
            const float dist2 = dot(v - Temporal, v - Temporal);
            if (score > myBest + kAmbiguityBand)
            {
                // Clear win: the best moves and the ambiguity band moves
                // with it; the old coherent choice may fall out of the band.
                myBest = score;
                myVector = v;
                if (myBest - myCoherentScore > kAmbiguityBand || dist2 < myCoherentDist2)
                {
                    myCoherentScore = score;
                    myCoherentVector = v;
                    myCoherentDist2 = dist2;
                }
            }
            else if (score >= myBest - kAmbiguityBand)
            {
                // Ambiguous with the current best.
                if (score > myBest)
                {
                    myBest = score;
                    myVector = v;
                }
                if (dist2 < myCoherentDist2 || myBest - myCoherentScore > kAmbiguityBand)
                {
                    myCoherentScore = score;
                    myCoherentVector = v;
                    myCoherentDist2 = dist2;
                }
            }
        }
    }
    // A near-tie goes to the temporally stable candidate; an unambiguous
    // best keeps the raw maximum. Either way the score reported to
    // GroupBest is the true maximum, so waves compare fairly.
    if (coherent && myBest - myCoherentScore <= kAmbiguityBand) myVector = myCoherentVector;
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
    RefBytes[tid.y][tid.x] = QuantizeLuma(refValue);
    EdgeMask[tid.y][tid.x] = edge ? 1u : 0u;
    const uint edgeCount = GroupSumU(edge ? 1u : 0u, wave, waveCount);   // barrier: bytes visible
    if (li < (uint)(kBlock * kQuads))
    {
        const int row = (int)li / kQuads;
        const int quad = (int)li % kQuads;
        uint values[4], masks[4];
        [unroll]
        for (int m = 0; m < 4; ++m)
        {
            const uint b = RefBytes[row][quad * 4 + m];
            const bool isEdge = EdgeMask[row][quad * 4 + m] != 0u;
            values[m] = b;
            masks[m] = isEdge ? 255u : 0u;
        }
        RefWords[row][quad] = PackWord(values[0], values[1], values[2], values[3]);
        EdgeWords[row][quad] = PackWord(masks[0], masks[1], masks[2], masks[3]);
    }
    GroupMemoryBarrierWithGroupSync();

    const float2 predictor = Predictor;
    const float2 temporal = Temporal;

    // Diagnostics (MOTION_ENHANCER_SAD_DEBUG=n sets SmoothnessWeight = -n):
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
    // keeps the parent predictor instead of inventing a zero vector.
    const bool searchWindow = edgeCount >= (uint)kMinEdgeSamples && !debugNoWindow;
    const float weightSum = (float)(kBlock * kBlock);
    const float keptScale = 1.0f / (weightSum * kLumaScale);

    if (NetControl.y > 0.0f)
    {
        // The learned feature network chooses a search anchor and directly
        // estimates the vector from the learned cost volume at every scale.
        // Every anchor is scored as the centre of its own window (offset
        // zero), so the anchors are ranked on learned feature cost alone, the
        // way the classical centre stage ranks them on SAD alone. Scoring
        // them as offsets from the predictor instead made the head's
        // proximity prior a distance penalty, and no anchor eight or more
        // pixels out could win - the estimator could never leave the
        // predictor's neighbourhood, which is what a large motion requires.
        // One candidate per thread: 35 serialized scores on lane 0 was also
        // 1120 feature loads with 63 lanes idle.
        float anchorScore = -1e30f;
        float2 anchorVector = predictor;
        if (li < 35u)
        {
            float2 candidate = predictor;
            bool valid = true;
            if (li == 1u)
            {
                candidate = temporal;
                valid = haveTemporal;
            }
            else if (li < 10u)
            {
                // Neighbouring parent vectors (EPZS).
                const int n = (int)li - 2;
                const int idx = n < 4 ? n : n + 1;            // skip the centre
                valid = LevelIndex < TotalLevels - 1;
                if (valid)
                    candidate = SampleParentFlow(CoarsePredictor,
                        float2(groupId.xy) + float2(idx % 3 - 1, idx / 3 - 1) * 2.0f);
            }
            else
            {
                // A foreground that disappears in the coarse pyramid may have
                // no moving parent vector. Offer the learned matcher a sparse
                // wider set of anchors at fine levels, rather than forcing it
                // to refine a zero predictor that is more than eight pixels
                // wrong.
                const int g = (int)li - 10;
                valid = LevelIndex <= 2;
                candidate = predictor + float2(g % 5 - 2, g / 5 - 2) * 8.0f;
            }
            if (valid)
            {
                anchorVector = candidate;
                anchorScore = LearnedScore(candidate, predictor, temporal,
                                           blockOrigin, levelMax);
            }
        }
        // Gating a cost-only anchor jump behind a single learned margin was
        // tried and reverted: one scalar threshold cannot separate a
        // background block that should stay put from a foreground block that
        // should jump, and training traded the many for the few (foreground
        // 11.95 -> 10.79 px, but global motion 2.35 -> 4.13 px).
        float anchorBest;
        float2 anchor;
        GroupBest(anchorScore, anchorVector, wave, waveCount, anchorBest, anchor);

        float score, zeroScore;
        float2 vector, zeroVector;
        LearnedWindow(anchor, radius, temporal, blockOrigin,
                      levelMax, li, wave, waveCount, score, vector);
        const bool zeroDistinct = any(abs(round(anchor)) > (float)radius);
        if (zeroDistinct)
        {
            LearnedWindow(float2(0.0f, 0.0f), radius, temporal,
                          blockOrigin, levelMax, li, wave, waveCount, zeroScore, zeroVector);
            // Two windows are two hypotheses, not two samples of one: a
            // sigmoid blend of "moved by twelve" and "did not move" returns
            // six, a vector neither window supports. Pick, as the classical
            // path does, and let ties go to motion rather than to a stall.
            if (zeroScore > score) vector = zeroVector;
        }
        if (li == 0) OutputFlow[groupId.xy] = vector;
        return;
    }

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
        CentreEdgeSad[li] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    // 80 tasks: (candidate, reference row); each compares eight luma
    // pixels with ordinary scalar SAD, plus an edge-only neural feature.
    [loop]
    for (int t = (int)li; t < kCentres * kBlock; t += kThreads)
    {
        const int c = t / kBlock;
        const int row = t % kBlock;
        if (!CentreValid[c]) continue;
        const int2 origin = blockOrigin + CentreVec[c] + int2(0, row);
        uint sadTotal = 0u;
        uint sadEdge = 0u;
        [unroll]
        for (int q = 0; q < kQuads; ++q)
        {
            const uint src = LoadCandWord(origin + int2(q * 4, 0), packedMax);
            const uint2 cost = SadWord(RefWords[row][q], EdgeWords[row][q], src);
            sadTotal += cost.x;
            sadEdge += cost.y;
        }
        InterlockedAdd(CentreSad[c], sadTotal * 16u);
        InterlockedAdd(CentreEdgeSad[c], sadEdge);
    }
    GroupMemoryBarrierWithGroupSync();
    float2 centre = predictor;
    if (FlowScale > 0.0f)
    {
        // Temporal chain bound: one luma level of weighted difference in
        // CentreSad fixed point (x16) counts as ambiguous, and the tie goes
        // to the candidate closest to the previous pair's vector, so
        // repeating/dense textures keep their decision across pairs and the
        // warp does not pop between alias peaks.
        uint best = 0xFFFFFFFFu;
        [unroll]
        for (int i = 0; i < kCentres; ++i)
        {
            if (CentreValid[i] && CentreSad[i] < best) best = CentreSad[i];
        }
        const uint band = (uint)max(1.0f, weightSum * 16.0f);
        float bestDist2 = 1e30f;
        [unroll]
        for (int i = 0; i < kCentres; ++i)
        {
            if (!CentreValid[i] || CentreSad[i] > best + band) continue;
            const float2 dv = float2(CentreVec[i]) - Temporal;
            const float dist2 = dot(dv, dv);
            if (dist2 < bestDist2)
            {
                bestDist2 = dist2;
                centre = float2(CentreVec[i]);
            }
        }
    }
    else
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
    SearchWindow(centre, radius, blockOrigin, packedMax, li, wave, waveCount, keptScale, edgeCount, scoreP, vectorP);

    // --- Stage 4: zero-centred recovery window ---------------------------
    float scoreZ = -1e30f;
    float2 vectorZ = float2(0.0f, 0.0f);
    const bool zeroDistinct = any(abs(round(centre)) > (float)radius);
    if (zeroDistinct)
        SearchWindow(float2(0.0f, 0.0f), radius, blockOrigin, packedMax, li, wave, waveCount, keptScale, edgeCount, scoreZ, vectorZ);

    float2 chosen = (zeroDistinct && scoreZ > scoreP + kZeroMargin) ? vectorZ : vectorP;
    if (li == 0) OutputFlow[groupId.xy] = chosen;
}
