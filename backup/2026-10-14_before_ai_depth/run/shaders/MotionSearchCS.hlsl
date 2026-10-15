#include "Common.hlsli"

// Scalar 8x8 SAD block matcher (SM 6.0: wave reductions).
// One 8x8 thread group per 8x8 flow block; the C++ side dispatches exactly
// (flowWidth, flowHeight, 1) groups, so groupId.xy is the flow block
// coordinate at this level.
//
//   RefImage        frame0 luma at this level (R16_FLOAT, 0..255 scale):
//                   gradients, edge mask and confidence only
//   CandPacked      frame1 at this level as PackLumaCS bytes - luma
//                   normalised to its local mean and contrast, 4 per texel:
//                   one load per word
//   RefPacked       frame0 the same way: the SAD compares these bytes
//   CoarsePredictor estimated flow of the coarser level (unbound at the
//                   coarsest level)
//   OutputFlow      best vector, in this level's pixels
//
//   * Match cost is the ordinary sum of |reference - candidate| over all
//     64 bytes of the block, with equal weight for every pixel.
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
//   1. Centre selection: the parent predictor, the eight neighbouring
//      parent vectors (EPZS) and zero are each scored once, on plain SAD. Zero is in that list so a bad coarse inheritance
//      can always return to "did not move", and it wins only when it is
//      *strictly* better by kZeroMargin: on ambiguous content ties go to
//      motion, never to a stall.
//   2. A +-SearchRadius window around the winning centre. One window, not
//      two: zero used to get a whole second +-8 window of its own whenever
//      it fell outside the first one, which on a camera pan is *every*
//      block, and cost 2.1 ms of 6.6 at 1080p for a recovery that ten
//      scored centres already provide. Whichever centre wins, the window
//      around it still explores the full radius.
//   Blocks with few edge samples pick among the centres only.
//
// Each pair is estimated from its own two frames only. There used to be a
// temporal predictor - the previous pair's vector at this block, offered as
// a centre and used to break ties - and it was removed: a vector about a
// frame that is gone kept steering blocks after the motion under them had
// changed, which showed as ghosting, and on the synthetic sequences every
// case measured equal or better without it (a fast object over a static
// texture +1.2 dB).
Texture2D<float> RefImage : register(t0);
Texture2D<uint> CandPacked : register(t1);
Texture2D<float4> CoarsePredictor : register(t2);
Texture2D<uint> RefPacked : register(t3);
Texture2D<float2> RefFeature : register(t4);
Texture2D<float2> CandFeature : register(t5);
RWTexture2D<float4> OutputFlow : register(u0);

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
static const int kCentres = 10;              // parent, 8 neighbours, zero
static const int kZeroCentre = kCentres - 1;
// Repeating or dense textures produce several near-equal SAD minima (alias
// peaks); a block on such content measures several "equally right" motions.
// Candidates within this band of the best score count as ambiguous and the
// one closest to the coarse predictor wins, so the choice follows this
// pair's own evidence one level up rather than whichever lane happened to
// hold the maximum. Vectors are never suppressed by ambiguity - only
// disambiguated.
static const float kAmbiguityBand = 1.0f / kLumaScale;
static const int kMinEdgeSamples = 6;
// The band inside which two centre candidates are "too close to call", in
// the x16 fixed point CentreSad accumulates: one luma level of mean absolute
// difference over a 64-pixel block, 64 * 16. Same quantity as
// kAmbiguityBand, in the units the centre stage works in.
static const uint kCentreBand = (uint)(kBlock * kBlock) * 16u;

// Which way an unresolvable tie falls, applied to "did not move".
//
// This used to be +kCentreBand: zero had to beat every other candidate by a
// full luma level, so a tie went to motion rather than to a stall. That was
// the only thing stopping a flat *moving* region from freezing, and it is
// no longer needed for that - FillFlowCS now gives a flat region the vector
// of the textured neighbours that actually measured one, which is a direct
// answer to the same problem rather than a thumb on the scale.
//
// Left positive it was actively harmful while the tie-break reference was
// the previous pair's vector (the temporal predictor, since removed): a
// block with no texture has no evidence either way, the bias pushed it off
// zero, and it then kept whatever it last had forever. Measured on two **identical** frames with the chain
// primed by a moving pair - where the true flow is exactly zero and copying
// either frame scores a perfect 99 dB - the engine reported motion on 61.5%
// of blocks (23.9% still at the stale vector, 22.1% at twice it) and the
// interpolated frame scored 37.5 dB. On a mostly static page that is
// continuous invented motion, which is what "random artifacts" looks like.
//
// Zero, so "did not move" competes on equal terms and an unresolvable tie
// is settled by the reference like any other. Making zero actively *win*
// ties was tried too and measured no better anywhere while costing the
// moving-object cases.
static const int kZeroBiasFixed = 0;
static const float kMinGradient = 2.0f;
static const float kGradientFraction = 0.5f;
// Half-confidence point of the per-axis structure measure: a mean squared
// central difference of 25 over the block, an rms gradient of five luma
// levels. See the confidence computation in CSMain.
static const float kStructureHalf = 25.0f;

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
groupshared float3 WaveSum3[kMaxWaves];
groupshared float  WaveSum[kMaxWaves];
groupshared uint   WaveSumU[kMaxWaves];
groupshared float  WaveBest[kMaxWaves];
groupshared float2 WaveBestVector[kMaxWaves];
groupshared float2 Predictor;

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

float2 SampleParentFlow(Texture2D<float4> parent, float2 childBlock)
{
    uint parentWidth, parentHeight;
    parent.GetDimensions(parentWidth, parentHeight);
    // Parent pixels are 2x larger; block sizes may differ per level.
    float ratio = (float)BlockSize / max(1.0f, (float)ParentBlockSize * 2.0f);
    int2 maxCoord = int2(parentWidth, parentHeight) - 1;
    // Block *centres*, not block corners. Child block b covers level pixels
    // [b*B, b*B+B-1], centred at b*B + (B-1)/2; halving that gives the parent
    // level coordinate, and the parent block whose centre sits there is
    // ((b*B + (B-1)/2)/2 - (PB-1)/2) / PB = b*ratio + centreShift. Sampling
    // at b*ratio alone pulled every predictor 0.21875 parent blocks (~3.5
    // child pixels) diagonally up-left, and every level re-applied the
    // shift: a block on one side of a motion boundary inherited the other
    // side's vector.
    const float centreShift = (0.25f * ((float)BlockSize - 1.0f) -
                               0.5f * ((float)ParentBlockSize - 1.0f)) /
                              max(1.0f, (float)ParentBlockSize);
    float2 p = clamp(childBlock * ratio + centreShift, 0.0f, float2(maxCoord));
    int2 p0 = int2(floor(p));
    float2 f = p - float2(p0);
    int2 c00 = clamp(p0, int2(0, 0), maxCoord);
    int2 c10 = clamp(p0 + int2(1, 0), int2(0, 0), maxCoord);
    int2 c01 = clamp(p0 + int2(0, 1), int2(0, 0), maxCoord);
    int2 c11 = clamp(p0 + int2(1, 1), int2(0, 0), maxCoord);
    float2 v00 = parent.Load(int3(c00, 0)).xy;
    float2 v10 = parent.Load(int3(c10, 0)).xy;
    float2 v01 = parent.Load(int3(c01, 0)).xy;
    float2 v11 = parent.Load(int3(c11, 0)).xy;
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y) * 2.0f;
}

// Group-wide sum of up to three per-thread floats (one barrier; none on a
// single wave: 64 threads at wave64 — the common AMD configuration — where
// the wave reduction alone is already group-wide).
//
// Three components in one reduction, not three reductions. Each of these
// helpers writes its wave slot, syncs once, and then every thread reads
// every slot; a *second* reduction issued before all waves have finished
// reading overwrites a slot that is still being read. Three back-to-back
// single-float sums are exactly that race, and the softmax below needs
// three sums (weight, x, y) that must agree with each other — a contaminated
// denominator is a vector no candidate in the window supports. Reducing them
// together is both correct and two barriers cheaper.
//
// Scratch reuse rule for every helper here: a group-wide sync must separate
// one call's read phase from the next call's write phase. See the call sites.
float3 GroupSum3(float3 v, uint wave, uint waveCount)
{
    if (waveCount == 1u)
        return float3(WaveActiveSum(v.x), WaveActiveSum(v.y), WaveActiveSum(v.z));
    const float3 partial = float3(WaveActiveSum(v.x), WaveActiveSum(v.y), WaveActiveSum(v.z));
    if (WaveIsFirstLane()) WaveSum3[wave] = partial;
    GroupMemoryBarrierWithGroupSync();
    float3 total = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0; i < waveCount; ++i) total += WaveSum3[i];
    return total;
}

// Separate scratch, so the classical path's single sum does not pay for the
// neural path's three (it is one reduction per block per level per
// direction; routing it through GroupSum3 cost two extra wave reductions
// and a wider groupshared write for nothing).
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

// Cost volume of the convolutional estimator. Six descriptions of the same
// sixteen sampled feature differences: block mean, centre mean, and one mean
// per quadrant. The quadrants are free - the samples are already loaded -
// and they are what tells the head that a block straddles a motion boundary
// rather than simply matching badly everywhere.
//
// Must stay identical to costDescriptors() in NeuralFlow.cpp and
// NeuralFlowTrain.cpp, including the clamping: the weights were fitted
// against that arithmetic. --neural-parity-check is the test.
// HLSL `round()` is round-half-to-even; the CPU's std::lround is
// half-away-from-zero. They disagree at exactly x.5, which a bilinear
// predictor doubled into this level's pixels hits often enough to tip the
// anchor argmax onto a different candidate - a whole window away. Both
// sides use this convention instead. Scoped to the neural path so the
// classical engine's measured behaviour is untouched.
int2 NeuralRound(float2 v)
{
    return (int2)floor(v + 0.5f);
}

// Anchor scores within this many logits of the best are treated as not
// separable. The score feeds a softmax, so a logit gap of 0.25 means the
// two candidates differ in probability by a factor of 1.3 - the window
// stage would blend displacements that close, and a whole-window anchor
// decision should not be made on a difference that small either.
//
// Without it the anchor stage is a bare argmax over ten often near-tied
// candidates, so the last bit of a float decides which window is searched.
// Measured against the CPU reference on identical inputs, 2-5% of blocks
// picked a different anchor purely from reassociation - and on video that
// same sensitivity is a block flipping between two anchors from frame to
// frame, which is visible as a popping artifact.
static const float kNeuralAnchorBand = 0.25f;

// Bucketed anchor key, tie-broken toward `reference` - the same device the
// classical path uses for near-equal SAD minima (see RankKey). An anchor a
// full band better always wins; inside a band the stable choice wins.
float NeuralAnchorKey(float score, float2 vector, float2 reference)
{
    const float2 dv = vector - reference;
    return floor(score / kNeuralAnchorBand) + rcp(1.0f + dot(dv, dv));
}

void LearnedCosts(float2 motion, int2 blockOrigin, int2 levelMax, out float cost[kNeuralCosts])
{
    const int2 shift = NeuralRound(motion);
    [unroll]
    for (int i = 0; i < kNeuralCosts; ++i) cost[i] = 0.0f;
    [unroll]
    for (int sy = 0; sy < 4; ++sy)
        [unroll]
        for (int sx = 0; sx < 4; ++sx)
        {
            const int2 p = clamp(blockOrigin + int2(2 * sx + 1, 2 * sy + 1),
                                 int2(0, 0), levelMax);
            const int2 q = clamp(p + shift, int2(0, 0), levelMax);
            const float2 difference = RefFeature.Load(int3(p, 0)) - CandFeature.Load(int3(q, 0));
            const float energy = 0.5f * dot(difference, difference);
            cost[0] += energy;
            if ((sx == 1 || sx == 2) && (sy == 1 || sy == 2)) cost[1] += energy;
            cost[2 + (sy / 2) * 2 + sx / 2] += energy;
        }
    cost[0] *= 1.0f / 16.0f;
    [unroll]
    for (int j = 1; j < kNeuralCosts; ++j) cost[j] *= 0.25f;
}

// Appearance score of a candidate displacement: six costs -> eight ReLU
// units -> one scalar, times a learned temperature.
//
// Appearance and nothing else. The previous head also took the candidate's
// offset from the predictor and from the previous frame's vector as inputs,
// which turned its learned proximity prior into a distance penalty: no
// candidate far from the predictor could win however well it matched, so a
// large motion was unreachable by construction. Geometry does not enter a
// match cost here. The temperature is learned because a fixed one left the
// scores so close together that the softmax came out uniform and the
// estimator returned its own anchor.
float LearnedScore(float2 motion, int2 blockOrigin, int2 levelMax)
{
    float cost[kNeuralCosts];
    LearnedCosts(motion, blockOrigin, levelMax, cost);
    float total = NeuralWeight(kwScoreBias);
    [unroll]
    for (int j = 0; j < kNeuralUnits; ++j)
    {
        float pre = NeuralWeight(kwHeadBias + j);
        [unroll]
        for (int i = 0; i < kNeuralCosts; ++i)
            pre += NeuralWeight(kwHeadWeight + j * kNeuralCosts + i) * cost[i];
        total += max(0.0f, pre) * NeuralWeight(kwOut + j);
    }
    return exp(NeuralWeight(kwLogTemperature)) * total;
}

// Soft-argmax of the learned cost over the +-radius window around `base`:
// the vector is the softmax-weighted mean of every displacement in it, which
// is what makes the estimate differentiable and so trainable end to end.
void LearnedWindow(float2 base, int radius, int2 blockOrigin, int2 levelMax,
                   uint li, uint wave, uint waveCount, out float2 vector)
{
    const int2 center = NeuralRound(base);
    // Dense, every displacement scored. A strided three-step search over the
    // same reach was 6x cheaper and far less accurate: it works on the smooth
    // luma SAD surface, but the learned cost is a feature correlation with
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
        const float score = LearnedScore(displacement, blockOrigin, levelMax);
        LearnedLogits[task] = score;
        localMax = max(localMax, score);
    }
    // Subtracting the group-wide maximum before exp() is what keeps the
    // softmax finite; it cancels out of the normalized weights.
    float bestScore;
    float2 unused;
    GroupBest(localMax, base, wave, waveCount, bestScore, unused);
    GroupMemoryBarrierWithGroupSync();   // WaveBest read, LearnedLogits complete
    float3 moments = float3(0.0f, 0.0f, 0.0f);   // weight, weight*x, weight*y
    [loop]
    for (int task = (int)li; task < candidates; task += kThreads)
    {
        const float weight = exp(clamp(LearnedLogits[task] - bestScore, -80.0f, 0.0f));
        const float2 motion = float2(center + int2(task % span - radius,
                                                   task / span - radius));
        moments += float3(weight, weight * motion.x, weight * motion.y);
    }
    const float3 total = GroupSum3(moments, wave, waveCount);
    vector = total.yz / max(total.x, 1e-10f);
}

// Offset of the minimum of the parabola through three equally spaced cost
// samples, in units of their spacing. `centre` is the lowest of the three.
//
// Returns zero unless the three are convex: a flat or monotone triple has
// no interior minimum, and fitting one anyway invents a displacement out of
// the shape of noise. Clamped to half a sample because a parabola fitted to
// three points cannot locate a minimum outside them.
float ParabolicMinimum(float minus, float centre, float plus)
{
    const float curvature = minus - 2.0f * centre + plus;
    if (curvature <= 0.0f) return 0.0f;
    return clamp(0.5f * (minus - plus) / curvature, -0.5f, 0.5f);
}

// One luma level of mean absolute difference per bucket: score * kLumaScale
// is 255 - sad/64, so consecutive integers are exactly kAmbiguityBand apart.
uint ScoreBucket(float score)
{
    return (uint)max(0.0f, floor(score * kLumaScale));
}

// Ranking key: the SAD bucket, tie-broken toward `reference`. A candidate a
// full luma level better always wins, whatever its distance - the tie-break
// cannot hold a vector back, which a continuous distance cost can and did
// (it is what made a 16 px object read 13 px at every level). Within one
// bucket the measurement genuinely cannot separate the candidates, and the
// one nearest the reference is chosen so the field stays coherent instead
// of each block independently picking its own noise minimum.
//
// Packed into one float so the group-wide reduction resolves ties across
// threads too. Scoring the raw maximum and tie-breaking per thread left the
// final choice to whichever lane happened to hold it.
float RankKey(float score, float2 vector, float2 reference)
{
    const float2 dv = vector - reference;
    return (float)ScoreBucket(score) + rcp(1.0f + dot(dv, dv));
}

// Scores the +-radius window around `base`: stages the candidate region as
// packed words, then each task (candidate row, group of 4 horizontal
// offsets) accumulates scalar per-pixel SAD over the 8 reference rows.
// Returns the group-wide best vector (uniform).
void SearchWindow(float2 base, int radius, float2 reference, int2 blockOrigin,
                  int2 packedMax, uint li, uint wave, uint waveCount, float keptScale,
                  out float2 bestVector)
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

    float myKey = -1e30f;
    float2 myVector = base;
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
            // Match cost alone, bucketed; see RankKey.
            const float score = 1.0f - (float)sadTotal[m] * keptScale;
            const float key = RankKey(score, v, reference);
            if (key > myKey)
            {
                myKey = key;
                myVector = v;
            }
        }
    }
    float bestKey;
    GroupBest(myKey, myVector, wave, waveCount, bestKey, bestVector);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    const int2 packedMax = int2((int)((LevelWidth + 3) / 4) - 1, (int)LevelHeight - 1);
    const int radius = clamp(SearchRadius, 0, kMaxRadius);
    const int2 blockOrigin = int2(groupId.xy) * kBlock;
    const uint waveLanes = WaveGetLaneCount();
    const uint wave = li / waveLanes;
    const uint waveCount = (kThreads + waveLanes - 1) / waveLanes;

    // --- Stage 1: predictors, reference bytes, edge mask -------------------
    if (li == 0)
    {
        Predictor = (LevelIndex < TotalLevels - 1)
            ? SampleParentFlow(CoarsePredictor, float2(groupId.xy))
            : float2(0.0f, 0.0f);
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
    // The mean gradient for the edge threshold, and the diagonal of the
    // block's structure tensor in the same reduction - the second and third
    // components cost nothing beyond two more wave adds.
    //
    // Sum(gx^2) is how much this block's content actually changes when it is
    // displaced along x, which is exactly how much the match cost can say
    // about the x component of its vector. A vertical line has a large
    // Sum(gx^2) and a near-zero Sum(gy^2): its x is measured and its y is
    // not determined by anything in the block. That is the aperture problem,
    // and without a per-axis confidence the undetermined component is
    // indistinguishable from a measured one downstream.
    const float3 blockMoments =
        GroupSum3(float3(gradient, gx * gx, gy * gy), wave, waveCount) / (float)kThreads;
    const float gradientMean = blockMoments.x;
    // Soft, bounded, and free of a hard classification boundary: a block at
    // kStructureHalf scores 0.5 and nothing is ever placed on one side of a
    // cliff. kStructureHalf is a mean squared central difference of 25, i.e.
    // an rms gradient of five luma levels across the block.
    const float2 confidence = blockMoments.yz / (blockMoments.yz + kStructureHalf);

    // Adaptive edge threshold: half the block's mean gradient. In a block
    // holding a cel edge over smooth texture the edge stays, the texture
    // drops out; in a fully textured block most samples stay.
    const float threshold = max(kMinGradient, kGradientFraction * gradientMean);
    const bool edge = gradient >= threshold;
    {
        // The SAD bytes come from the packed pyramid, the same locally
        // normalised luma the candidates are read from (see PackLumaCS) -
        // the raw luma above is only for the gradients and the confidence.
        const int2 p = clamp(blockOrigin + int2(tid.xy), int2(0, 0), levelMax);
        RefBytes[tid.y][tid.x] = (RefPacked.Load(int3(p.x >> 2, p.y, 0)) >> ((uint)(p.x & 3) * 8u)) & 255u;
    }
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
    // What an unresolvable tie falls back to: the coarse predictor, this
    // pair's own evidence measured from these two frames one level up. Only
    // ties consult it; it never competes with a measurably better match,
    // and no candidate is ever penalised for being large.
    const float2 reference = predictor;

    // Diagnostics (MOTION_ENHANCER_SAD_DEBUG=n sets SmoothnessWeight = -n):
    //   1: report the edge count and threshold instead of a vector
    //   3: level 0 reports the coarser level's result (parent predictor)
    //   5: skip the window search   6: skip the centre scoring   (timing)
    if (SmoothnessWeight < 0.0f && SmoothnessWeight > -1.5f)
    {
        if (li == 0) OutputFlow[groupId.xy] = float4((float)edgeCount, threshold, confidence);
        return;
    }
    if (SmoothnessWeight < -2.5f && SmoothnessWeight > -3.5f && LevelIndex == 0)
    {
        if (li == 0) OutputFlow[groupId.xy] = float4(predictor, confidence);
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

    if (NeuralControl.x > 0.0f)
    {
        // The convolutional estimator: rank the anchors on learned appearance
        // cost, then take the soft-argmax of the cost volume around the
        // winner. Structurally the same two stages as the classical path
        // below, with the learned cost in place of SAD, which is what makes
        // the two directly comparable.
        //
        // Ten anchors - predictor, eight parent neighbours, zero
        // - and no more. The previous version also offered a 5x5 grid of
        // +-16 px jumps around the predictor; with a nearly flat cost those
        // were selected by noise and produced -35 px vectors on a 10 px
        // motion. Reach comes from the pyramid, the same place the classical
        // engine gets it, where it demonstrably finds a 48 px motion.
        // One candidate per thread: 10 serialized scores on lane 0 would also
        // be 160 feature loads with 63 lanes idle.
        float anchorScore = -1e30f;
        float2 anchorVector = predictor;
        if (li < (uint)kCentres)
        {
            float2 candidate = predictor;
            bool valid = true;
            if (li == (uint)kZeroCentre)
            {
                candidate = float2(0.0f, 0.0f);
            }
            else if (li != 0u)
            {
                // Neighbouring parent vectors (EPZS).
                const int n = (int)li - 1;
                const int idx = n < 4 ? n : n + 1;            // skip the centre
                valid = LevelIndex < TotalLevels - 1;
                if (valid)
                    candidate = SampleParentFlow(CoarsePredictor,
                        float2(groupId.xy) + float2(idx % 3 - 1, idx / 3 - 1) * 2.0f);
            }
            if (valid)
            {
                anchorVector = candidate;
                anchorScore = NeuralAnchorKey(
                    LearnedScore(candidate, blockOrigin, levelMax), candidate, reference);
            }
        }
        float anchorBest;
        float2 anchor;
        GroupBest(anchorScore, anchorVector, wave, waveCount, anchorBest, anchor);

        // Zero is one of the anchors, so there is no second window around it:
        // two windows are two hypotheses, not two samples of one, and a
        // softmax blend of "moved by twelve" and "did not move" returns six,
        // a vector neither window supports.
        float2 vector;
        LearnedWindow(anchor, radius, blockOrigin, levelMax, li, wave, waveCount, vector);
        if (li == 0) OutputFlow[groupId.xy] = float4(vector, confidence);
        return;
    }

    // --- Stage 2: centre candidates ------------------------------------------
    if (li < (uint)kCentres)
    {
        float2 candidate = predictor;
        bool valid = true;
        if (li == (uint)kZeroCentre)
        {
            // "Did not move", scored on the same SAD as every other
            // candidate. Always offered, so a coarse level that locked onto
            // the wrong vector can be abandoned here instead of needing a
            // second full window around zero later.
            candidate = float2(0.0f, 0.0f);
        }
        else if (li != 0u)
        {
            // Neighbouring parent blocks (EPZS): an object that straddles
            // blocks at the coarse level is often carried by a neighbour.
            //
            // Slot 0 is the predictor itself. It used to fall through to
            // this branch with n = -2, which sampled a parent three blocks
            // away instead, so the predictor was never scored as a centre.
            const int n = (int)li - 1;
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
    // Ranking cost of a centre: its SAD, with "did not move" charged the
    // one-luma-level margin so an ambiguous block goes to motion.
    uint centreCost[kCentres];
    [unroll]
    for (int ci = 0; ci < kCentres; ++ci)
    {
        // The bias is signed and the cost is unsigned, so bias the *others*
        // upward rather than zero downward: equivalent ordering, and no
        // wrap at a block whose SAD is smaller than the bias (a perfect
        // match on identical frames scores exactly zero, which is precisely
        // the case this has to get right).
        centreCost[ci] = CentreValid[ci]
            ? CentreSad[ci] + (ci == kZeroCentre ? 0u : (uint)(-kZeroBiasFixed))
            : 0xFFFFFFFFu;
    }
    uint bestCost = 0xFFFFFFFFu;
    [unroll]
    for (int i = 0; i < kCentres; ++i) bestCost = min(bestCost, centreCost[i]);

    // Candidates within one luma level of the best are not separable by this
    // measurement; the nearest to `reference` wins, so repeating or dense
    // texture keeps its decision across frames and the warp does not pop
    // between alias peaks. Same band and same reference as the window stage.
    float2 centre = predictor;
    float bestDist2 = 1e30f;
    [unroll]
    for (int i = 0; i < kCentres; ++i)
    {
        // The ambiguity band, not the zero bias - the two happen to be the
        // same magnitude and were sharing a constant.
        if (centreCost[i] > bestCost + kCentreBand) continue;
        const float2 dv = float2(CentreVec[i]) - reference;
        const float dist2 = dot(dv, dv);
        if (dist2 < bestDist2)
        {
            bestDist2 = dist2;
            centre = float2(CentreVec[i]);
        }
    }
    if (!searchWindow)
    {
        if (li == 0) OutputFlow[groupId.xy] = float4(centre, confidence);
        return;
    }

    // --- Stage 3: window around the winning centre ------------------------
    float2 vector;
    SearchWindow(centre, radius, reference, blockOrigin, packedMax, li, wave, waveCount,
                 keptScale, vector);

    // --- Stage 4: subpixel refinement -------------------------------------
    // The window can only return a whole number of pixels, and on text a
    // whole number is not enough. A browser scrolls by a fractional number
    // of pixels per frame, so rounding every block to the nearest pixel
    // turns a smooth scroll into a shimmer: on a page scrolling by
    // (5.5, 2.5) this engine measured 19.1 dB against 27.7 dB for not
    // interpolating at all, while the neural engine - whose soft-argmax is
    // continuous - measured 40.9 dB on the same pair.
    //
    // A parabola through the SAD at the winner and its four axis
    // neighbours recovers the fraction for five more block comparisons,
    // about 3% of the window that already ran. The fit is only used where
    // the three samples are actually convex; a flat or monotone triple
    // means there is no minimum to locate between them and the integer
    // answer stands.
    const int2 best = (int2)round(vector);
    if (li < 5u)
    {
        const int2 offsets[5] = {int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
        CentreVec[li] = best + offsets[li];
        CentreSad[li] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (int t = (int)li; t < 5 * kBlock; t += kThreads)
    {
        const int c = t / kBlock;
        const int row = t % kBlock;
        const int2 origin = blockOrigin + CentreVec[c] + int2(0, row);
        uint sad = 0u;
        [unroll]
        for (int q = 0; q < kQuads; ++q)
            sad += SadWord(RefWords[row][q], EdgeWords[row][q],
                           LoadCandWord(origin + int2(q * 4, 0), packedMax)).x;
        InterlockedAdd(CentreSad[c], sad);
    }
    GroupMemoryBarrierWithGroupSync();
    const float2 subpixel = float2(
        ParabolicMinimum((float)CentreSad[1], (float)CentreSad[0], (float)CentreSad[2]),
        ParabolicMinimum((float)CentreSad[3], (float)CentreSad[0], (float)CentreSad[4]));

    if (li == 0) OutputFlow[groupId.xy] = float4(float2(best) + subpixel, confidence);
}
