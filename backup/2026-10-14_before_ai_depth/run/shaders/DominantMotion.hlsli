// The frame's few dominant motions, as candidates for every block or patch
// (DominantMotionCS, for DIS: the next coarser level's field).
//
// Coarse-to-fine estimation loses an object that is small next to its own
// motion: at the coarse levels it shrinks into the background around it,
// that background's motion is the only start its blocks are offered, and
// each finer level searches only a few pixels around that start, so the
// object never gets its own motion back (HMFlow, ICPR 2020; the same
// weakness is documented for 3-D recursive search, where "the small object
// is submerged in the background"). A character's head during a camera turn
// is exactly that. The remedy in the literature is a non-local candidate: a
// motion offered from outside the block's neighbourhood. Here those are the
// strongest peaks of the whole field's motion histogram - in a camera turn
// around a character, the world's motion and the character's - and a block
// keeps one only when it matches its pictures clearly better than its own
// start, so a wrong one costs nothing but the test.
//
// One 32x32 group: 4096 samples of the field on a regular grid, a 64x64
// histogram of BinSize-wide bins centred on zero motion, then kDominant
// peaks, each followed by suppressing the 5x5 bins around it. A peak holding
// fewer than kMinShare of the samples is not written. Output: kDominant x 1
// texels of R32G32_FLOAT, the bin centre in the field's own units, or
// kNoDominant.
//
// Constants: FlowScale = bin width (the field's units). The including shader
// includes Common.hlsli and defines FieldAt(texel) and FieldSize() first.

static const uint kDominant = 4;
static const uint kBins = 64;
static const float kMinShare = 0.01f;
static const float kNoDominant = 1e30f;

groupshared uint Histogram[kBins * kBins];
groupshared uint Peak;

RWTexture2D<float2> Dominant : register(u0);

void FindDominant(uint li)
{
    for (uint b = li; b < kBins * kBins; b += 1024u) Histogram[b] = 0u;
    if (li == 0u) Peak = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint2 size = FieldSize();
    const float binWidth = max(FlowScale, 1e-3f);
    [unroll]
    for (uint s = 0u; s < 4u; ++s)
    {
        const uint i = li * 4u + s;                       // 0..4095 on a 64x64 grid
        const uint2 at = min(uint2((float2(i % 64u, i / 64u) + 0.5f) / 64.0f * float2(size)), size - 1u);
        const float2 v = FieldAt(at);
        const int2 bin = int2(floor(v / binWidth + 0.5f)) + int2(kBins / 2, kBins / 2);
        if (all(bin >= 0) && all(bin < int(kBins)))
            InterlockedAdd(Histogram[bin.y * kBins + bin.x], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint minCount = (uint)(kMinShare * 4096.0f);
    for (uint k = 0u; k < kDominant; ++k)
    {
        // Count in the high bits, bin index in the low 12: the largest key
        // is the fullest bin.
        uint best = 0u;
        for (uint b2 = li; b2 < kBins * kBins; b2 += 1024u)
            best = max(best, (Histogram[b2] << 12) | b2);
        InterlockedMax(Peak, best);
        GroupMemoryBarrierWithGroupSync();
        const uint key = Peak;
        const uint count = key >> 12;
        const uint index = key & 0xFFFu;
        const int2 bin = int2(index % kBins, index / kBins);
        if (li == 0u)
        {
            Dominant[uint2(k, 0)] = count >= minCount
                ? float2(bin - int2(kBins / 2, kBins / 2)) * binWidth
                : float2(kNoDominant, kNoDominant);
        }
        GroupMemoryBarrierWithGroupSync();
        // Suppress the peak's neighbourhood: one motion, measured a little
        // differently across the frame, is one candidate.
        if (li < 25u)
        {
            const int2 n = bin + int2(int(li % 5u) - 2, int(li / 5u) - 2);
            if (all(n >= 0) && all(n < int(kBins))) Histogram[n.y * kBins + n.x] = 0u;
        }
        if (li == 0u) Peak = 0u;
        GroupMemoryBarrierWithGroupSync();
    }
}
