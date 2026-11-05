#include "Common.hlsli"
#include "Photometric.hlsli"

// DIS (Dense Inverse Search, Kroeger et al., ECCV 2016), step 1 of 2 at one
// pyramid level: the motion of every 8x8 patch, on a grid of patches every
// BlockSize pixels (they overlap), by inverse compositional Lucas-Kanade.
// One 8x8 thread group per patch, one thread per patch pixel (SM 6.0 wave
// reductions).
//
// Inverse compositional: the template (frame0's patch) is fixed, so its
// gradients and their 2x2 Hessian are computed once, and each iteration
// only samples frame1 at the current displacement and solves
//     H du = sum(grad T * error),   u <- u - du.
// The patch and its sample are both mean-normalised - the cost is the SSD
// of their deviations from their own means - so a brightness offset
// between the frames changes nothing, and the gradients are centred to
// match (the mean term's Jacobian is the mean gradient). A brightness gain
// is removed beforehand by the pair's photometric mapping
// (Photometric.hlsli), the same one SAD's packing uses.
//
// The start is the next coarser level's densified flow at the patch centre
// (zero at the coarsest level), so reach comes from the pyramid as in every
// coarse-to-fine scheme. The answer is continuous: no integer search, no
// subpixel fit. If the final error is worse than the start's, the start
// stands.
//
// Samples that fall outside frame1 are the clamped edge, which a patch can
// match by sliding into the smear: on a 41 px pan the border band carried
// nearly all of DIS's error. A patch whose coarse start already puts more
// than a quarter of it outside frame1 is therefore not refined at all - it
// keeps the coarse flow, which is the extrapolation a frame edge needs.
// Masking the outside samples out of the update instead was measured and
// is worse: a patch touching the edge then mean-normalises over a different
// set of pixels than its template, which biases it toward smaller motion,
// and the pyramid spread that across the flat sky of an 88 px pan (0.9 ->
// 16 px mean error).
//
// Constants: LevelWidth/Height = this level, BlockSize = patch stride,
// SearchRadius = iterations, LevelIndex / TotalLevels.
Texture2D<float> RefLuma : register(t0);     // frame0 at this level (0..255)
Texture2D<float> CandLuma : register(t1);    // frame1 at this level
Texture2D<float2> CoarseFlow : register(t2); // dense flow of the next coarser level (its pixels)
Texture2D<float2> RefStats : register(t3);   // whole-frame luma mean, deviation (FrameStatsCS)
Texture2D<float2> CandStats : register(t4);
// A second coarse field scored as further starts when NeuralControl.z is set:
// the AI engine's own DIS coarse levels next to the network's field in
// CoarseFlow (FlowEngine::recordAiFlow). DIS itself never binds it.
Texture2D<float2> AltFlow : register(t5);
RWTexture2D<float2> PatchFlow : register(u0);

static const int kPatch = 8;
static const int kThreads = kPatch * kPatch;
static const int kMaxWaves = 4;              // 64 threads at >= 16 lanes per wave
static const float kMaxStep = 1.0f;          // pixels per Lucas-Kanade iteration
static const float kCandidateMargin = 0.8f;  // relative error a neighbour's start must beat
static const float kZeroMargin = 0.5f;       // and the relative error zero motion must beat
groupshared float4 WaveSums[kMaxWaves];

// Group-wide sum of four floats; uniform result. One barrier pair when the
// group spans several waves, none when one wave holds it (wave64).
float4 GroupSum(float4 v, uint li)
{
    const float4 waveSum = WaveActiveSum(v);
    const uint lanes = WaveGetLaneCount();
    const uint waves = (kThreads + lanes - 1) / lanes;
    if (waves == 1) return waveSum;
    GroupMemoryBarrierWithGroupSync();            // earlier readers of WaveSums are done
    if (WaveIsFirstLane()) WaveSums[li / lanes] = waveSum;
    GroupMemoryBarrierWithGroupSync();
    float4 total = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint w = 0; w < waves; ++w) total += WaveSums[w];
    return total;
}

float RefAt(int2 p, int2 levelMax)
{
    return RefLuma.Load(int3(clamp(p, int2(0, 0), levelMax), 0));
}

// Mean-normalised SSD of the patch: i = the sample, tn = the template's
// deviation from its mean (which sums to zero).
// sum((i - mi) - tn)^2 = sum i^2 - n mi^2 - 2 sum i tn + sum tn^2.
float PatchError(float i, float tn, uint li)
{
    const float4 s = GroupSum(float4(i, i * i, i * tn, tn * tn), li);
    return s.y - s.x * s.x / (float)kThreads - 2.0f * s.z + s.w;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 levelSize = int2(LevelWidth, LevelHeight);
    const int2 levelMax = levelSize - 1;
    const float2 inverseSize = rcp(float2(levelSize));
    // The last row/column of patches is pulled inside the level.
    const int2 origin = min(int2(groupId.xy) * BlockSize, max(int2(0, 0), levelSize - kPatch));
    const int2 p = min(origin + int2(tid.xy), levelMax);
    const float2 pf = float2(p);

    float refGain, refOffset, candGain, candOffset;
    const float2 refStats = RefStats.Load(int3(0, 0, 0));
    const float2 candStats = CandStats.Load(int3(0, 0, 0));
    PairMapping(refStats, candStats, refGain, refOffset);
    PairMapping(candStats, refStats, candGain, candOffset);

    // --- template, centred gradients, Hessian ------------------------------
    const float t = refGain * RefAt(p, levelMax) + refOffset;
    const float gx = refGain * 0.5f * (RefAt(p + int2(1, 0), levelMax) - RefAt(p - int2(1, 0), levelMax));
    const float gy = refGain * 0.5f * (RefAt(p + int2(0, 1), levelMax) - RefAt(p - int2(0, 1), levelMax));
    const float4 means = GroupSum(float4(t, gx, gy, 0.0f), li) / (float)kThreads;
    const float tn = t - means.x;
    const float gxc = gx - means.y;
    const float gyc = gy - means.z;
    const float4 hessian = GroupSum(float4(gxc * gxc, gxc * gyc, gyc * gyc, 0.0f), li);
    // A ridge keeps the solve finite and honest on a flat or one-directional
    // patch: the unconstrained direction then barely moves from the start,
    // which is the coarser level's answer. The absolute part is the Hessian
    // of a patch whose gradient is one luma level per pixel - textured
    // patches sum to thousands and do not notice it. With only a token
    // ridge, one stray gradient where the warped sample touched an edge
    // threw a flat patch tens of pixels, and since flat matches flat
    // anywhere the error check accepted it: 7x the gross errors of SAD on
    // flat regions of the game scene.
    const float ridge = 1e-3f * (hessian.x + hessian.z) + (float)kThreads;
    const float hxx = hessian.x + ridge, hxy = hessian.y, hyy = hessian.z + ridge;
    const float inverseDet = rcp(hxx * hyy - hxy * hxy);

    // --- start: the best of a few candidates --------------------------------
    // The coarser level's flow at the patch centre is the natural start, but
    // it is a coarse answer: at a motion boundary or the frame edge it is the
    // blend of both sides or an extrapolation, and Lucas-Kanade only
    // refines - it cannot leave the basin it starts in. So the coarse flow
    // one patch away on each side and "did not move" are scored as well
    // (mean-normalised SSD, one sample per pixel each) and the best one is
    // refined, the same hedge SAD's centre stage makes.
    float2 u = float2(0.0f, 0.0f);
    float sample = 0.0f;
    float startError = 1e30f;
    if (LevelIndex < TotalLevels - 1)
    {
        uint cw, ch;
        CoarseFlow.GetDimensions(cw, ch);
        // Coarse pixel i sits on fine pixel 2i (PyramidCS).
        const float2 centre = float2(origin) + 0.5f * (float)(kPatch - 1);
        const float2 coarse = 2.0f * CoarseFlow.SampleLevel(LinearClamp, (0.5f * centre + 0.5f) / float2(cw, ch), 0);
        const float2 q = pf + coarse;
        const float outside = any(q < -0.5f) || any(q > float2(levelSize) - 0.5f) ? 1.0f : 0.0f;
        if (GroupSum(float4(outside, 0.0f, 0.0f, 0.0f), li).x > 0.25f * (float)kThreads)
        {
            if (li == 0) PatchFlow[groupId.xy] = coarse;
            return;
        }
        const float2 offsets[5] = {float2(0, 0), float2(-kPatch, 0), float2(kPatch, 0),
                                   float2(0, -kPatch), float2(0, kPatch)};
        [unroll]
        for (int c = 0; c < 5; ++c)
        {
            const float2 v = 2.0f * CoarseFlow.SampleLevel(
                LinearClamp, (0.5f * (centre + offsets[c]) + 0.5f) / float2(cw, ch), 0);
            const float s = candGain * CandLuma.SampleLevel(LinearClamp, (pf + v + 0.5f) * inverseSize, 0) + candOffset;
            const float e = PatchError(s, tn, li);
            // A neighbour's vector must be clearly better to replace the
            // patch's own coarse flow: on flat sky every candidate scores
            // about the same, and letting noise pick cost a landscape pan 2 dB.
            if (c == 0 || e < kCandidateMargin * startError)
            {
                u = v;
                sample = s;
                startError = e;
            }
        }
        // The AI engine: DIS's own coarse answer around the patch, which
        // replaces the network's where it is clearly better - a fast motion
        // the network missed, which two refinement steps from its start
        // could not reach, and the two frames were blended out of register
        // (a blurred picture).
        if (NeuralControl.z > 0.5f)
        {
            [unroll]
            for (int c = 0; c < 5; ++c)
            {
                const float2 v = 2.0f * AltFlow.SampleLevel(
                    LinearClamp, (0.5f * (centre + offsets[c]) + 0.5f) / float2(cw, ch), 0);
                const float s = candGain * CandLuma.SampleLevel(LinearClamp, (pf + v + 0.5f) * inverseSize, 0) + candOffset;
                const float e = PatchError(s, tn, li);
                if (e < kCandidateMargin * startError)
                {
                    u = v;
                    sample = s;
                    startError = e;
                }
            }
        }
    }
    {
        // "Did not move", last and only when strictly better, so a tie on a
        // flat patch keeps the motion around it.
        const float s = candGain * CandLuma.SampleLevel(LinearClamp, (pf + 0.5f) * inverseSize, 0) + candOffset;
        const float e = PatchError(s, tn, li);
        if (e < kZeroMargin * startError)
        {
            u = float2(0.0f, 0.0f);
            sample = s;
            startError = e;
        }
    }
    const float2 start = u;

    const int iterations = clamp(SearchRadius, 1, 64);
    [loop]
    for (int k = 0; k < iterations; ++k)
    {
        // sum(gc * error) = sum(gc * i1) - sum(gc * tn): the mean of i1 drops
        // out because the centred gradients sum to zero.
        const float4 b = GroupSum(float4(gxc * (sample - tn), gyc * (sample - tn), 0.0f, 0.0f), li);
        float2 du = inverseDet * float2(hyy * b.x - hxy * b.y, hxx * b.y - hxy * b.x);
        // The linearisation only holds within about a pixel of the current
        // estimate; a longer step is extrapolation, so it is taken in
        // pieces over the following iterations instead.
        const float stepLength = length(du);
        if (stepLength > kMaxStep) du *= kMaxStep / stepLength;
        u -= du;
        sample = candGain * CandLuma.SampleLevel(LinearClamp, (pf + u + 0.5f) * inverseSize, 0) + candOffset;
        if (dot(du, du) < 1e-4f) break;   // a hundredth of a pixel
    }

    const float finalError = PatchError(sample, tn, li);
    if (!(finalError <= startError) || any(abs(u) > float2(levelSize))) u = start;

    if (li == 0) PatchFlow[groupId.xy] = u;
}
