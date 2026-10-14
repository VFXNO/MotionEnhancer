#include "Synthesis.hlsli"

// Interpolation consumes the estimator's forward and backward vector fields
// directly. No per-pixel matching, vector reselection, or refining pass.
// The warp itself (field conventions, trajectory solve) is WarpSources in
// Synthesis.hlsli.
//
// The one exception is screen-fixed UI, and only where UiMaskCS has
// confirmed it over several frame pairs (unbound, it reads 0 and this pass
// is exactly the plain warp):
//   * a UI pixel is shown where it is, not warped: the time blend of the
//     two frames at that same pixel, which for an unchanged pixel is the
//     pixel itself;
//   * a warped sample that lands on UI is not the moving content it is
//     supposed to fetch - the content is underneath the UI in that frame -
//     so it is dropped and the other frame supplies the pixel. Without
//     this, background moving past a HUD drags copies of the HUD with it.
// This is not a per-pixel guess between candidate vectors: the mask is a
// temporally confirmed fact about which pixels do not move, and it changes
// nothing elsewhere.
//
// Dropping a sample whose source lies outside its frame (the clamped edge
// pixel) was measured as well and is not done: it is only as good as the
// vectors at the frame edge, which are the least reliable in the field, and
// it cost SAD 0.3-1.5 dB on pans where the plain blend hides them.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> UiMask : register(t4);
Texture2D<uint> Projection : register(t5);   // two-field pairs: ProjectSplatCS's winners at time t
RWTexture2D<float4> Output : register(u0);

// Confirmed UI pixels (of 9, including this one) a pixel needs around it to
// be held in place: itself and one neighbour, so a lone confirmed pixel is
// still left alone. At 4, a 1-2 px stroke - HUD text, a thin icon outline -
// has only about three confirmed pixels in any 3x3 and was never held: it
// was warped with the background behind it and broke into pixel fragments
// (text on a translucent panel over a panning background: HUD +1.2 dB with
// FFX, +0.5 dB with DIS at 2; the rest of the frame and the sequences
// without UI unchanged).
static const float kUiSupport = 2.0f;

// Trajectory error (pixels) at which a sample counts as half valid. An
// occlusion leaves a residual of the relative motion times t - 10-25 px for
// a character the camera orbits - while the field inside a moving object is
// only accurate to a pixel or two. At 1 px those small errors tipped
// ordinary moving objects to one frame (-0.3..-0.5 dB); at 5 px they keep
// the plain blend and an occlusion still drops to a few percent.
static const float kTrajectoryScale = 5.0f;
// How far either side of a pixel to look for the vectors of an edge's two
// sides, and how much better (pixels of residual) one must close to replace
// the plain warp's vector.
static const float kEdgeReach = 8.0f;
static const float kFarReach = 24.0f;
static const float kSwitchMargin = 0.25f;
// Soft choice at a near tie (ClosestTrajectory): the runner-up's sample is
// blended in with weight exp(-((r2 - r1) / kTieWidth)^2) when the best
// trajectory itself closes worse than kTieGate pixels (an occlusion band,
// not the clean field elsewhere).
static const float kTieWidth = 1.0f;
static const float kTieGate = 0.5f;
// How far (px) one trajectory step may move a projected vector and still be
// taken (see the projection in CSMain).
static const float kRefineReach = 3.0f;
// Brightness disagreement (levels) between a pair's two samples at which a
// candidate pair counts e^-1 in the choice between the projected pair and
// the trajectory-search pair.
static const float kAgreement = 8.0f;
// Below this disagreement (levels) between the projected pair's samples the
// trajectory search is skipped: the projected pair is already right, and
// that is most of the frame (the pass went from 1.2 to about 0.6 ms at
// 1440p with no measurable change in quality).
static const float kSearchBelow = 2.0f;

// Debug view 4: one arrow per kArrowCell pixels, drawn at a quarter of the
// vector and capped below the cell size so arrows never run into each
// other; a capped one keeps its direction.
static const int kArrowCell = 16;
static const float kArrowScale = 0.25f;
static const float kArrowMaxLength = 14.0f;

float SegmentDistance(float2 p, float2 a, float2 b)
{
    const float2 ab = b - a;
    const float along = saturate(dot(p - a, ab) / max(dot(ab, ab), 1e-6f));
    return length(p - (a + along * ab));
}

// Occlusion. Next to an object moving against the background, the
// background point a pixel needs is hidden in one of the two frames - behind
// the object - and in that frame no point of the field actually travels to
// this pixel at this time: the fixed point the warp solves,
// p + t * F(p) = pixel, has no solution there, and the two iterations end on
// the object, whose sample is then blended in at full weight. That leaves a
// half-transparent copy of the object beside it, as wide as the relative
// motion: tracking a target during a 20 px/frame turn doubled its head and
// body.
//
// So each sample is weighted by how well its own trajectory closes: the
// residual |p + t * F(p) - pixel| at the frame0 position the warp ended on,
// and the same for frame1 with (1 - t) and B. A sample on a consistent
// trajectory has a residual of a fraction of a pixel; one taken from the
// wrong side of an occlusion is off by the relative motion. The weight falls
// off smoothly, so there is no per-pixel switch to flicker, and where both
// trajectories close (everywhere but occlusions) the plain time blend
// stands.
//
// A round trip (F(p) + B(p + F(p)) = 0) was tried first and is the wrong
// test: it asks whether the sampled point is visible in both frames, and
// the object the bad sample lands on is - it is just not this pixel.
float TrajectoryWeight(float residual)
{
    const float r = residual / kTrajectoryScale;
    return rcp(1.0f + r * r);
}

// The trajectory through `flow` (scaled by s = t or 1 - t) that best closes
// at `pixel`. The field is smooth across an object's edge - block vectors
// bilinearly blended, a 50 px background motion against a still character
// passes through every value in between over a few pixels - and in that
// ramp the fixed point the warp solves has no consistent solution for
// either side, so both samples come out wrong and the blend ghosts. The
// actual vectors on either side are close by: the field a few pixels away
// in each direction. Each is tried as a trajectory, v -> pixel - s * v, and
// kept if it closes better (|s * (flow(pixel - s * v) - v)|). This is a
// geometric test, not a photometric one: in uniform motion every candidate
// is the same vector and nothing can switch, and at an edge the vector that
// wins is the one the field itself confirms at the place it points to.
//
// The runner-up - the best clearly different vector (more than half a pixel
// from the winner) and its residual - is returned as well. In an occlusion
// band two candidates often close almost equally, and a hard choice made
// per pixel there picks one or the other from pixel to pixel: the band
// came out as blocky fragments of both sources ("pixelated noise" beside a
// turning character). The caller blends the two samples by how close the
// tie is instead (kTieWidth, kTieGate). Measured on the third-person 4x
// test: high-frequency noise in the 1-12 px outside the character -12%,
// that band +0.1 dB, the character's exact silhouette +0.14 dB, temporal
// error unchanged. Blending every near tie, not only where the best
// trajectory closes badly, removed more noise (-17%) but also blended two
// valid vectors in the clean background and raised the whole frame's
// temporal error 3%.
float2 ClosestTrajectory(Texture2D<float4> flow, float2 pixel, float s, out float residual,
                         out float2 runnerUp, out float runnerUpResidual)
{
    float2 best = TrackFlow(flow, pixel, s);
    residual = s * length(SampleFlow(flow, pixel - s * best) - best);
    runnerUp = best;
    runnerUpResidual = 1e30f;
    // Two reaches: a few pixels covers the bilinear ramp of the block grid;
    // the band uncovered beside a fast object is as wide as its relative
    // motion, so the far side is also looked for further out.
    const float2 directions[4] = {float2(-1.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, -1.0f), float2(0.0f, 1.0f)};
    [unroll]
    for (int k = 0; k < 8; ++k)
    {
        const float2 v = SampleFlow(flow, pixel + directions[k & 3] * (k < 4 ? kEdgeReach : kFarReach));
        const float r = s * length(SampleFlow(flow, pixel - s * v) - v);
        if (r < residual - kSwitchMargin)
        {
            if (length(best - v) > 0.5f)
            {
                runnerUp = best;
                runnerUpResidual = residual;
            }
            best = v;
            residual = r;
        }
        else if (r < runnerUpResidual && length(best - v) > 0.5f)
        {
            runnerUp = v;
            runnerUpResidual = r;
        }
    }
    return best;
}

// Confirmed UI under a sample - for a sample that moves, within 1.5 px of
// it. A moving trajectory must not take its colour from static UI, and the
// UI's antialiased rim changes with the background behind it, so it is
// never confirmed itself: read as background, it was dragged across a camera
// turn as white specks beside every icon. A trajectory that stays put reads
// the pixel under it, UI or not.
float UiUnder(float2 position, float2 pixel, float2 inverseSize)
{
    const float2 uv = (position + 0.5f) * inverseSize;
    float coverage = UiCoverage(UiMask.SampleLevel(LinearClamp, uv, 0));
    const float2 moved = position - pixel;
    if (dot(moved, moved) > 1.0f)
    {
        const float2 o = 1.5f * inverseSize;
        coverage = max(coverage, UiCoverage(UiMask.SampleLevel(LinearClamp, uv + float2(o.x, o.y), 0)));
        coverage = max(coverage, UiCoverage(UiMask.SampleLevel(LinearClamp, uv + float2(-o.x, o.y), 0)));
        coverage = max(coverage, UiCoverage(UiMask.SampleLevel(LinearClamp, uv + float2(o.x, -o.y), 0)));
        coverage = max(coverage, UiCoverage(UiMask.SampleLevel(LinearClamp, uv + float2(-o.x, -o.y), 0)));
    }
    return coverage;
}

// Confirmed UI within 16 px of the pixel.
bool NearUi(int2 pixel)
{
    const int2 limit = int2(Width, Height) - 1;
    [unroll]
    for (int r = 0; r < 2; ++r)
    {
        [unroll]
        for (int k = 0; k < 8; ++k)
        {
            const int2 at = clamp(pixel + int2(round((6.0f + 10.0f * r) * kSurroundDirections[k])), int2(0, 0), limit);
            if (UiCoverage(UiMask.Load(int3(at, 0))) > 0.0f) return true;
        }
    }
    return false;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    float2 from0 = pixel, from1 = pixel;
    float3 c0, c1;
    float trust0 = 1.0f, trust1 = 1.0f;
    bool search = false;     // run the trajectory search
    bool hybrid = false;     // ... as the second candidate next to the projected pair
    if (FlowScale == 0.0f)
    {
        // The projected fields: each frame's sample comes from the block of
        // that frame that is at this pixel at time t, and a frame none of
        // whose blocks is here does not show this point - its sample is
        // left out. Only where neither frame has a block here (rare: both
        // fields spread apart) does the trajectory search below decide.
        //
        // Where both frames have a block here and their samples disagree by
        // more than kSearchBelow, the trajectory search's pair is computed
        // as well and the two results are blended by how well
        // each pair's two samples agree - the right motion shows the same
        // surface in both frames (AMD FSR 3 frame interpolation chooses
        // between its game-vector and optical-flow results the same way).
        // A projected vector painted wherever it lands, so one wrong vector
        // at an edge drew its colour elsewhere (a hair-coloured ring beside a
        // head); the search's pair is exact inside a surface. Where one
        // frame has no block here, that is the occlusion the projection is
        // for, and its single sample is taken. Measured on the third-person
        // 4x test against the trajectory search alone: the band beside the
        // character +0.5 dB, the character +0.4 dB, the rest of the frame
        // +0.6 dB, step-to-step temporal error 3.03 -> 2.74, high-frequency
        // noise in the band -16%. The projection alone was ahead on the
        // background and the temporal error but 0.7 dB behind on the
        // character.
        //
        // The priority only decides which block wins a place; a winner is
        // then trusted fully. Down-weighting winners by their priority
        // distrusted exactly the background still visible at time t that is
        // covered later (its round trip fails, and it is still the right
        // sample here), and blended both frames across wide bands beside a
        // character.
        bool found0, found1;
        float2 v0 = ProjectedVector(Projection, ForwardFlow, 0u, t, pixel, found0);
        float2 v1 = ProjectedVector(Projection, BackwardFlow, 1u, 1.0f - t, pixel, found1);
        // The winner is one block's vector, constant over 2 px; one step of
        // the trajectory solve on the bilinear field makes it continuous
        // inside the surface it chose, and is kept only while it stays on
        // that surface.
        const float2 r0 = SampleFlow(ForwardFlow, pixel - t * v0);
        const float2 r1 = SampleFlow(BackwardFlow, pixel - (1.0f - t) * v1);
        if (length(r0 - v0) < kRefineReach) v0 = r0;
        if (length(r1 - v1) < kRefineReach) v1 = r1;
        if (found0 || found1)
        {
            from0 = pixel - t * v0;
            from1 = pixel - (1.0f - t) * v1;
            trust0 = found0 ? 1.0f : 0.0f;
            trust1 = found1 ? 1.0f : 0.0f;
            c0 = SampleSharp(Frame0, from0);
            c1 = SampleSharp(Frame1, from1);
            // Samples that already agree leave the search nothing to add,
            // and skipping it there is most of the frame.
            hybrid = found0 && found1 && MaxComponent(abs(c0 - c1)) * kLumaScale > kSearchBelow;
            search = hybrid;
        }
        else
        {
            search = true;
        }
    }
    else
    {
        WarpSources(ForwardFlow, BackwardFlow, pixel, t, from0, from1);
        c0 = SampleSharp(Frame0, from0);
        c1 = SampleSharp(Frame1, from1);
    }

    // The trajectory search, from one call site (called from two branches,
    // the shader's register use doubled the pass's time).
    float2 fromB0 = pixel, fromB1 = pixel;
    float3 b0 = float3(0.0f, 0.0f, 0.0f), b1 = float3(0.0f, 0.0f, 0.0f);
    float trustB0 = 1.0f, trustB1 = 1.0f;
    if (search)
    {
        float residual0, residual1, runnerUpResidual0, runnerUpResidual1;
        float2 runnerUp0, runnerUp1;
        fromB0 = pixel - t * ClosestTrajectory(ForwardFlow, pixel, t, residual0, runnerUp0, runnerUpResidual0);
        fromB1 = pixel - (1.0f - t) * ClosestTrajectory(BackwardFlow, pixel, 1.0f - t, residual1,
                                                       runnerUp1, runnerUpResidual1);
        trustB0 = TrajectoryWeight(residual0);
        trustB1 = TrajectoryWeight(residual1);
        b0 = SampleSharp(Frame0, fromB0);
        b1 = SampleSharp(Frame1, fromB1);
        if (!hybrid)
        {
            // A near tie in an occlusion band: blend the runner-up's sample
            // in by how close the tie is (see ClosestTrajectory).
            const float tie0 = (runnerUpResidual0 - residual0) / kTieWidth;
            const float tie1 = (runnerUpResidual1 - residual1) / kTieWidth;
            if (residual0 > kTieGate && tie0 < 3.0f)
            {
                const float w = exp(-tie0 * tie0);
                b0 = (b0 + w * SampleSharp(Frame0, pixel - t * runnerUp0)) / (1.0f + w);
            }
            if (residual1 > kTieGate && tie1 < 3.0f)
            {
                const float w = exp(-tie1 * tie1);
                b1 = (b1 + w * SampleSharp(Frame1, pixel - (1.0f - t) * runnerUp1)) / (1.0f + w);
            }
            from0 = fromB0;
            from1 = fromB1;
            c0 = b0;
            c1 = b1;
            trust0 = trustB0;
            trust1 = trustB1;
        }
    }
    const float2 uv0 = (from0 + 0.5f) * inverseSize;
    const float2 uv1 = (from1 + 0.5f) * inverseSize;
    float3 color = lerp(c0, c1, t);

    // How much each sample is the content it is meant to be: not on UI, and
    // on a trajectory that actually reaches this pixel.
    float valid0 = (1.0f - UiUnder(from0, pixel, inverseSize)) * trust0;
    float valid1 = (1.0f - UiUnder(from1, pixel, inverseSize)) * trust1;
    if (valid0 < 1.0f || valid1 < 1.0f)
    {
        const float w0 = (1.0f - t) * valid0;
        const float w1 = t * valid1;
        // Neither sample usable: nothing better to fetch, keep the plain blend.
        if (w0 + w1 > 1e-4f) color = (w0 * c0 + w1 * c1) / (w0 + w1);
    }
    if (hybrid)
    {
        const float vb0 = (1.0f - UiUnder(fromB0, pixel, inverseSize)) * trustB0;
        const float vb1 = (1.0f - UiUnder(fromB1, pixel, inverseSize)) * trustB1;
        const float wb0 = (1.0f - t) * vb0, wb1 = t * vb1;
        const float3 colorB = wb0 + wb1 > 1e-4f ? (wb0 * b0 + wb1 * b1) / (wb0 + wb1) : lerp(b0, b1, t);
        const float dA = MaxComponent(abs(c0 - c1)) * kLumaScale / kAgreement;
        const float dB = MaxComponent(abs(b0 - b1)) * kLumaScale / kAgreement;
        const float wA = exp(-dA * dA), wB = exp(-dB * dB);
        color = wA + wB > 1e-6f ? (wA * color + wB * colorB) / (wA + wB) : 0.5f * (color + colorB);
    }
    // Background beside static UI whose own pair of samples disagrees: the
    // FFX blocks and refined vectors around a static icon lock onto the icon
    // (zero motion), and the background right beside it blended two frames'
    // backgrounds - a dark, frozen rectangle next to each icon through a
    // camera turn. UiRepairCS retries those pixels with the motion of the
    // scene around them; here they are only flagged, in the output's alpha
    // (the pair's mismatch, below 1), because the retry loop in this shader
    // made every pixel slower (0.20 -> 0.34 ms at 1440p) even where it never
    // ran. Only within 16 px of confirmed UI and where nothing is held.
    float repairFlag = 1.0f;
    if (NearUi(int2(id.xy)))
    {
        float pairMismatch = MaxComponent(abs(c0 - c1)) * kLumaScale;
        if (hybrid) pairMismatch = min(pairMismatch, MaxComponent(abs(b0 - b1)) * kLumaScale);
        if (pairMismatch > 2.0f * kAgreement) repairFlag = min(pairMismatch, 254.0f) / 255.0f;
    }
    // A confirmed pixel is held in place only inside a patch of UI: at least
    // kUiSupport of its 3x3 neighbourhood confirmed too. Isolated confirmed
    // pixels are noise in the evidence, and holding them still while every
    // neighbour is warped speckles the image.
    float uiHere = UiCoverage(UiMask.Load(int3(id.xy, 0)));
    if (uiHere > 0.0f)
    {
        const int2 limit = int2(Width, Height) - 1;
        float support = 0.0f;
        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
                support += UiCoverage(UiMask.Load(int3(clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), limit), 0)));
        uiHere *= saturate(support - (kUiSupport - 1.0f));
    }
    if (uiHere > 0.0f)
    {
        repairFlag = 1.0f;
        const float3 still = lerp(Frame0.Load(int3(id.xy, 0)).rgb, Frame1.Load(int3(id.xy, 0)).rgb, t);
        color = lerp(color, still, uiHere);
    }

    const int view = DebugView();
    if (view > 0)
    {
        const float3 grey = DebugGrey(color);
        if (view == 1)
        {
            // The motion this pixel was interpolated along, frame0 -> frame1:
            // hue = direction, strength = speed (full at 48 px per frame).
            const float2 motion = from1 - from0;
            const float hue = atan2(motion.y, motion.x) / 6.2831853f + 0.5f;
            color = lerp(grey, DebugHue(hue), 0.85f * saturate(length(motion) / 48.0f));
        }
        else if (view == 2)
        {
            // Static-UI evidence: yellow while it builds, green once
            // confirmed (light green where too isolated to be held).
            const float count = DecodeUiCount(UiMask.Load(int3(id.xy, 0)));
            if (count >= kUiConfirm) color = lerp(grey, float3(0.0f, 1.0f, 0.0f), uiHere > 0.0f ? 0.75f : 0.35f);
            else if (count > 0.0f) color = lerp(grey, float3(1.0f, 0.85f, 0.0f), 0.5f);
            else color = grey;
        }
        else if (view == 3)
        {
            // Occlusion: red where frame0's sample does not reach this
            // pixel and is weighted down, blue the same for frame1.
            const float3 tint = float3(1.0f - trust0, 0.0f, 1.0f - trust1);
            color = saturate(lerp(grey, tint, 0.85f * max(tint.x, tint.z)));
        }
        else
        {
            // Vector arrows: the estimated field frame0 -> frame1, sampled
            // at the centre of every kArrowCell cell and drawn from there,
            // coloured by direction as in view 1. A pixel looks at the
            // cells near enough for their arrow to reach it.
            color = grey;
            const int reach = (int)ceil(kArrowMaxLength / kArrowCell);
            const int2 home = int2(id.xy) / kArrowCell;
            float nearest = 1e30f;
            float3 arrowColor = float3(0.0f, 0.0f, 0.0f);
            for (int cy = -reach; cy <= reach; ++cy)
            {
                for (int cx = -reach; cx <= reach; ++cx)
                {
                    const float2 start = float2((home + int2(cx, cy)) * kArrowCell) + 0.5f * kArrowCell;
                    float2 v = kArrowScale * (FlowScale != 0.0f ? -SampleFlow(BackwardFlow, start) : SampleFlow(ForwardFlow, start));
                    const float speed = length(v);
                    if (speed > kArrowMaxLength) v *= kArrowMaxLength / speed;
                    const float2 tip = start + v;
                    float d = length(pixel - start) - 1.0f;                  // the dot at the start
                    if (speed > 0.5f)
                    {
                        const float2 back = -normalize(v) * min(4.0f, 0.5f * length(v) + 1.5f);
                        const float2 side = float2(-back.y, back.x) * 0.5f;
                        d = min(d, SegmentDistance(pixel, start, tip));
                        d = min(d, SegmentDistance(pixel, tip, tip + back + side));
                        d = min(d, SegmentDistance(pixel, tip, tip + back - side));
                    }
                    if (d < nearest)
                    {
                        nearest = d;
                        arrowColor = speed > 0.5f ? DebugHue(atan2(v.y, v.x) / 6.2831853f + 0.5f) : float3(1.0f, 1.0f, 1.0f);
                    }
                }
            }
            if (nearest < 0.75f) color = arrowColor;
            else if (nearest < 1.5f) color = float3(0.0f, 0.0f, 0.0f);   // outline, readable on any picture
        }
        const bool real = t < 0.01f || t > 0.99f;
        color = DebugMarker(id.xy, color, real ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 1.0f));
        repairFlag = 1.0f;
    }
    Output[id.xy] = float4(color, repairFlag);
}
