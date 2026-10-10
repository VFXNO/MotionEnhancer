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
RWTexture2D<float4> Output : register(u0);

// Confirmed UI pixels (of 9, including this one) a pixel needs around it to
// be held in place.
static const float kUiSupport = 4.0f;

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
float2 ClosestTrajectory(Texture2D<float4> flow, float2 pixel, float s, out float residual)
{
    float2 best = TrackFlow(flow, pixel, s);
    residual = s * length(SampleFlow(flow, pixel - s * best) - best);
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
            best = v;
            residual = r;
        }
    }
    return best;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    float2 from0, from1;
    float residual0 = 0.0f, residual1 = 0.0f;
    if (FlowScale == 0.0f)
    {
        from0 = pixel - t * ClosestTrajectory(ForwardFlow, pixel, t, residual0);
        from1 = pixel - (1.0f - t) * ClosestTrajectory(BackwardFlow, pixel, 1.0f - t, residual1);
    }
    else
    {
        WarpSources(ForwardFlow, BackwardFlow, pixel, t, from0, from1);
    }
    const float2 uv0 = (from0 + 0.5f) * inverseSize;
    const float2 uv1 = (from1 + 0.5f) * inverseSize;
    const float3 c0 = SampleSharp(Frame0, from0);
    const float3 c1 = SampleSharp(Frame1, from1);
    float3 color = lerp(c0, c1, t);

    // How much each sample is the content it is meant to be: not on UI, and
    // on a trajectory that actually reaches this pixel.
    float valid0 = 1.0f - UiCoverage(UiMask.SampleLevel(LinearClamp, uv0, 0));
    float valid1 = 1.0f - UiCoverage(UiMask.SampleLevel(LinearClamp, uv1, 0));
    float trust0 = 1.0f, trust1 = 1.0f;
    if (FlowScale == 0.0f)
    {
        trust0 = TrajectoryWeight(residual0);
        trust1 = TrajectoryWeight(residual1);
        valid0 *= trust0;
        valid1 *= trust1;
    }
    if (valid0 < 1.0f || valid1 < 1.0f)
    {
        const float w0 = (1.0f - t) * valid0;
        const float w1 = t * valid1;
        // Neither sample usable: nothing better to fetch, keep the plain blend.
        if (w0 + w1 > 1e-4f) color = (w0 * c0 + w1 * c1) / (w0 + w1);
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
    }
    Output[id.xy] = float4(color, 1.0f);
}
