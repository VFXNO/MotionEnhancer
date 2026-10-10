#include "Common.hlsli"

// What InterpolateCS and UiMaskCS share: where the synthesis reads each
// source frame for an output pixel, and how the static-UI evidence is
// encoded. UiMaskCS decides "would interpolation damage this pixel?" by
// running exactly the warp the output uses, so the two must never diverge -
// hence one definition here rather than a copy in each pass.

// ---------------------------------------------------------------- the warp

float2 SampleFlow(Texture2D<float4> flow, float2 pixel)
{
    const int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
    const float2 position = (pixel - (BlockSize - 1) * 0.5f) / BlockSize;
    const int2 base = int2(floor(position));
    const float2 fraction = frac(position);
    const float2 v00 = flow.Load(int3(clamp(base, int2(0, 0), gridMax), 0)).xy;
    const float2 v10 = flow.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0)).xy;
    const float2 v01 = flow.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0)).xy;
    const float2 v11 = flow.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0)).xy;
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

// Flow of the scene point that is at `pixel` at time t, for a field defined
// on that field's own frame grid.
//
// ForwardFlow lives on frame0's grid: F(p) displaces p from frame0 to
// frame1, so the point visible at `pixel` at time t is the p with
// p + t*F(p) = pixel, i.e. p = pixel - t*F(p) - a fixed point, not a direct
// lookup. Reading F(pixel) and stopping there (what this pass used to do)
// asks the field for the motion at the point's *destination* instead of at
// the point: on a 16 px motion at t = 0.5 that samples 8 px away, so the
// vector an object's own block voted for gets applied to the background
// beside it and vice versa. That mismatch is what tears object edges.
//
// Two Picard iterations converge wherever the field is Lipschitz below
// 1/t, which covers everything except the boundary blocks themselves, and
// each one costs four loads of a (Width/8 x Height/8) texture that is
// resident in cache. Nothing here rejects, blends or smooths a vector: the
// estimate is only read at the right place.
float2 TrackFlow(Texture2D<float4> flow, float2 pixel, float scale)
{
    float2 v = SampleFlow(flow, pixel);
    v = SampleFlow(flow, pixel - scale * v);
    v = SampleFlow(flow, pixel - scale * v);
    return v;
}

// Positions (in pixels) to read frame0 and frame1 at for output `pixel` at
// time t.
//
// Field convention (FlowScale selects it; see FlowEngine::Entry::singleField):
//   FlowScale == 0  two independently estimated fields. ForwardFlow lives on
//                   frame0's grid (frame0 -> frame1); BackwardFlow lives on
//                   frame1's grid (frame1 -> frame0). Each needs its own
//                   trajectory solve. This is what the SAD and neural graphs
//                   produce - two separate searches with the roles swapped.
//   FlowScale != 0  one field, negated. FFX estimates current -> previous
//                   only, so "forward" is exactly -backward and both are
//                   indexed on frame1's grid. There is a single trajectory,
//                   and solving the frame0-grid fixed point for a field that
//                   is not on frame0's grid is simply the wrong equation -
//                   it cost ~1.3 dB on a pure global shift before this split.
void WarpSources(Texture2D<float4> forwardFlow, Texture2D<float4> backwardFlow,
                 float2 pixel, float t, out float2 from0, out float2 from1)
{
    if (FlowScale != 0.0f)
    {
        // One trajectory: q = pixel - (1-t)*B(q) is the point's position in
        // frame1, and its frame0 position is q + B(q) = pixel + t*B(q).
        const float2 backward = TrackFlow(backwardFlow, pixel, 1.0f - t);
        from1 = pixel - (1.0f - t) * backward;
        from0 = pixel + t * backward;
    }
    else
    {
        from0 = pixel - t * TrackFlow(forwardFlow, pixel, t);
        from1 = pixel - (1.0f - t) * TrackFlow(backwardFlow, pixel, 1.0f - t);
    }
}

// ------------------------------------------------------------ static UI
//
// Per-pixel evidence that a pixel is screen-fixed UI (a HUD, a subtitle, a
// channel logo) drawn over moving content, kept as a small count in an
// R8_UNORM texture per flow-ring entry and carried from pair to pair.
//
// Evidence is not "the pixel did not change" - a flat sky or a static
// background did not change either, and showing those unwarped would hide
// whatever moves across them. It is "the warp would change the pixel by far
// more than the pixel itself changed" (see UiMaskCS for the exact test,
// which also accepts translucent UI): the motion field around the pixel
// belongs to content moving underneath, and interpolating would drag the
// UI along with it. Where interpolation would leave the pixel alone anyway
// there is nothing to learn and nothing to protect.
//
// A pixel is protected only after kUiConfirm such pairs in an unbroken
// chain, so a moving texture that happens to fit the test at one pixel for
// one pair cannot be mistaken for UI. Any other change knocks it back below
// the threshold at once (UI that moves, animates or disappears is
// interpolated again from that pair on), and the count rebuilds only from
// fresh evidence.

static const float kUiMaxCount = 4.0f;
static const float kUiConfirm = 3.0f;
static const float kUiChangePenalty = 2.0f;
// In 8-bit levels, max over RGB. "Unchanged" must survive the noise of a
// lossless game capture, not of a compressed video: a static logo whose
// pixels shimmer by more than this is simply interpolated as before.
static const float kUiStaticTolerance = 4.0f;
// How far the warped result must stray from the unchanged pixel to count
// as damage. Well above the static tolerance, so noise can never supply
// evidence on its own.
static const float kUiDamageTolerance = 16.0f;

float DecodeUiCount(float stored)
{
    return round(saturate(stored) * kUiMaxCount);
}

float EncodeUiCount(float count)
{
    return clamp(count, 0.0f, kUiMaxCount) / kUiMaxCount;
}

// 1 where the pixel is confirmed UI, 0 below the threshold. Fed a bilinear
// sample of the stored value, it ramps across the one-texel border between
// the two, which antialiases the mask edge instead of stair-stepping it.
float UiCoverage(float stored)
{
    return saturate(saturate(stored) * kUiMaxCount - (kUiConfirm - 1.0f));
}

float MaxComponent(float3 v)
{
    return max(v.x, max(v.y, v.z));
}
