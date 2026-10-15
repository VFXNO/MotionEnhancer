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

// Colour of `frame` at `position` (pixels, texel centres on integers):
// Catmull-Rom, from five bilinear taps, clamped to the four texels around
// the position so it cannot ring or overshoot.
//
// A warp lands between texels almost everywhere, and a bilinear read there
// averages its neighbours - the generated frame came out softer than the
// real frames on either side of it, by a varying amount, which shows as the
// image pulsing between sharp and soft. Measured on rendered pairs, the fine
// detail (mean |Laplacian|) of the generated frame was 56-59% of the real
// frame's on game textures and about 90% on photographic pans; with this it
// is 75-80% and 94-102%, and PSNR rises 2.6-3.4 dB on sub-pixel motion. At
// a texel centre (no motion, or a whole-pixel one) it returns the texel
// exactly, as the bilinear read did.
float3 SampleSharp(Texture2D<float4> frame, float2 position)
{
    const float2 inverseSize = rcp(float2(Width, Height));
    const float2 texPos1 = floor(position) + 0.5f;     // the texel at or before the position
    const float2 f = position + 0.5f - texPos1;
    const float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    const float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    const float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    const float2 w3 = f * f * (-0.5f + 0.5f * f);
    // The middle two taps of each axis share one bilinear read placed
    // between them; the four corner taps (weights near zero) are skipped.
    const float2 w12 = w1 + w2;
    const float2 texPos0 = (texPos1 - 1.0f) * inverseSize;
    const float2 texPos3 = (texPos1 + 2.0f) * inverseSize;
    const float2 texPos12 = (texPos1 + w2 / w12) * inverseSize;
    float3 result = frame.SampleLevel(LinearClamp, float2(texPos12.x, texPos0.y), 0).rgb * (w12.x * w0.y)
                  + frame.SampleLevel(LinearClamp, float2(texPos0.x, texPos12.y), 0).rgb * (w0.x * w12.y)
                  + frame.SampleLevel(LinearClamp, texPos12, 0).rgb * (w12.x * w12.y)
                  + frame.SampleLevel(LinearClamp, float2(texPos3.x, texPos12.y), 0).rgb * (w3.x * w12.y)
                  + frame.SampleLevel(LinearClamp, float2(texPos12.x, texPos3.y), 0).rgb * (w12.x * w3.y);
    result /= w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    const int2 limit = int2(Width, Height) - 1;
    const int2 base = int2(floor(position));
    const float3 a = frame.Load(int3(clamp(base, int2(0, 0), limit), 0)).rgb;
    const float3 b = frame.Load(int3(clamp(base + int2(1, 0), int2(0, 0), limit), 0)).rgb;
    const float3 c = frame.Load(int3(clamp(base + int2(0, 1), int2(0, 0), limit), 0)).rgb;
    const float3 d = frame.Load(int3(clamp(base + int2(1, 1), int2(0, 0), limit), 0)).rgb;
    return clamp(result, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}

// ------------------------------------------------------ motion projection
//
// For two-field pairs both fields are projected to the output time before
// interpolating (ProjectClearCS, ProjectSplatCS): each block is written at
// the place it occupies at time t, and where several land on one place the
// one with the lowest priority (most likely visible there) wins. A key is
// (priority << kProjectionIndexBits) | source block index.

static const uint kProjectionEmpty = 0xFFFFFFFFu;
// Source block index: 2^21 blocks is a 4K frame at a 2 px grid.
static const uint kProjectionIndexBits = 21u;
// Largest stored priority step; one below the top so a key is never empty.
static const uint kPriorityMax = (1u << (32u - kProjectionIndexBits)) - 2u;
static const float kPriorityQuantum = 4.0f;    // steps per priority unit
// Priority per pixel of round-trip error, against one luma level of
// brightness error: a point hidden in the other frame fails the round trip
// by the relative motion, tens of pixels, and loses to whatever covers it
// even where both are flat and the brightness says nothing.
static const float kRoundTripWeight = 16.0f;
float ProjectionPriority(float roundTrip, float brightnessError)
{
    return kRoundTripWeight * roundTrip + brightnessError;
}

// The vector of the block projected to this pixel from one half of the
// projection (0 = forward field, scale t; 1 = backward, scale 1 - t). The
// winners of the four cells around the pixel are candidates, and the one
// whose block actually lands nearest the pixel is taken: the cell grid is
// 2 px, and taking the nearest cell's winner put every outline on that grid
// (a character's exact silhouette -1.3 dB). Where all four are empty the
// 3x3 around them is searched - blocks that move apart leave single empty
// cells - and found = false when that is empty too: that frame does not
// show this point at time t.
float2 ProjectedVector(Texture2D<uint> projection, Texture2D<float4> field, uint half, float s, float2 pixel,
                       out bool found)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    const int2 cellMax = int2(grid) - 1;
    const int offset = (int)(half * grid.x);
    const float2 position = (pixel - 0.5f * (float)(BlockSize - 1)) / (float)BlockSize;
    const int2 base = int2(floor(position));
    float best = 1e30f;
    float2 vector = float2(0.0f, 0.0f);
    found = false;
    [unroll]
    for (int pass = 0; pass < 2; ++pass)
    {
        if (found) break;
        const int lo = pass == 0 ? 0 : -1;
        const int hi = pass == 0 ? 1 : 2;
        for (int dy = lo; dy <= hi; ++dy)
            for (int dx = lo; dx <= hi; ++dx)
            {
                const int2 n = clamp(base + int2(dx, dy), int2(0, 0), cellMax);
                const uint key = projection.Load(int3(n.x + offset, n.y, 0));
                if (key == kProjectionEmpty) continue;
                const uint index = key & ((1u << kProjectionIndexBits) - 1u);
                const uint2 source = uint2(index % grid.x, index / grid.x);
                const float2 v = field.Load(int3(source, 0)).xy;
                const float2 landing = float2(source) * (float)BlockSize + 0.5f * (float)(BlockSize - 1) + s * v;
                const float d = dot(landing - pixel, landing - pixel);
                if (d < best)
                {
                    best = d;
                    vector = v;
                    found = true;
                }
            }
    }
    return vector;
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
