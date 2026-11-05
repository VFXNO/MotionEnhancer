#include "Synthesis.hlsli"

// Static-UI evidence for one frame pair (see the "static UI" notes in
// Synthesis.hlsli). Runs on the flow queue right after the pair's flow, at
// full resolution, and writes the updated count into the pair's ring entry;
// the next pair reads it back as PreviousUi while the frame chain is
// unbroken (unbound otherwise, which reads 0 and starts from nothing).
//
// Constants are the interpolation constants for this pair at t = 0.5, the
// midpoint, where a warp moves a pixel furthest from both frames.
//
// Two numbers per pixel:
//   damage  - how far the interpolated pixel would land from the pixel
//             left where it is (the exact warp InterpolateCS will run);
//   change  - how much the pixel itself changed between the two frames.
// UI drawn over moving content is the case where the warp would change a
// pixel by far more than the pixel actually changed. Opaque UI does not
// change at all. Translucent UI - CS2's panels, a crosshair outline,
// antialiased glyph edges, a 90% radar ring - changes by the see-through
// fraction of the background's change, which a plain "did not change" test
// can never accept, while the warp drags the UI's whole contrast across it.
// So a pair is evidence when the damage is clear and the pixel's own change
// is less than a third of it. A moving background fails that: its change is
// its texture moving, and the warp follows the same texture, so the damage
// is no larger than the change. The interior of a uniform translucent panel
// fails it too - the warp samples more of the same panel - and it keeps
// being warped, which is correct: the background behind it does move.
//
// Any other change takes the pixel out of the mask at once. Only UI that is
// static *right now* is held in place; HUD that animates, slides or changes
// is interpolated like everything else, because holding a changing pixel
// shows its old and new content blended on top of each other - a ghost.

Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> PreviousUi : register(t4);
Texture2D<float> TemporalUiAllow : register(t5);
Texture2D<float> HudPersistence : register(t6);   // UiPersistCS, one value per 4x4 cell
RWTexture2D<float> Ui : register(u0);

// The pixel's own change must be below this fraction of the damage. A half
// let ~120 scattered pixels of a plain textured pan through (a pixel that
// barely changes where the warp happens to be poor, three pairs running);
// a third lets none through and keeps the translucent CS2 HUD.
static const float kUiChangeRatio = 0.33f;

// HUD persistence: a cell is screen-fixed structure from this value on.
static const float kHudPersistent = 0.5f;
static const uint kHudCell = 4u;

bool HudEnabled()
{
    return NeuralWeights[59].x > 0.5f;
}

// The largest persistence of the pixel's cell and its eight neighbours: the
// flat inside of a HUD element (a bar's fill, a panel) has no edges of its
// own and borrows its border's, up to a cell or two away.
float HudNear(uint2 pixel)
{
    uint w, h;
    HudPersistence.GetDimensions(w, h);
    const int2 cell = int2(pixel / kHudCell);
    const int2 limit = int2(w, h) - 1;
    float p = 0.0f;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
            p = max(p, HudPersistence.Load(int3(clamp(cell + int2(dx, dy), int2(0, 0), limit), 0)));
    return p;
}

// A pixel with texture of its own: its colour differs from one of its eight
// neighbours by more than kHudTexture (8-bit levels, max over RGB) - an
// edge or a stroke, not the flat fill of a panel, sky or ground.
static const float kHudTexture = 12.0f;

bool Textured(uint2 pixel, float3 centre)
{
    const int2 limit = int2(Width, Height) - 1;
    float most = 0.0f;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
            most = max(most, MaxComponent(abs(Frame1.Load(int3(clamp(int2(pixel) + int2(dx, dy), int2(0, 0), limit), 0)).rgb - centre)));
    return most * kLumaScale > kHudTexture;
}

// Where protection may start: the union of what the detectors allow - HUD
// persistence, the trained temporal UI, the icon boxes - and, with none of
// them on, everywhere (the native tests alone). With HUD persistence on,
// nothing outside screen-fixed structure is protected, not even by the
// native tests: a protected pixel is shown where it is, and InterpolateCS
// drops every warped sample that lands on one, so a wrongly protected patch
// of sky or ground freezes and also stalls the moving scene around it - the
// motion stutters there. (The boxes used to be the only place allowed when
// they were on, and the few boxes the icon detector found in a game left
// the mask emptier than with no detector.)
//
// This pass alone reuses the otherwise unused neural constants for a detector
// allow-list: x = enabled, z = box count, weights[0..31] = normalised xyxy.
// These are NOT the AI flow model's weights or the synthesis blending mask.
bool InDetectedUiRegion(uint2 pixel)
{
    if (NeuralControl.x < 0.5f && NeuralControl.y < 0.5f && !HudEnabled()) return true;
    if (HudEnabled() && HudNear(pixel) >= kHudPersistent) return true;
    const float2 uv = (float2(pixel) + 0.5f) / float2(Width, Height);
    if (NeuralControl.y >= 0.5f && NeuralControl.w >= 0.5f)
    {
        const float candidate = TemporalUiAllow.SampleLevel(LinearClamp, uv, 0);
        if (isfinite(candidate) && candidate >= 0.5f && candidate <= 1.0f) return true;
    }
    if (NeuralControl.x < 0.5f) return false;
    const int count = clamp((int)NeuralControl.z, 0, 32);
    for (int i = 0; i < count; ++i)
    {
        const float4 box = NeuralWeights[i];
        if (all(uv >= box.xy) && all(uv < box.zw)) return true;
    }
    return false;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const int3 texel = int3(id.xy, 0);
    if (!InDetectedUiRegion(id.xy))
    {
        // Clear old evidence too: a lost/shrunk/expired detection cannot leave
        // a protected pixel behind. Never fill a detector rectangle as UI.
        Ui[id.xy] = 0.0f;
        return;
    }
    const float2 pixel = float2(id.xy);
    const float3 p0 = Frame0.Load(texel).rgb;
    const float3 p1 = Frame1.Load(texel).rgb;
    float count = DecodeUiCount(PreviousUi.Load(texel));

    const float t = saturate(TimeT);
    const float2 inverseSize = rcp(float2(Width, Height));
    float2 from0, from1;
    WarpSources(ForwardFlow, BackwardFlow, pixel, t, from0, from1);
    const float3 c0 = SampleSharp(Frame0, from0);
    const float3 c1 = SampleSharp(Frame1, from1);
    const float damage = MaxComponent(abs(lerp(c0, c1, t) - lerp(p0, p1, t))) * kLumaScale;
    const float change = MaxComponent(abs(p1 - p0)) * kLumaScale;

    if (HudEnabled() && change <= kUiStaticTolerance && HudNear(id.xy) >= kHudPersistent &&
        Textured(id.xy, p1))
    {
        // Screen-fixed structure that did not change in this pair: held at
        // once. Opaque HUD gives the warp test nothing to measure - the flow
        // usually finds it still, until a fast pan drags its edges - and an
        // unchanged pixel is the same in both frames, so holding it cannot
        // ghost. Only a textured pixel, though: an edge or a stroke of the
        // HUD itself. Holding every unchanged pixel near one also held the
        // plain sky, ground and panel fill around text (41% of all held
        // pixels were flat and outside the HUD), where a thin line moving
        // across could vanish from the generated frame. The inside of a
        // glyph or bar joins through the neighbour rule below, while the
        // scene under it moves. A HUD that changes leaves through the
        // branch below.
        count = max(count + 1.0f, kUiConfirm);
    }
    else if (damage > kUiDamageTolerance && change < kUiChangeRatio * damage)
    {
        count += 1.0f;
    }
    else if (change > kUiStaticTolerance)
    {
        count -= kUiChangePenalty;
    }
    else
    {
        // Unchanged, and the warp would leave it alone too: the middle of a
        // glyph or a bar, whose warped samples land on more of the same UI.
        // It never earns evidence by itself - and then, the moment the warp
        // reaches it, it is dragged as a ghost copy into the panel beside
        // it. It joins when a neighbour is confirmed UI by its own evidence
        // (the full count) and the scene under it is moving: where nothing
        // moves there is nothing to protect. A pixel that joined this way
        // stays at the confirm count and never recruits its own neighbours:
        // when it could, a flat sky or ground beside the HUD was crept
        // across one pixel per pair, as long as the camera moved, and the
        // mask spread over plain areas far from any HUD.
        const float2 v = FlowScale != 0.0f ? SampleFlow(BackwardFlow, pixel)
                                           : SampleFlow(ForwardFlow, pixel);
        if (dot(v, v) > 1.0f)
        {
            const int2 limit = int2(Width, Height) - 1;
            float neighbour = 0.0f;
            [unroll]
            for (int dy = -1; dy <= 1; ++dy)
                [unroll]
                for (int dx = -1; dx <= 1; ++dx)
                    neighbour = max(neighbour, DecodeUiCount(PreviousUi.Load(
                        int3(clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), limit), 0))));
            if (neighbour >= kUiMaxCount) count = max(count, kUiConfirm);
        }
    }
    Ui[id.xy] = EncodeUiCount(count);
}
