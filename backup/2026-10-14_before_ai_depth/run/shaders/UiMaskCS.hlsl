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
RWTexture2D<float> Ui : register(u0);

// The pixel's own change must be below this fraction of the damage. A half
// let ~120 scattered pixels of a plain textured pan through (a pixel that
// barely changes where the warp happens to be poor, three pairs running);
// a third lets none through and keeps the translucent CS2 HUD.
static const float kUiChangeRatio = 0.33f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const int3 texel = int3(id.xy, 0);
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

    if (damage > kUiDamageTolerance && change < kUiChangeRatio * damage)
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
        // it. It joins when a neighbour is already confirmed UI and the
        // scene under it is moving: where nothing moves there is nothing to
        // protect, so a still scene can never be crept across, and a moving
        // flat region it absorbs looks the same either way and leaves the
        // mask the moment it changes.
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
            if (neighbour >= kUiConfirm) count = max(count + 1.0f, kUiConfirm);
        }
    }
    Ui[id.xy] = EncodeUiCount(count);
}
