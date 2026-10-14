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
        // Unchanged, and its own vector would leave it alone. That is not
        // "nothing to protect" when the scene around it moves: FFX locks the
        // blocks over a high-contrast static icon onto the icon (zero
        // motion), and the refinement does the same per pixel, so a HUD over
        // a camera turn earned no evidence at all - 1-1.5% of a 5.5% HUD
        // confirmed - while the mixed blocks at its edges, carrying the
        // background's motion, dragged fragments of the icons across the
        // turn (the user's blocky top-right icons; worse with the refinement,
        // which confirmed even less). So the pixel is also judged against the
        // motion of the scene around it (12 and 24 px, eight directions):
        // the most different nearby motion would carry it off by
        // damageAround while it did not change - the same evidence test.
        // Static scenery beside a moving object fits too, as it already did
        // where its own vector was wrong; holding a pixel that is unchanged
        // in both frames shows what both frames show.
        // The search reads one flow cell per point and only the frame0 side;
        // the chosen point's full offsets are worked out once.
        const float2 own0 = from0 - pixel;
        const int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
        const float sign0 = FlowScale != 0.0f ? t : -t;
        float2 aroundAt = pixel;
        float spread = 0.0f;
        [unroll]
        for (int r = 0; r < 2; ++r)   // kSurroundRadii[0..1]
        {
            [unroll]
            for (int k = 0; k < 8; ++k)
            {
                const float2 q = pixel + kSurroundRadii[r] * kSurroundDirections[k];
                const int2 cell = clamp(int2(q / BlockSize), int2(0, 0), gridMax);
                const float2 v = FlowScale != 0.0f ? BackwardFlow.Load(int3(cell, 0)).xy : ForwardFlow.Load(int3(cell, 0)).xy;
                const float differs = length(sign0 * v - own0);
                if (differs > spread) { spread = differs; aroundAt = q; }
            }
        }
        if (spread > 1.0f)
        {
            float2 around0, around1;
            WarpOffsets(ForwardFlow, BackwardFlow, aroundAt, t, around0, around1);
            const float3 a0 = SampleSharp(Frame0, pixel + around0);
            const float3 a1 = SampleSharp(Frame1, pixel + around1);
            const float damageAround = MaxComponent(abs(lerp(a0, a1, t) - lerp(p0, p1, t))) * kLumaScale;
            if (damageAround > kUiDamageTolerance && change < kUiChangeRatio * damageAround)
            {
                count += 1.0f;
            }
            else
            {
                // The middle of a glyph or a bar, whose warped samples land
                // on more of the same UI, never earns evidence by itself -
                // and then, the moment the warp reaches it, it is dragged as
                // a ghost copy into the panel beside it. It joins when a
                // neighbour is already confirmed UI and the scene around it
                // is moving: where nothing moves there is nothing to protect,
                // so a still scene can never be crept across, and a moving
                // flat region it absorbs looks the same either way and
                // leaves the mask the moment it changes.
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
    }
    Ui[id.xy] = EncodeUiCount(count);
}
