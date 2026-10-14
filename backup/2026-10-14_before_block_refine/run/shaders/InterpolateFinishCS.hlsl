#include "Synthesis.hlsli"

// The last pass of every interpolated frame (recordInterpolate): InterpolateCS
// writes into a scratch texture, flagging pixels in its alpha, and this pass
// writes the output from it - so a pixel can look at its neighbours as they
// were synthesised, with nothing rewriting them meanwhile.
//
// Alpha 1: copied as it is.
//
// Alpha kBandFlag: an occlusion band. Beside a character the camera turns
// around, every pixel chose between two surfaces - which frame's sample to
// trust, which block's vector won the projection, which trajectory closed -
// from its own evidence, and that evidence is weakest exactly there: the
// field straddles the outline, and the background just uncovered has no
// match in one frame at all. On grass and hair neighbouring pixels chose
// differently, and the outline came out as pixel noise: single pixels and
// 2x2 clumps (the 2 px grid of the fields and the projection) of the other
// surface. A flagged pixel whose colour lies outside the range of the ring of
// pixels two away from it (the 5x5 border), by more than kSpeckle in some
// channel, belongs to nothing around it: it is clamped into that range. The
// ring skips the 3x3 core so that a 2x2 clump is no support for itself. Real
// detail survives: an edge, or a 1 px strand or blade, runs on through the
// ring and keeps the centre inside its range.
//
// Alpha 253/255 and below: background beside static UI whose own pair of
// samples disagrees - the pair's mismatch * 255. FFX locks the 8x8 blocks
// around a high-contrast static icon onto the icon (zero motion), and the
// refinement does the same per pixel, so through a camera turn the
// background right beside each icon was interpolated with zero motion: two
// frames' different backgrounds blended there, a dark and frozen rectangle
// next to the icon. Each such pixel tries the motion of the scene around it
// (12 and 24 px away, eight directions); the candidate whose two samples
// show the same surface, off anything with UI evidence, replaces the pixel
// when it agrees within kAgreement and at least twice as well. On the
// user's HUD over their own game background panning 24 and 48 px/frame
// (FFX, refined): HUD +0.2 dB and visibly cleaner on top of the mask fix. A
// separate pass because the same loop inside InterpolateCS slowed every
// pixel (0.20 -> 0.34 ms at 1440p) even where it never ran.

Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> UiMask : register(t4);
Texture2D<float4> Synthesised : register(t5);   // InterpolateCS's result and flags
RWTexture2D<float4> Output : register(u0);

static const float kAgreement = 8.0f;           // InterpolateCS's
static const float kBandFlag = 254.0f / 255.0f; // InterpolateCS's
static const float kSpeckle = 24.0f;            // levels outside the ring's range

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float4 current = Synthesised.Load(int3(id.xy, 0));
    if (current.a >= 1.0f)
    {
        Output[id.xy] = current;
        return;
    }
    float3 color = current.rgb;
    if (current.a > 253.5f / 255.0f)
    {
        const int2 limit = int2(Width, Height) - 1;
        float3 low = float3(1.0f, 1.0f, 1.0f), high = float3(0.0f, 0.0f, 0.0f);
        [unroll]
        for (int i = 0; i < 16; ++i)
        {
            // The 5x5 border, clockwise from the top-left corner.
            const int side = i / 4, along = i % 4;
            const int2 offset = side == 0 ? int2(along - 2, -2) : side == 1 ? int2(2, along - 2)
                              : side == 2 ? int2(2 - along, 2) : int2(-2, 2 - along);
            const float3 c = Synthesised.Load(int3(clamp(int2(id.xy) + offset, int2(0, 0), limit), 0)).rgb;
            low = min(low, c);
            high = max(high, c);
        }
        const float3 outside = max(low - color, color - high);
        if (MaxComponent(outside) * kLumaScale > kSpeckle) color = clamp(color, low, high);
        Output[id.xy] = float4(color, 1.0f);
        return;
    }

    const float pairMismatch = current.a * 255.0f;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    float bestMismatch = pairMismatch;
    float2 best0 = float2(0.0f, 0.0f), best1 = float2(0.0f, 0.0f);
    [loop]
    for (int r = 0; r < 2; ++r)
    {
        [loop]
        for (int k = 0; k < 8; ++k)
        {
            float2 d0, d1;
            WarpOffsets(ForwardFlow, BackwardFlow, pixel + kSurroundRadii[r] * kSurroundDirections[k], t, d0, d1);
            const float2 uv0 = (pixel + d0 + 0.5f) * inverseSize;
            const float2 uv1 = (pixel + d1 + 0.5f) * inverseSize;
            if (UiMask.SampleLevel(LinearClamp, uv0, 0) + UiMask.SampleLevel(LinearClamp, uv1, 0) > 0.0f)
                continue;   // off anything with UI evidence at all
            const float3 s0 = Frame0.SampleLevel(LinearClamp, uv0, 0).rgb;
            const float3 s1 = Frame1.SampleLevel(LinearClamp, uv1, 0).rgb;
            const float mismatch = MaxComponent(abs(s0 - s1)) * kLumaScale;
            if (mismatch < bestMismatch)
            {
                bestMismatch = mismatch;
                best0 = d0;
                best1 = d1;
            }
        }
    }
    if (bestMismatch < kAgreement && bestMismatch < 0.5f * pairMismatch)
        color = lerp(SampleSharp(Frame0, pixel + best0), SampleSharp(Frame1, pixel + best1), t);
    Output[id.xy] = float4(color, 1.0f);
}
