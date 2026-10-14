#include "Synthesis.hlsli"

// Background beside static UI, after InterpolateCS (recordInterpolate).
//
// FFX locks the 8x8 blocks around a high-contrast static icon onto the icon
// (zero motion), and the refinement does the same per pixel, so through a
// camera turn the background right beside each icon was interpolated with
// zero motion: two frames' different backgrounds blended there, a dark and
// frozen rectangle next to the icon. InterpolateCS flags those pixels -
// within 16 px of confirmed UI, not held, and its pair of samples disagrees
// by more than 2 x kAgreement - by writing that mismatch / 255 into the
// output's alpha. Here each flagged pixel tries the motion of the scene
// around it (12 and 24 px away, eight directions); the candidate whose two
// samples show the same surface, off anything with UI evidence, replaces the
// pixel when it agrees within kAgreement and at least twice as well. Every
// flagged pixel gets its alpha back to 1. On the user's HUD over their own
// game background panning 24 and 48 px/frame (FFX, refined): HUD +0.2 dB and
// visibly cleaner on top of the mask fix; nothing else is touched. A
// separate pass because the same loop inside InterpolateCS slowed every
// pixel (0.20 -> 0.34 ms at 1440p) even where it never ran.

Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> UiMask : register(t4);
RWTexture2D<float4> Output : register(u0);

static const float kAgreement = 8.0f;   // InterpolateCS's

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float4 current = Output[id.xy];
    if (current.a >= 1.0f) return;
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
    float3 color = current.rgb;
    if (bestMismatch < kAgreement && bestMismatch < 0.5f * pairMismatch)
        color = lerp(SampleSharp(Frame0, pixel + best0), SampleSharp(Frame1, pixel + best1), t);
    Output[id.xy] = float4(color, 1.0f);
}
