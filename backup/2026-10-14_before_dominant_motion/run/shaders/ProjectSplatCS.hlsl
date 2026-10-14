#include "Synthesis.hlsli"

// Motion projection, step 2 of 2: every block of both fields writes itself
// at the place it occupies at time t, and where several land on one place
// the one most likely to be visible there wins (InterpolateCS reads the
// winner). This is how frame generation decides occlusion without a depth
// buffer: AMD FSR 3 frame interpolation splats its motion vectors to the
// interpolated frame with atomics and a priority, and Softmax Splatting
// (Niklaus & Liu, CVPR 2020) uses the brightness-constancy error as the
// priority when there is no depth. Here the priority is
//
//   kRoundTripWeight * |v + other(x + v)|  +  |I(x) - I_other(x + v)|
//
// (lower wins): the round trip is large for a point hidden in the other
// frame - background that a character is covering loses to the character
// even where both are flat - and the brightness error separates the rest.
//
// Projection (R32_UINT, 2 * gridW x gridH): the left half holds the forward
// field's winners (frame0's blocks at frame0 + t * F), the right half the
// backward field's (frame1's blocks at frame1 + (1 - t) * B). Each texel is
// min over (priority << 21 | source block index); 0xFFFFFFFF = nothing
// landed (ProjectClearCS).
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
RWTexture2D<uint> Projection : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    if (id.x >= 2 * grid.x || id.y >= grid.y) return;
    const bool backward = id.x >= grid.x;
    const uint2 texel = uint2(backward ? id.x - grid.x : id.x, id.y);
    const float2 centre = float2(texel) * (float)BlockSize + 0.5f * (float)(BlockSize - 1);
    const float t = saturate(TimeT);
    const float2 v = backward ? BackwardFlow.Load(int3(texel, 0)).xy : ForwardFlow.Load(int3(texel, 0)).xy;
    const float2 back = backward ? SampleFlow(ForwardFlow, centre + v) : SampleFlow(BackwardFlow, centre + v);
    const float2 inverseSize = rcp(float2(Width, Height));
    const int2 limit = int2(Width, Height) - 1;
    const int2 here = clamp(int2(round(centre)), int2(0, 0), limit);
    const float3 own = backward ? Frame1.Load(int3(here, 0)).rgb : Frame0.Load(int3(here, 0)).rgb;
    const float2 uv = (centre + v + 0.5f) * inverseSize;
    const float3 other = backward ? Frame0.SampleLevel(LinearClamp, uv, 0).rgb
                                  : Frame1.SampleLevel(LinearClamp, uv, 0).rgb;
    const float priority = ProjectionPriority(length(v + back), MaxComponent(abs(own - other)) * kLumaScale, length(v));

    const float2 target = centre + (backward ? 1.0f - t : t) * v;
    const int2 cell = int2(floor((target - 0.5f * (float)(BlockSize - 1)) / (float)BlockSize + 0.5f));
    if (any(cell < 0) || any(cell >= int2(grid))) return;
    const uint key = (min((uint)(priority * kPriorityQuantum), kPriorityMax) << kProjectionIndexBits) |
                     (texel.y * grid.x + texel.x);
    uint previous;
    InterlockedMin(Projection[uint2(cell.x + (backward ? (int)grid.x : 0), cell.y)], key, previous);
}
