#include "Common.hlsli"

// One direction of the AI pyramid flow network's output into a flow-ring
// entry (FlowEngine::recordAiFlow): the network's field (LevelWidth x
// LevelHeight, in network pixels) sampled bilinearly at each ring block's
// centre and scaled to full-resolution pixels, on the ring's grid
// (BlockSize px, the 2 px DIS uses), so everything downstream - occlusion
// fill, projection, interpolation - reads it exactly as it reads DIS's.
//
// Flow is the output copied into a LevelWidth x 4*LevelHeight texture:
// planes F01 x, F01 y (forward, frame0's grid), F10 x, F10 y (backward,
// frame1's grid). LevelIndex selects the direction: 0 forward, 1 backward.
Texture2D<float> Flow : register(t0);
RWTexture2D<float2> Ring : register(u0);

float Plane(uint plane, float2 q)
{
    const int2 limit = int2(LevelWidth, LevelHeight) - 1;
    const float2 c = clamp(q, float2(0.0f, 0.0f), float2(limit));
    const int2 base = min(int2(floor(c)), limit);
    const int2 next = min(base + 1, limit);
    const float2 f = c - float2(base);
    const int row = (int)(plane * LevelHeight);
    const float v00 = Flow.Load(int3(base.x, base.y + row, 0));
    const float v10 = Flow.Load(int3(next.x, base.y + row, 0));
    const float v01 = Flow.Load(int3(base.x, next.y + row, 0));
    const float v11 = Flow.Load(int3(next.x, next.y + row, 0));
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    // Block centre in full-resolution pixel coordinates, then in the
    // network's (pixel centres at +0.5 in both).
    const float2 centre = float2(id.xy) * (float)BlockSize + 0.5f * (float)(BlockSize - 1);
    const float2 ratio = float2(LevelWidth, LevelHeight) / float2(Width, Height);
    const float2 q = (centre + 0.5f) * ratio - 0.5f;
    const uint plane = LevelIndex == 0 ? 0u : 2u;
    const float2 v = float2(Plane(plane, q), Plane(plane + 1u, q));
    Ring[id.xy] = v / ratio;
}
