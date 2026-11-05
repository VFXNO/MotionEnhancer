#include "Common.hlsli"

// One direction of the AI pyramid flow network's output, sampled from the
// network resolution (LevelWidth x LevelHeight) onto a grid of BlockSize
// pixel blocks of the capture (FlowEngine::recordAiFlow): the DIS dense field
// of the level DIS then refines it from (BlockSize 2^l, FlowScale 2^-l: that
// level's pixels), or the flow ring itself (BlockSize 2, FlowScale 1:
// full-resolution pixels).
//
// Flow is the network output copied into a LevelWidth x 4*LevelHeight
// texture: planes F01 x, F01 y (forward, frame0's grid), F10 x, F10 y
// (backward, frame1's grid). LevelIndex selects the direction: 0 forward,
// 1 backward.
Texture2D<float> Flow : register(t0);
RWTexture2D<float2> Out : register(u0);

float Plane(uint plane, int2 p)
{
    const int2 limit = int2(LevelWidth, LevelHeight) - 1;
    const int2 c = clamp(p, int2(0, 0), limit);
    return Flow.Load(int3(c.x, c.y + (int)(plane * LevelHeight), 0));
}

float2 Vector(uint plane, int2 p)
{
    return float2(Plane(plane, p), Plane(plane + 1u, p));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    // The block's centre in network pixels, bilinear between the four
    // network vectors around it.
    const float2 ratio = float2(LevelWidth, LevelHeight) / float2(Width, Height);
    const float2 centre = float2(id.xy) * (float)BlockSize + 0.5f * (float)BlockSize;
    const float2 q = centre * ratio - 0.5f;
    const int2 base = int2(floor(q));
    const float2 f = q - float2(base);
    const uint plane = LevelIndex == 0 ? 0u : 2u;
    const float2 v = lerp(lerp(Vector(plane, base), Vector(plane, base + int2(1, 0)), f.x),
                          lerp(Vector(plane, base + int2(0, 1)), Vector(plane, base + int2(1, 1)), f.x), f.y);
    // Network pixels to capture pixels, then to the target's units.
    Out[id.xy] = v / ratio * FlowScale;
}
