#include "FfxRefine.hlsli"

// One refined half-resolution field (FfxRefine.hlsli) into a flow-ring
// entry: a 2x2-pixel grid at full resolution is exactly the half-resolution
// pixel grid, so each ring texel is its pixel's vector in full-resolution
// pixels. The field already has its direction's convention (forward on
// frame0's grid, backward on frame1's); FlowScale = -1 negates it, which the
// single-field mode uses to write the forward entry from the backward field.
Texture2D<float2> Field : register(t0);
RWTexture2D<float2> RingFlow : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, 2);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    RingFlow[id.xy] = FlowScale * 2.0f * Field.Load(int3(min(int2(id.xy), levelMax), 0));
}
