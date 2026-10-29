#include "Common.hlsli"

// AI depth edge dilation of one field (Presenter recordInterpolate), as AMD
// FSR's Reconstruct & Dilate does with the game's depth: at an outline each
// block takes the vector of the nearest-to-camera block around it, so the
// outline moves with the object in front instead of with a blend or the
// background behind it. The network's depth (256 x 256) is soft, so this
// only acts where the nearest neighbour is clearly nearer than the block
// itself (kEdge; a ground plane's gradient between neighbours is far below
// it). SearchRadius = reach in blocks.
//
// Input/output on the flow grid (BlockSize px per texel, xy = vector).
Texture2D<float4> Field : register(t0);
Texture2D<float> Depth : register(t1);      // nearness of the field's frame, 0 far .. 1 near
RWTexture2D<float2> Dilated : register(u0);

static const float kEdge = 0.02f;

float NearnessAt(int2 texel)
{
    const float2 centre = float2(texel) * (float)BlockSize + 0.5f * (float)(BlockSize - 1);
    return Depth.SampleLevel(LinearClamp, (centre + 0.5f) / float2(Width, Height), 0);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 gridMax = int2(grid) - 1;
    const float2 own = Field.Load(int3(id.xy, 0)).xy;
    const float ownNear = NearnessAt(int2(id.xy));
    float bestNear = ownNear;
    int2 best = int2(id.xy);
    const int reach = max(SearchRadius, 1);
    for (int dy = -reach; dy <= reach; ++dy)
        for (int dx = -reach; dx <= reach; ++dx)
        {
            const int2 n = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), gridMax);
            const float near = NearnessAt(n);
            if (near > bestNear)
            {
                bestNear = near;
                best = n;
            }
        }
    Dilated[id.xy] = bestNear - ownNear > kEdge ? Field.Load(int3(best, 0)).xy : own;
}
