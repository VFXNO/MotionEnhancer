#include "Synthesis.hlsli"

// Motion projection, step 1 of 2: empty the projection (ProjectSplatCS) for
// this output frame - 0xFFFFFFFF means nothing landed there.
RWTexture2D<uint> Projection : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    if (id.x >= 2 * grid.x || id.y >= grid.y) return;
    Projection[id.xy] = kProjectionEmpty;
}
