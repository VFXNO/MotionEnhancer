#include "Common.hlsli"

// AMD FSR frame generation's depth input (FsrFrameGen, offline prototype).
// There is no game depth buffer: every pixel gets the same depth, FlowScale
// (0 = near plane, 1 = far plane, not inverted), so FSR sees one flat
// surface and takes all motion from the motion vectors.
RWTexture2D<float> Depth : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    Depth[id.xy] = FlowScale;
}
