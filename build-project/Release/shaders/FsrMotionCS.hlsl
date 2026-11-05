#include "Synthesis.hlsli"

// AMD FSR frame generation's motion-vector input (FsrFrameGen, offline
// prototype) from this program's own optical flow: there is no game engine
// to supply motion vectors, so the backward field of the pair stands in.
//
// FSR's convention is the same as the backward field's: for each pixel of
// the current frame, where it was in the previous frame (previous position =
// position + vector). The backward field is on the flow ring's BlockSize
// grid in full-resolution pixels; FSR reads it at its render size
// (LevelWidth x LevelHeight), in that size's pixels (motionVectorScale 1).
// An unbound field reads zero: no motion, for the first frame of a sequence.
Texture2D<float4> BackwardFlow : register(t0);
RWTexture2D<float2> MotionVectors : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const float2 toFull = float2(Width, Height) / float2(LevelWidth, LevelHeight);
    const float2 full = (float2(id.xy) + 0.5f) * toFull - 0.5f;
    MotionVectors[id.xy] = SampleFlow(BackwardFlow, full) / toFull;
}
