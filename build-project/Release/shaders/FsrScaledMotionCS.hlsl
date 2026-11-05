#include "Synthesis.hlsli"

// AMD FSR upscaling's motion vectors for one presented frame (Presenter,
// --upscale fsr4): the pair's backward field (each pixel of frame1 -> where
// it was in frame0, full-resolution pixels of the captured frames) scaled by
// TimeT, the share of the pair between the previous presented frame and
// this one. FSR's convention: this frame -> the previous one, in pixels of
// the frames it upscales (motionVectorScale 1). LevelWidth x LevelHeight is
// that size, the same as the flow's here. An unbound field, or TimeT = 0,
// reads as no motion.
Texture2D<float4> BackwardFlow : register(t0);
RWTexture2D<float2> MotionVectors : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const float2 toFull = float2(Width, Height) / float2(LevelWidth, LevelHeight);
    const float2 full = (float2(id.xy) + 0.5f) * toFull - 0.5f;
    MotionVectors[id.xy] = TimeT * SampleFlow(BackwardFlow, full) / toFull;
}
