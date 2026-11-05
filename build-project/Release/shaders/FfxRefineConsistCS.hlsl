#include "FfxRefine.hlsli"

// The forward ring entry of the two-field refinement (FfxRefine.hlsli), from
// the refined forward field and the finished backward entry.
//
// The two fields are refined separately, and where the pictures leave a
// field free - flat regions, where any vector matches - each wanders on its
// own (on a uniform 1 px pan both spread by +-1.6 px). One field moved both
// samples together there; two independent ones pull them apart, which cost
// 1 dB on that pan. So where the forward field agrees with the backward one
// turned around, F(p) = -B(p + F(p)), it is replaced by exactly that: the
// pair is then consistent, as a single field is. Where they disagree by more
// than a couple of pixels - a point of frame0 that is hidden in frame1, which
// the backward field cannot describe - the refined forward field stands, and
// InterpolateCS's occlusion weighting tells the two samples apart.
//
// Constants: Width/Height = full resolution, LevelWidth/Height = the
// half-resolution field. The backward entry is on the same 2x2 grid, in
// full-resolution pixels.
Texture2D<float2> Forward : register(t0);    // refined forward field, half-resolution pixels
Texture2D<float2> Backward : register(t1);   // backward ring entry (2x2 grid, full-resolution pixels)
RWTexture2D<float2> RingFlow : register(u0);

// Disagreement (half-resolution pixels) up to which the turned-around
// backward field is taken, and from which the refined forward one is.
static const float kAgree = 0.5f;
static const float kDisagree = 1.5f;

float2 BackwardAt(float2 q)
{
    uint w, h;
    Backward.GetDimensions(w, h);
    return 0.5f * Backward.SampleLevel(LinearClamp, (q + 0.5f) / float2(w, h), 0);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, 2);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    const float2 p = float2(min(int2(id.xy), levelMax));
    const float2 f = Forward.Load(int3(int2(p), 0));
    float2 turned = -BackwardAt(p + f);
    turned = -BackwardAt(p + turned);
    const float w = saturate((length(f - turned) - kAgree) / (kDisagree - kAgree));
    RingFlow[id.xy] = 2.0f * lerp(turned, f, w);
}
