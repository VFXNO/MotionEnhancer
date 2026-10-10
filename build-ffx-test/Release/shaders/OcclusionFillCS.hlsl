#include "Synthesis.hlsli"

// Motion for what one frame shows and the other hides.
//
// Beside an object moving against the background, a band of background is
// visible in only one of the two frames - uncovered behind the object, or
// about to be covered in front of it. Its motion cannot be measured (there
// is nothing to match it against), and the estimator's answer there is a
// smear between the object's motion and the background's. The warp then
// fills the band with background taken from the wrong place.
//
// Such a block is found by its round trip through the opposite field: its
// vector v takes it to the other frame, and the other field there does not
// bring it back (|v + Other(centre + v)| above kConsistent). It is given the
// motion of a confirmed block within kReach that is closest to the camera's
// motion (DominantMotionCS): a hidden band is background by definition, and
// the background is what moves with the camera. Confirmed blocks pass
// through untouched, so everywhere but occlusions the field is unchanged.
//
// A failed round trip alone is not enough. Parts of a turning character
// fail it too - the opposite field is less reliable on a model that changes
// shape - and filling those with the camera's motion smeared the character
// (-2.2 dB on it). What separates them is the picture: a hidden block has
// nothing to match in the other frame, while the character's block matches
// well at its own vector. So a block is filled only when its own vector
// also matches badly (mean luma difference above kMatched).
//
// And the background motion it would get must itself be hidden here. A
// truly hidden block, moved with the background's motion, lands behind the
// occluder in the other frame, where the other field reports the
// occluder's motion: that round trip fails too. A block of the character
// moved with the background's motion lands on ordinary visible background,
// where the round trip succeeds - it is not hidden background, and filling
// it smeared the character's own outline (-3.2 dB on its exact silhouette).
//
// Constants: the interpolation constants of the pair (block size).
Texture2D<float4> Field : register(t0);
Texture2D<float4> Other : register(t1);
Texture2D<float2> Dominant : register(t2);
Texture2D<float4> ThisFrame : register(t3);   // the frame Field is on
Texture2D<float4> OtherFrame : register(t4);  // the frame it points into
RWTexture2D<float4> Filled : register(u0);

static const float kConsistent = 2.0f;   // round-trip pixels
static const float kReach = 24.0f;       // pixels to look for confirmed background
static const float kMatched = 8.0f;      // mean luma levels: the block's own vector still matches
static const float kObjectMotion = 4.0f; // pixels from the camera's motion: something else moving
static const int kMargin = 2;            // blocks left alone around such motion

// Mean luma difference of block b against the other frame at displacement v.
float MatchError(int2 b, float2 v)
{
    const float2 inverseSize = rcp(float2(Width, Height));
    float sum = 0.0f;
    for (int y = 0; y < BlockSize; ++y)
        for (int x = 0; x < BlockSize; ++x)
        {
            const float2 p = float2(b * BlockSize + int2(x, y));
            const float a = Luminance(ThisFrame.SampleLevel(LinearClamp, (p + 0.5f) * inverseSize, 0).rgb);
            const float c = Luminance(OtherFrame.SampleLevel(LinearClamp, (p + v + 0.5f) * inverseSize, 0).rgb);
            sum += abs(a - c);
        }
    return sum * kLumaScale / (float)(BlockSize * BlockSize);
}

float RoundTrip(int2 b, float2 v)
{
    const float2 centre = float2(b) * BlockSize + 0.5f * (BlockSize - 1);
    return length(v + SampleFlow(Other, centre + v));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const float4 own = Field.Load(int3(id.xy, 0));
    if (RoundTrip(int2(id.xy), own.xy) < kConsistent || MatchError(int2(id.xy), own.xy) < kMatched)
    {
        Filled[id.xy] = own;
        return;
    }
    const float2 camera = Dominant.Load(int3(LevelIndex, 0, 0));
    const int reach = (int)ceil(kReach / (float)BlockSize);
    const int2 limit = int2(grid) - 1;
    // A block within kMargin blocks of a confirmed block that moves unlike
    // the camera is the object's own edge, not the band behind it: left
    // alone. Filling right up to the object made the motion step at its
    // outline so sharp that the warp's blend of neighbouring blocks dragged
    // background into the outline (-3.2 dB on the character's exact
    // silhouette); one block of margin left -0.8 dB, two leave -0.15 dB
    // while keeping the band's gain.
    [unroll]
    for (int ny = -kMargin; ny <= kMargin; ++ny)
        [unroll]
        for (int nx = -kMargin; nx <= kMargin; ++nx)
        {
            const int2 b = clamp(int2(id.xy) + int2(nx, ny), int2(0, 0), limit);
            const float2 v = Field.Load(int3(b, 0)).xy;
            if (length(v - camera) > kObjectMotion && RoundTrip(b, v) < kConsistent)
            {
                Filled[id.xy] = own;
                return;
            }
        }
    float2 best = own.xy;
    float bestDistance = 1e30f;
    for (int dy = -reach; dy <= reach; ++dy)
    {
        for (int dx = -reach; dx <= reach; ++dx)
        {
            const int2 b = int2(id.xy) + int2(dx, dy);
            if (any(b < 0) || any(b > limit)) continue;
            const float2 v = Field.Load(int3(b, 0)).xy;
            const float distance = length(v - camera);
            if (distance >= bestDistance) continue;
            if (RoundTrip(b, v) >= kConsistent) continue;
            best = v;
            bestDistance = distance;
        }
    }
    if (RoundTrip(int2(id.xy), best) < kConsistent) best = own.xy;   // not hidden: keep it
    Filled[id.xy] = float4(best, own.zw);
}
