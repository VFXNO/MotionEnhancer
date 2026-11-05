#include "Common.hlsli"

// Occlusion fill for one direction of a two-field pair, on the ring's block
// grid, once per pair (FlowEngine::recordOcclusionFill).
//
// Beside a moving object the estimated field is a blend: patches straddle
// the outline, so the vector fades from the object's motion to the
// background's over several pixels, and much wider next to a flat region,
// which matches any motion equally well. Those vectors are wrong for both
// surfaces, and the interpolation's per-pixel trajectory choice made from
// them left the band as blocky fragments of both sources. This pass gives
// each such block one consistent vector instead:
//
//   * A block whose vector survives the round trip - out along it, back
//     along the other field, within kConsistent px of where it started - is
//     confirmed and left alone.
//   * Otherwise the vectors of confirmed blocks nearby are tried. One that
//     also survives the round trip from here and matches the picture (mean
//     luma difference over the block below kMatch) means the block is
//     visible in both frames and only had a blended vector: the best match
//     is taken. That keeps an object's own edge on the object's motion.
//   * If none does, the block is hidden in the other frame. Among the
//     confirmed vectors under which it would be hidden, the one taken is
//     that of the block that looks most like it in this frame (mean colour
//     over a few pixels). Geometry alone cannot choose here: beside a
//     character the camera turns around, the background's vector carries
//     the block onto the character and the character's vector carries it
//     onto background that came from elsewhere - both fail the round trip,
//     and both fail the picture. Taking the nearest put the character's
//     motion on the background beside it and smeared the character's
//     colour 15-20 px into it. The block is background, and it looks like
//     the background next to it, not like the character.
//   * A block with no candidate keeps its vector.
//
// Measured (DIS, 960x540 third-person 4x test): the band 1-12 px beside the
// character +0.84 dB, the character +0.63 dB, step-to-step temporal error
// -16% and the worst step -19%; a target tracked during a 20 px/frame turn
// 21.2 -> 24.6 dB. The cost is the moving-objects pair (-2.3 dB): a smooth
// shape's trailing edge keeps a clean trajectory in the frame it left, so
// the interpolation still blends some of it into the filled band. Requiring
// a hidden block to land on a surface both fields agree on halved that loss
// and also half the gains. --no-occlusion-fill (or the GUI checkbox) turns
// the pass off.
//
// Input/output on the ring grid (BlockSize px per texel, xy = vector in
// full-resolution pixels); both directions read the unfilled pair.
Texture2D<float4> Field : register(t0);     // this direction, on this frame's grid
Texture2D<float4> Other : register(t1);     // the other direction, on the other frame's grid
Texture2D<float4> ThisFrame : register(t2);
Texture2D<float4> OtherFrame : register(t3);
RWTexture2D<float2> Filled : register(u0);

static const float kConsistent = 1.5f;      // round-trip error (px) that confirms a vector
static const float kMatch = 10.0f;          // mean luma difference (levels) for "the picture agrees"
static const int kDirections = 8;
static const int kDistances = 4;

float2 OtherAt(float2 position)
{
    uint w, h;
    Other.GetDimensions(w, h);
    // Texel i's centre is at full-resolution i * BlockSize + (BlockSize - 1) / 2.
    const float2 texel = (position - 0.5f * (float)(BlockSize - 1)) / (float)BlockSize;
    return Other.SampleLevel(LinearClamp, (texel + 0.5f) / float2(w, h), 0).xy;
}

float RoundTrip(float2 position, float2 v)
{
    return length(v + OtherAt(position + v));
}

float Luma(float3 c)
{
    return 255.0f * dot(c, float3(0.299f, 0.587f, 0.114f));
}

// Mean colour of this frame over the 4x4 pixels around a block centre.
float3 Appearance(float2 centre)
{
    const int2 limit = int2(Width, Height) - 1;
    const int2 base = int2(floor(centre)) - 1;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            sum += ThisFrame.Load(int3(clamp(base + int2(x, y), int2(0, 0), limit), 0)).rgb;
    return sum / 16.0f;
}

// Mean luma difference between this frame's block and the other frame
// displaced by v. A wider window (4x4 or 8x8 px around the block) was
// measured as well and is no better.
float PictureError(float2 centre, float2 v)
{
    const float2 inverseSize = rcp(float2(Width, Height));
    const int2 limit = int2(Width, Height) - 1;
    const int2 origin = int2(floor(centre)) - (BlockSize - 1) / 2;
    float sum = 0.0f;
    float count = 0.0f;
    for (int y = 0; y < BlockSize; ++y)
        for (int x = 0; x < BlockSize; ++x)
        {
            const int2 p = clamp(origin + int2(x, y), int2(0, 0), limit);
            const float here = Luma(ThisFrame.Load(int3(p, 0)).rgb);
            const float there = Luma(OtherFrame.SampleLevel(LinearClamp, (float2(p) + v + 0.5f) * inverseSize, 0).rgb);
            sum += abs(here - there);
            count += 1.0f;
        }
    return sum / count;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 gridMax = int2(grid) - 1;
    const float2 centre = float2(id.xy) * (float)BlockSize + 0.5f * (float)(BlockSize - 1);
    const float2 own = Field.Load(int3(id.xy, 0)).xy;
    if (RoundTrip(centre, own) <= kConsistent)
    {
        Filled[id.xy] = own;
        return;
    }

    const int2 directions[kDirections] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
                                          int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
    // In ring texels: 2, 4, 8 and 12 blocks (4-24 px at a 2 px grid).
    const int distances[kDistances] = {2, 4, 8, 12};
    float bestScore = 1e30f;
    float2 visible = own;
    bool haveVisible = false;
    float2 hidden = own;
    bool haveHidden = false;
    float hiddenScore = 1e30f;
    const float3 look = Appearance(centre);
    [loop]
    for (int d = 0; d < kDistances; ++d)
    {
        [loop]
        for (int k = 0; k < kDirections; ++k)
        {
            const int2 source = clamp(int2(id.xy) + directions[k] * distances[d], int2(0, 0), gridMax);
            const float2 v = Field.Load(int3(source, 0)).xy;
            const float2 sourceCentre = float2(source) * (float)BlockSize + 0.5f * (float)(BlockSize - 1);
            if (RoundTrip(sourceCentre, v) > kConsistent) continue;      // not confirmed where it came from
            if (RoundTrip(centre, v) <= kConsistent)
            {
                const float e = PictureError(centre, v);
                if (e < kMatch && e < bestScore)
                {
                    bestScore = e;
                    visible = v;
                    haveVisible = true;
                }
            }
            else
            {
                // How unlike this block its source looks (0..255 levels), and
                // a small preference for the nearer one on a tie.
                const float3 difference = 255.0f * (Appearance(sourceCentre) - look);
                const float score = length(difference) + 0.5f * (float)d;
                if (score < hiddenScore)
                {
                    hiddenScore = score;
                    hidden = v;
                    haveHidden = true;
                }
            }
        }
    }
    Filled[id.xy] = haveVisible ? visible : (haveHidden ? hidden : own);
}
