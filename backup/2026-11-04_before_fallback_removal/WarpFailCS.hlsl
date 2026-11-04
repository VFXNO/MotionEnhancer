#include "Synthesis.hlsli"

// Where the motion cannot explain the pair at all (FlowEngine::recordWarpFail,
// once per pair on the flow queue, after the fields are final). One value per
// kFailCell x kFailCell pixel cell: 1 where warping is no better than not
// moving at all, 0 elsewhere. InterpolateCS shows the nearer real frame
// there instead of the warp.
//
// What it catches: a pose that jumps between two frames (a dodge, an attack
// starting, a character's animation skipping), a scene cut the capture's cut
// test missed, grass whipping in the wind. No vector matches the content
// there, so each output pixel took whichever wrong sample fitted best, and
// the frame showed pixel fragments of the old pose - its spear, its outline,
// a doubled prompt - for the whole pair.
//
// A cell fails when, over 8 x 8 samples (every second pixel) in both
// directions,
//   residual = mean |frame_a(p) - frame_b(p + field(p))|   (luma levels)
// is above kFailLevel and above kFailRatio x the cell's change with no
// motion at all. Correct motion leaves a small residual; a static region a
// small change. Both directions: beside a moving object the background
// one frame uncovers fails in that frame's field only (the other field
// finds the object there), and holding the real frame on that band put a
// second outline beside the object (-3 dB on a rendered pair). A cell is
// also kept only with at least kFailSupport of its 3x3 neighbourhood
// (itself included) failing, so only areas remain.
//
// Tiles of 8 x 8 cells, each writing its inner 6 x 6 (the outer ring is the
// neighbourhood). Constants: the pair's interpolation constants
// (setInterpolationConstants), LevelWidth/Height = the cell grid.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
RWTexture2D<float> Fail : register(u0);

static const int kFailCell = 16;          // FlowEngine::kWarpFailCell
static const float kFailLevel = 24.0f;
static const float kFailRatio = 0.6f;
static const float kFailSupport = 4.0f;
static const int kTile = 8;
static const int kInner = kTile - 2;

groupshared float g_raw[kTile][kTile];

float LumaAt(Texture2D<float4> frame, float2 position)
{
    const float2 uv = (position + 0.5f) / float2(Width, Height);
    return Luminance(frame.SampleLevel(LinearClamp, uv, 0).rgb) * kLumaScale;
}

float2 FieldAt(Texture2D<float4> flow, int2 pixel)
{
    const int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
    return flow.Load(int3(clamp(pixel / BlockSize, int2(0, 0), gridMax), 0)).xy;
}

float CellFails(int2 cell)
{
    if (any(cell < 0) || cell.x >= (int)LevelWidth || cell.y >= (int)LevelHeight) return 0.0f;
    const bool single = FlowScale > 0.5f;   // one field: backward only
    float still = 0.0f, forward = 0.0f, backward = 0.0f;
    [loop]
    for (int y = 0; y < kFailCell; y += 2)
    {
        [loop]
        for (int x = 0; x < kFailCell; x += 2)
        {
            const int2 p = min(cell * kFailCell + int2(x, y), int2(Width, Height) - 1);
            const float l0 = Luminance(Frame0.Load(int3(p, 0)).rgb) * kLumaScale;
            const float l1 = Luminance(Frame1.Load(int3(p, 0)).rgb) * kLumaScale;
            still += abs(l0 - l1);
            if (!single) forward += abs(l0 - LumaAt(Frame1, float2(p) + FieldAt(ForwardFlow, p)));
            backward += abs(l1 - LumaAt(Frame0, float2(p) + FieldAt(BackwardFlow, p)));
        }
    }
    const float n = (kFailCell / 2) * (kFailCell / 2);
    still /= n;
    forward /= n;
    backward /= n;
    const bool f = !single && forward > kFailLevel && forward > kFailRatio * still;
    const bool b = backward > kFailLevel && backward > kFailRatio * still;
    return (single ? b : f && b) ? 1.0f : 0.0f;
}

[numthreads(kTile, kTile, 1)]
void CSMain(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    const int2 cell = int2(group.xy) * kInner - 1 + int2(local.xy);
    g_raw[local.y][local.x] = CellFails(cell);
    GroupMemoryBarrierWithGroupSync();
    if (local.x == 0 || local.y == 0 || local.x == kTile - 1 || local.y == kTile - 1) return;
    if (cell.x >= (int)LevelWidth || cell.y >= (int)LevelHeight) return;
    float support = 0.0f;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
            support += g_raw[local.y + dy][local.x + dx];
    Fail[cell] = g_raw[local.y][local.x] > 0.0f && support >= kFailSupport ? 1.0f : 0.0f;
}
