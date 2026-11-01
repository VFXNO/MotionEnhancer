#include "Synthesis.hlsli"

// HUD persistence: where on the screen structure stays put while the world
// behind it moves. One value per 4x4-pixel cell (LevelWidth x LevelHeight),
// carried from pair to pair and updated once per pair on the flow queue,
// before UiMaskCS reads it.
//
// A HUD is the part of the picture that does not move with the camera: its
// edges sit at the same screen position, with the same gradient, frame
// after frame, while the scene around them moves. A cell gains persistence
// when enough of its pixels are fixed structure - an edge that kept its
// gradient between the two frames (the gradient, not the colour: a
// translucent panel's glyph edges keep theirs while the background shows
// through and changes), or a pixel that kept its colour and stands out from
// its surroundings (a stroke too thin to have an inside) - and the flow says
// the scene around the cell is moving. Counting pixels, not testing the
// cell as a whole, finds a thin HUD stroke that shares its cell with
// moving scene. Where nothing moves there
// is nothing to learn - a still camera shows every pixel as static - so the
// value is left as it is. A cell whose content changes loses most of its
// persistence at once, so a HUD element that animates, slides or closes
// leaves the protected set within a pair.
//
// No model, no labels: it works for any game and any HUD layout. It only
// says where screen-fixed structure is; UiMaskCS still decides per pixel,
// and only ever holds pixels that did not change between the two frames.
//
// Constants: the interpolation constants of this pair (for SampleFlow),
// LevelWidth x LevelHeight = the cell grid, NeuralWeights[59].y = 1 to start
// from nothing (no previous persistence: first pair or after a resize).
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
Texture2D<float> PreviousPersistence : register(t4);
RWTexture2D<float> Persistence : register(u0);

static const int kCell = 4;
// An edge pixel: luma gradient magnitude above this (8-bit levels per
// pixel). Flat sky, fog and smooth gradients stay below it; text, icons,
// frames and bar outlines are far above.
static const float kEdgeLevel = 12.0f;
// A kept edge pixel: its gradient changed by at most this fraction of its
// energy (a third of its magnitude).
static const float kStructureChange = 0.1f;
// ...or a pixel whose own colour stayed (within this, 8-bit levels) while
// it stands out from its 3x3 surroundings by more than kContrastLevel: a
// stroke too thin to have an inside, whose gradient always takes in the
// moving background beside it (a 1 px crosshair, small text without an
// outline). Flat sky stays the same too, but does not stand out.
static const float kSameLevel = 8.0f;
static const float kContrastLevel = 24.0f;
// A cell counts as fixed structure with this many kept pixels (of 16), and
// when they are at least a quarter of its edge pixels.
static const float kKeptPixels = 2.0f;
// A cell has changed when a pixel's colour moved by more than this, or it
// has edges and none of them was kept.
static const float kChangeLevel = 24.0f;
// The scene around the cell is moving: the largest flow vector on a ring of
// samples 24 px around it, in pixels between the two frames.
static const float kMotionAround = 1.5f;
static const float kRingRadius = 24.0f;
// The camera moves when this many of 16 samples over the screen do.
static const float kCameraSamples = 10.0f;
// About 14 pairs of evidence (half a second at 30 frames per second, all of
// it while the camera moves) take a cell past one half; one pair of change
// takes it back below it. Seven, earlier, let scenery that held still for a
// moment in.
static const float kRise = 0.05f;
static const float kFall = 0.25f;

float Luma(Texture2D<float4> frame, int2 p, int2 limit)
{
    return dot(frame.Load(int3(clamp(p, int2(0, 0), limit), 0)).rgb, float3(0.299f, 0.587f, 0.114f)) * kLumaScale;
}

float FlowMagnitude(float2 pixel)
{
    const float2 v = FlowScale != 0.0f ? SampleFlow(BackwardFlow, pixel) : SampleFlow(ForwardFlow, pixel);
    return length(v);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const int2 limit = int2(Width, Height) - 1;
    const int2 origin = int2(id.xy) * kCell;

    // Luma of the cell and a one-pixel border, in both frames.
    float l0[kCell + 2][kCell + 2], l1[kCell + 2][kCell + 2];
    [unroll]
    for (int y = 0; y < kCell + 2; ++y)
        [unroll]
        for (int x = 0; x < kCell + 2; ++x)
        {
            const int2 p = origin + int2(x - 1, y - 1);
            l0[y][x] = Luma(Frame0, p, limit);
            l1[y][x] = Luma(Frame1, p, limit);
        }
    float kept = 0.0f;     // edge pixels that kept their gradient, or stayed and stand out
    float edgeCount = 0.0f;
    float change = 0.0f;   // max RGB change, 8-bit levels
    [unroll]
    for (int cy = 1; cy <= kCell; ++cy)
        [unroll]
        for (int cx = 1; cx <= kCell; ++cx)
        {
            const float2 g0 = 0.5f * float2(l0[cy][cx + 1] - l0[cy][cx - 1], l0[cy + 1][cx] - l0[cy - 1][cx]);
            const float2 g1 = 0.5f * float2(l1[cy][cx + 1] - l1[cy][cx - 1], l1[cy + 1][cx] - l1[cy - 1][cx]);
            const float energy = dot(g1, g1);
            const bool edge = energy > kEdgeLevel * kEdgeLevel;
            float mean = 0.0f;
            [unroll]
            for (int dy = -1; dy <= 1; ++dy)
                [unroll]
                for (int dx = -1; dx <= 1; ++dx)
                    mean += l1[cy + dy][cx + dx];
            const bool standsOut = abs(l1[cy][cx] - mean / 9.0f) > kContrastLevel;
            const int2 p = min(origin + int2(cx - 1, cy - 1), limit);
            const float own = MaxComponent(abs(Frame1.Load(int3(p, 0)).rgb - Frame0.Load(int3(p, 0)).rgb)) * kLumaScale;
            change = max(change, own);
            if ((edge && dot(g1 - g0, g1 - g0) < kStructureChange * energy) || (standsOut && own <= kSameLevel))
                kept += 1.0f;
            if (edge || standsOut)
                edgeCount += 1.0f;
        }
    const bool edges = kept >= kKeptPixels;
    const bool steady = kept >= 0.25f * edgeCount;
    const bool changed = change > kChangeLevel || (edgeCount >= kKeptPixels && kept == 0.0f);

    const float2 centre = float2(origin) + 0.5f * (float)(kCell - 1);
    float around = 0.0f;
    [unroll]
    for (int k = 0; k < 8; ++k)
    {
        float s, c;
        sincos(k * 0.785398f, s, c);
        around = max(around, FlowMagnitude(centre + kRingRadius * float2(c, s)));
    }
    // ...and the camera is moving: most of a 4x4 grid of samples over the
    // whole screen moves. A HUD is what stays put while the whole world
    // moves under it. With only something near the cell moving - a
    // character walking past a still camera - a crate's or a fence's edges
    // stay put just like a HUD's, and used to be held as one.
    float movingSamples = 0.0f;
    [unroll]
    for (int gy = 0; gy < 4; ++gy)
        [unroll]
        for (int gx = 0; gx < 4; ++gx)
            movingSamples += FlowMagnitude(float2(Width, Height) * (float2(gx, gy) + 0.5f) / 4.0f) > kMotionAround
                ? 1.0f : 0.0f;
    const bool moving = around > kMotionAround && movingSamples >= kCameraSamples;

    float p = NeuralWeights[59].y > 0.5f ? 0.0f : saturate(PreviousPersistence.Load(int3(id.xy, 0)));
    if (edges && steady && moving)
        p += kRise * (1.0f - p);
    else if (changed)
        p *= kFall;
    Persistence[id.xy] = p;
}
