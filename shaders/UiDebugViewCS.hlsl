#include "Synthesis.hlsli"

// View 7 only. Source-picture backdrop + the EXACT allow-list saved with the
// pair, and its native pixel evidence. Never changes a mask or runs inference.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float> UiMask : register(t4);
Texture2D<float> HudPersistence : register(t5);   // UiPersistCS (unbound: none)
RWTexture2D<float4> Output : register(u0);

static const float3 kBoxes = float3(0.0f, 0.85f, 1.0f);
static const float3 kEvidence = float3(1.0f, 0.8f, 0.0f);
static const float3 kProtected = float3(0.1f, 1.0f, 0.2f);
static const float3 kFixed = float3(0.35f, 0.45f, 1.0f);   // HUD persistence: screen-fixed structure

// Tiny 3x5 diagnostic font, no texture/font allocation or separate draw call.
uint Glyph(uint ch)
{
    switch (ch)
    {
        case 48: return 0x7b6fu; // 0
        case 49: return 0x2c97u; // 1
        case 50: return 0x62a7u; // 2
        case 51: return 0x628eu; // 3
        case 52: return 0x5bc9u; // 4
        case 53: return 0x798eu; // 5
        case 54: return 0x39aau; // 6
        case 55: return 0x7292u; // 7
        case 56: return 0x2aaau; // 8
        case 57: return 0x2aceu; // 9
        case 65: return 0x2bedu; // A
        case 66: return 0x6baeu; // B
        case 67: return 0x3923u; // C
        case 68: return 0x6b6eu; // D
        case 69: return 0x79a7u; // E
        case 70: return 0x79a4u; // F
        case 71: return 0x396bu; // G
        case 73: return 0x7497u; // I
        case 75: return 0x5badu; // K
        case 76: return 0x4927u; // L
        case 77: return 0x5fedu; // M
        case 78: return 0x5ffdu; // N
        case 79: return 0x2b6au; // O
        case 80: return 0x6ba4u; // P
        case 82: return 0x6badu; // R
        case 83: return 0x388eu; // S
        case 84: return 0x7492u; // T
        case 85: return 0x5b6fu; // U
        case 86: return 0x5b6au; // V
        case 88: return 0x5aadu; // X
        case 58: return 0x410u; // :
    }
    return 0u;
}

uint StatusChar(uint i, int state, bool paired, uint count)
{
    static const uint prefix[] = {65, 73, 32, 85, 73, 58, 32}; // AI UI:
    if (i < 7u) return prefix[i];
    i -= 7u;
    static const uint maskOff[] = {77, 65, 83, 75, 32, 79, 70, 70};
    static const uint aiOff[] = {65, 73, 32, 79, 70, 70};
    static const uint noPair[] = {78, 79, 32, 80, 65, 73, 82};
    static const uint noResult[] = {78, 79, 32, 82, 69, 83, 85, 76, 84};
    static const uint regions[] = {82, 69, 71, 73, 79, 78, 83, 58, 32};
    if (state == 0) return i < 8u ? maskOff[i] : 32u;
    if (state == 1) return i < 6u ? aiOff[i] : 32u;
    if (!paired) return i < 7u ? noPair[i] : 32u;
    if (state == 2) return i < 9u ? noResult[i] : 32u;
    if (i < 9u) return regions[i];
    if (i == 9u) return 48u + count / 10u;
    if (i == 10u) return 48u + count % 10u;
    return 32u;
}

float3 Legend(uint2 pixel, float3 color, int state, bool paired, uint count)
{
    const uint scale = Width >= 1600u ? 3u : 2u;
    if (pixel.x < 6u || pixel.y < 36u) return color;
    const uint2 p = (pixel - uint2(6, 36)) / scale;
    if (p.x >= 132u || p.y >= 17u) return color;
    color = float3(0.025f, 0.025f, 0.035f);
    if (p.x < 2u) return color;
    const uint x = p.x - 2u;
    uint ch = 32u, y = 0u;
    float3 ink = state == 0 || state == 1 ? float3(0.65f, 0.65f, 0.65f)
        : paired && state == 3 ? float3(1, 1, 1) : kEvidence;
    if (p.y >= 2u && p.y < 7u)
    {
        ch = StatusChar(x / 4u, state, paired, count);
        y = p.y - 2u;
    }
    else if (p.y >= 10u && p.y < 15u)
    {
        // BOXES   EVIDENCE   MASK   FIXED
        static const uint key[] = {66,79,88,69,83,32,32,32,69,86,73,68,69,78,67,69,32,32,32,77,65,83,75,
                                   32,32,32,70,73,88,69,68};
        const uint index = x / 4u;
        ch = index < 31u ? key[index] : 32u;
        y = p.y - 10u;
        ink = index < 8u ? kBoxes : index < 19u ? kEvidence : index < 26u ? kProtected : kFixed;
    }
    if (x % 4u < 3u && ((Glyph(ch) >> (14u - (y * 3u + x % 4u))) & 1u) != 0u) color = ink;
    return color;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const int3 texel = int3(id.xy, 0);
    const bool paired = NeuralControl.w > 0.5f;
    const int state = clamp((int)NeuralControl.x, 0, 3);
    const uint count = state == 3 && paired ? (uint)clamp((int)NeuralControl.z, 0, 32) : 0u;
    float3 source = Frame0.Load(texel).rgb;
    if (paired) source = lerp(source, Frame1.Load(texel).rgb, saturate(TimeT));
    float3 color = 0.65f * DebugGrey(source);
    if (paired && state != 0)
    {
        // Screen-fixed structure (UiPersistCS cells at or above one half).
        uint hw, hh;
        HudPersistence.GetDimensions(hw, hh);
        if (hw > 0u && HudPersistence.Load(int3(min(id.xy / 4u, uint2(hw, hh) - 1u), 0)) >= 0.5f)
            color = lerp(color, kFixed, 0.45f);
    }

    if (paired && state != 0)
    {
        const float value = UiMask.Load(texel);
        if (DecodeUiCount(value) > 0.0f) color = lerp(color, kEvidence, 0.75f);
        float coverage = UiCoverage(value);
        if (coverage > 0.0f)
        {
            // Same 3x3 support rule as InterpolateCS (kUiSupport = 2). Green
            // means actual protected coverage, not just a confirmed lone pixel.
            const int2 limit = int2(Width, Height) - 1;
            float support = 0.0f;
            [unroll]
            for (int dy = -1; dy <= 1; ++dy)
                [unroll]
                for (int dx = -1; dx <= 1; ++dx)
                    support += UiCoverage(UiMask.Load(int3(clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), limit), 0)));
            coverage *= saturate(support - 1.0f);
        }
        color = lerp(color, kProtected, coverage);
    }
    const float2 p = float2(id.xy) + 0.5f;
    for (uint i = 0; i < count; ++i)
    {
        const float4 bounds = NeuralWeights[i] * float4(Width, Height, Width, Height);
        if (all(p >= bounds.xy) && all(p < bounds.zw) &&
            min(min(p.x - bounds.x, p.y - bounds.y), min(bounds.z - p.x, bounds.w - p.y)) < 2.0f)
            color = kBoxes;
    }
    color = Legend(id.xy, color, state, paired, count);
    const bool real = TimeT < 0.01f || TimeT > 0.99f;
    Output[id.xy] = float4(DebugMarker(id.xy, color, !paired ? float3(1.0f, 0.85f, 0.0f)
        : real ? float3(0.1f, 1.0f, 0.2f) : float3(1.0f, 0.15f, 0.8f)), 1.0f);
}
