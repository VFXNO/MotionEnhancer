// Layout must match ShaderConstants in include/GraphicsDevice.h exactly
// (sixty-four 16-byte rows).
cbuffer PipelineConstants : register(b0)
{
    uint Width;          // full-resolution frame size
    uint Height;
    uint LevelWidth;     // size of the pyramid level the current pass works on
    uint LevelHeight;

    int SearchRadius;    // block-match search radius in level pixels (0..8)
    int BlockSize;
    int LevelIndex;      // 0 = finest, TotalLevels-1 = coarsest
    int TotalLevels;

    float TimeT;         // interpolation time in [0,1]: 0 = frame0, 1 = frame1
    float SmoothnessWeight;
    uint ParentBlockSize;
    float FlowScale;      // explicit scale/sign for flow conversion passes

    // The convolutional estimator's weights in NeuralFlowLayout order,
    // indexed through NeuralWeight() below with the same constants the C++
    // uses. 240 floats = 60 rows, of which 238 are the model.
    // UiMaskCS / UiDebugViewCS: first 32 float4s instead hold UI region bounds.
    float4 NeuralWeights[60];
    float4 NeuralControl;   // x = model enabled, y = debug view, z = alternate DIS seed, w = AI blend mask
};

// Indices into NeuralWeights, mirroring namespace NeuralFlowLayout in
// include/NeuralFlow.h. Changing one without the other is the failure mode
// --neural-parity-check exists to catch.
static const int kNeuralHiddenChannels = 6;
static const int kNeuralFeatures = 2;
static const int kNeuralCosts = 6;
static const int kNeuralUnits = 8;

static const int kwConv1 = 0;                                        // [6][9]
static const int kwConv1Bias = kwConv1 + kNeuralHiddenChannels * 9;  // 54
static const int kwConv2 = kwConv1Bias + kNeuralHiddenChannels;      // 60, [2][6][9]
static const int kwConv2Bias = kwConv2 + kNeuralFeatures * kNeuralHiddenChannels * 9; // 168
static const int kwParentGain = kwConv2Bias + kNeuralFeatures;       // 170
static const int kwHeadWeight = kwParentGain + kNeuralFeatures;      // 172, [8][6]
static const int kwHeadBias = kwHeadWeight + kNeuralUnits * kNeuralCosts; // 220
static const int kwOut = kwHeadBias + kNeuralUnits;                  // 228
static const int kwScoreBias = kwOut + kNeuralUnits;                 // 236
static const int kwLogTemperature = kwScoreBias + 1;                 // 237

// Every call site indexes with a compile-time constant (the loops that read
// weights are all [unroll]), so this folds to a direct constant-buffer read
// rather than the select chain a dynamic component index would produce.
float NeuralWeight(int index)
{
    return NeuralWeights[index / 4][index % 4];
}

SamplerState LinearClamp : register(s0);
SamplerState PointClamp : register(s1);

// ------------------------------------------------------------ debug views
// Live diagnostics for the presented frames, cycled with Ctrl+Alt+F3 in the
// overlay (Presenter sets NeuralControl.y; MOTION_ENHANCER_DEBUG_VIEW does
// the same offline): 0 = off, 1 = motion, 2 = static-UI evidence,
// 3 = occlusion trust, 4 = vector arrows, 5 = AI depth, 6 = raw AI blend mask,
// 7 = AI UI regions + native pixel mask (UiDebugViewCS).
// Views 1..4 grey the picture so the colours read; views 5..6 show maps.
// Every view marks the top-left corner with what kind of frame this is - green a real
// frame, magenta a generated one, yellow a frame shown with no
// interpolation (held, a scene cut, startup) - followed by one white dot per
// view number.
int DebugView()
{
    return (int)NeuralControl.y;
}

float3 DebugGrey(float3 color)
{
    return (0.2f + 0.6f * dot(color, float3(0.299f, 0.587f, 0.114f))).xxx;
}

// A missing/disabled pair mask must not look like a valid neutral grey mask.
// Shared by interpolation and pass-through frames (which have no pair).
float3 DebugAiBlendMaskInactive(uint2 p)
{
    return (((p.x / 16u) ^ (p.y / 16u)) & 1u) != 0u
        ? float3(0.24f, 0.04f, 0.32f) : float3(0.12f, 0.02f, 0.18f);
}

// A fully saturated colour for hue in [0, 1).
float3 DebugHue(float hue)
{
    return saturate(abs(frac(hue + float3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f) - 1.0f);
}

float3 DebugMarker(uint2 p, float3 color, float3 kind)
{
    if (p.y < 6 || p.y >= 30) return color;
    if (p.x >= 6 && p.x < 30) return (p.x >= 8 && p.x < 28 && p.y >= 8 && p.y < 28) ? kind : float3(0.0f, 0.0f, 0.0f);
    const int views = DebugView();
    for (int k = 0; k < views; ++k)
    {
        const uint x0 = 36 + 12 * (uint)k;
        if (p.x >= x0 && p.x < x0 + 8 && p.y >= 14 && p.y < 22) return float3(1.0f, 1.0f, 1.0f);
        if (p.x >= x0 - 1 && p.x < x0 + 9 && p.y >= 13 && p.y < 23) return float3(0.0f, 0.0f, 0.0f);
    }
    return color;
}

// Flow block size is supplied per pyramid level through BlockSize.

// Luma is stored as 0..255 float (R16_FLOAT), so every threshold expressed
// in these units is an 8-bit luma level.
static const float kLumaScale = 255.0f;

float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

uint2 FlowGridSize(uint levelWidth, uint levelHeight, uint blockSize)
{
    return uint2((levelWidth + blockSize - 1) / blockSize,
                 (levelHeight + blockSize - 1) / blockSize);
}


// The ZNCC/area-statistics helpers that used to live here went with the
// CPU matcher they served: no shader references them. Flow block size is
// supplied per pyramid level through BlockSize.
