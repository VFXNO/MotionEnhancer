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
    float4 NeuralWeights[60];
    float4 NeuralControl;   // x = model enabled, yzw spare
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
