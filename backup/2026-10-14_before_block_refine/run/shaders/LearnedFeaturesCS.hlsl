#include "Common.hlsli"

// Feature extractor of the convolutional flow estimator: 3x3 -> kHidden
// ReLU, then 3x3 -> 2, plus a gain on the next coarser level's features,
// then tanh. Must stay bit-for-bit comparable with buildLevel() in
// NeuralFlowTrain.cpp and featureLevel() in NeuralFlow.cpp - the matcher
// uses weights trained against that CPU forward pass, and
// --neural-parity-check is what holds the three in agreement.
//
// One pass, with conv1 staged in groupshared memory. The alternatives were
// both worse: a second texture for the hidden layer makes conv2 read
// 9 taps x kHidden channels through memory (the one thing the 1080p
// pipeline has no spare bandwidth for), and recomputing conv1 at each of
// conv2's nine taps does nine times the multiplies for the same answer.
// A 16x16 output tile needs an 18x18 hidden tile, which needs a 20x20
// luma tile.
Texture2D<float> Luma : register(t0);
Texture2D<float2> ParentFeatures : register(t1);
RWTexture2D<float2> Features : register(u0);

static const int kHidden = kNeuralHiddenChannels;
static const int kTile = 16;
static const int kHiddenTile = kTile + 2;   // 18: conv2 reads one ring out

groupshared float HiddenTile[kHiddenTile][kHiddenTile][kHidden];

[numthreads(kTile, kTile, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 levelMax = int2(LevelWidth, LevelHeight) - 1;
    // Output tile origin; the hidden tile starts one pixel before it.
    const int2 tileOrigin = int2(groupId.xy) * kTile;
    const int2 hiddenOrigin = tileOrigin - int2(1, 1);

    // conv1 for every pixel of the hidden tile. 324 positions over 256
    // threads, so two rounds.
    [loop]
    for (int t = (int)li; t < kHiddenTile * kHiddenTile; t += kTile * kTile)
    {
        const int2 local = int2(t % kHiddenTile, t / kHiddenTile);
        const int2 centre = hiddenOrigin + local;
        // The 3x3 luma neighbourhood, clamped exactly as Image::get clamps
        // on the CPU, or the border pixels disagree between the two.
        float luma[9];
        [unroll]
        for (int tap = 0; tap < 9; ++tap)
        {
            const int2 p = clamp(centre + int2(tap % 3 - 1, tap / 3 - 1),
                                 int2(0, 0), levelMax);
            luma[tap] = Luma.Load(int3(p, 0)) / kLumaScale;
        }
        [unroll]
        for (int k = 0; k < kHidden; ++k)
        {
            float sum = NeuralWeight(kwConv1Bias + k);
            [unroll]
            for (int tap = 0; tap < 9; ++tap)
                sum += NeuralWeight(kwConv1 + k * 9 + tap) * luma[tap];
            HiddenTile[local.y][local.x][k] = max(0.0f, sum);   // ReLU
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 pixel = tileOrigin + int2(tid.xy);
    if (pixel.x >= (int)LevelWidth || pixel.y >= (int)LevelHeight) return;

    // conv2 over the staged hidden tile. This thread's output pixel sits at
    // HiddenTile[tid.y + 1][tid.x + 1], so its 3x3 ring is in range.
    float2 value = float2(NeuralWeight(kwConv2Bias + 0), NeuralWeight(kwConv2Bias + 1));
    [unroll]
    for (int k = 0; k < kHidden; ++k)
        [unroll]
        for (int tap = 0; tap < 9; ++tap)
        {
            const float hidden = HiddenTile[(int)tid.y + 1 + tap / 3 - 1]
                                           [(int)tid.x + 1 + tap % 3 - 1][k];
            value.x += NeuralWeight(kwConv2 + (0 * kHidden + k) * 9 + tap) * hidden;
            value.y += NeuralWeight(kwConv2 + (1 * kHidden + k) * 9 + tap) * hidden;
        }
    if (LevelIndex < TotalLevels - 1)
    {
        uint parentWidth, parentHeight;
        ParentFeatures.GetDimensions(parentWidth, parentHeight);
        const int2 parent = clamp(pixel / 2, int2(0, 0),
                                  int2(parentWidth, parentHeight) - 1);
        value += ParentFeatures.Load(int3(parent, 0)) *
                 float2(NeuralWeight(kwParentGain + 0), NeuralWeight(kwParentGain + 1));
    }
    Features[pixel] = tanh(value);
}
