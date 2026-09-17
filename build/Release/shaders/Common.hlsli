// Layout must match ShaderConstants in include/D3D11Context.h exactly
// (three 16-byte rows).
cbuffer PipelineConstants : register(b0)
{
    uint Width;          // full-resolution frame size
    uint Height;
    uint LevelWidth;     // size of the pyramid level the current pass works on
    uint LevelHeight;

    int SearchRadius;    // block-match search radius in level pixels (0..8)
    int BlockSize;       // unused by the shaders (blocks are fixed 16x16)
    int LevelIndex;      // 0 = finest, TotalLevels-1 = coarsest
    int TotalLevels;

    float TimeT;         // interpolation time in [0,1]: 0 = frame0, 1 = frame1
    float SmoothnessWeight;
    uint Pad0;
    uint Pad1;
};

SamplerState LinearClamp : register(s0);
SamplerState PointClamp : register(s1);

// Flow is estimated on a grid of one float vector per 16x16 block, at every
// pyramid level, in that level's pixel units. The vector for block b is the
// displacement that moves the reference (frame0) block at b*8 onto its best
// match in the candidate frame (frame1).
static const int kFlowBlock = 16;

// Luma is stored as 0..255 float (R16_FLOAT), so every threshold below is in
// 8-bit units.
static const float kLumaScale = 255.0f;

// Blocks whose standard deviation is below this are "flat": ZNCC on them is
// 0/0 noise, so they are matched by mean brightness instead.
static const float kFlatStd = 1.0f;

float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

uint2 FlowGridSize(uint levelWidth, uint levelHeight)
{
    return uint2((levelWidth + kFlowBlock - 1) / kFlowBlock,
                 (levelHeight + kFlowBlock - 1) / kFlowBlock);
}

static const int kContext = 16;
static const int kContextPad = (kContext - kFlowBlock) / 2;
static const int kSmallSupportPad = (kFlowBlock - 8) / 2;

struct RefArea
{
    int2 origin;
    int size;
    float mean;
    float std;
};



// Score of candidate displacement `motion` for reference area `ref` in
// [-1, 1]: ZNCC, with flat areas matched by mean brightness the same way as
// the CPU matcher (ZNCCMatcher.cpp computeWeightedZNCC).
float ScoreFromSums(RefArea ref, float sumC, float sumC2, float sumRC)
{
    float n = (float)(ref.size * ref.size);
    float meanC = sumC / n;
    float stdC = sqrt(max(sumC2 / n - meanC * meanC, 0.0f));
    bool refFlat = ref.std < kFlatStd;
    bool candFlat = stdC < kFlatStd;
    if (refFlat && candFlat)
    {
        float diff = abs(ref.mean - meanC);
        return (diff < 5.0f) ? (1.0f - diff * 0.05f) : -1.0f;
    }
    if (refFlat || candFlat)
    {
        return -1.0f;
    }
    float cov = sumRC / n - ref.mean * meanC;
    return clamp(cov / (ref.std * stdC + 1e-6f), -1.0f, 1.0f);
}

RefArea AreaStats(Texture2D<float> image, int2 origin, int size, int2 levelMax)
{
    float sum = 0.0f, sumSq = 0.0f;
    [loop]
    for (int y = 0; y < size; ++y)
    {
        [loop]
        for (int x = 0; x < size; ++x)
        {
            int2 p = origin + int2(x, y);
            float v = image.Load(int3(clamp(p, int2(0, 0), levelMax), 0));
            sum += v;
            sumSq += v * v;
        }
    }
    float n = (float)(size * size);
    RefArea r;
    r.origin = origin;
    r.size = size;
    r.mean = sum / n;
    float var = max(sumSq / n - r.mean * r.mean, 0.0f);
    r.std = sqrt(var);
    return r;
}

// Picks the reference area for the flow block at `blockOrigin` (see above).
RefArea ChooseReferenceArea(Texture2D<float> image, int2 blockOrigin, int2 levelMax)
{
    return AreaStats(image, blockOrigin - int2(kContextPad, kContextPad),
                     kContext, levelMax);
}

float ScoreCandidate(Texture2D<float> reference, Texture2D<float> candidate,
                     RefArea ref, int2 motion, int2 levelMax)
{
    float sumC = 0.0f, sumC2 = 0.0f, sumRC = 0.0f;
    [loop]
    for (int y = 0; y < ref.size; ++y)
    {
        [loop]
        for (int x = 0; x < ref.size; ++x)
        {
            int2 p = clamp(ref.origin + int2(x, y), int2(0, 0), levelMax);
            int2 q = clamp(p + motion, int2(0, 0), levelMax);
            float rv = reference.Load(int3(p, 0));
            float cv = candidate.Load(int3(q, 0));
            sumC += cv;
            sumC2 += cv * cv;
            sumRC += rv * cv;
        }
    }
    return ScoreFromSums(ref, sumC, sumC2, sumRC);
}
