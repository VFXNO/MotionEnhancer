#include "Common.hlsli"

// The AI depth network's input, made on the GPU (DepthService): the captured
// frame (Width x Height) averaged down to kSize x kSize, normalised the way
// MiDaS was trained (ImageNet mean and deviation) and laid out planar - R in
// rows 0..kSize-1, G below, B below that - in a kSize x 3*kSize R32_FLOAT
// texture whose rows, copied into a buffer, are exactly the NCHW tensor. The
// frame's aspect is not kept; MiDaS takes a square image either way.
Texture2D<float4> Frame : register(t0);
RWTexture2D<float> Planar : register(u0);

static const uint kSize = 256;
static const float3 kMean = float3(0.485f, 0.456f, 0.406f);
static const float3 kDeviation = float3(0.229f, 0.224f, 0.225f);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= kSize || id.y >= kSize) return;
    const uint2 source = uint2(Width, Height);
    const uint2 from = id.xy * source / kSize;
    const uint2 to = max(from + 1u, (id.xy + 1u) * source / kSize);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (uint y = from.y; y < to.y; ++y)
        for (uint x = from.x; x < to.x; ++x)
            sum += Frame.Load(int3(x, y, 0)).rgb;
    const float3 rgb = (sum / float((to.x - from.x) * (to.y - from.y)) - kMean) / kDeviation;
    Planar[uint2(id.x, id.y)] = rgb.r;
    Planar[uint2(id.x, id.y + kSize)] = rgb.g;
    Planar[uint2(id.x, id.y + 2u * kSize)] = rgb.b;
}
