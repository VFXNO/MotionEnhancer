#include "Common.hlsli"

// A captured frame shrunk to the AI depth network's input (DepthService):
// Width x Height is the source, the output is DepthEstimator::kSize
// squared, each texel the average of the source rectangle it covers. The
// frame's aspect is not kept; MiDaS takes a square image either way.
Texture2D<float4> Frame : register(t0);
RWTexture2D<float4> Small : register(u0);

static const uint kSize = 256;

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
    Small[id.xy] = float4(sum / float((to.x - from.x) * (to.y - from.y)), 1.0f);
}
