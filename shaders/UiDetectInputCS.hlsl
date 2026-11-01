#include "Common.hlsli"

// OmniParser/YOLO: RGB [0,1], centred 640x640 letterbox with 114/255 padding.
// Planar R32 texture -> contiguous GPU NCHW tensor (no full-frame CPU copy).
Texture2D<float4> Frame : register(t0);
RWTexture2D<float> Planar : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint size = 640;
    if (id.x >= size || id.y >= size) return;
    const float scale = min(float(size) / Width, float(size) / Height);
    const float2 resized = max(1.0f, floor(float2(Width, Height) * scale + 0.5f));
    const float2 padding = floor((float2(size, size) - resized) * 0.5f);
    const float2 p = float2(id.xy) - padding;
    float3 rgb = (114.0f / 255.0f).xxx;
    if (all(p >= 0) && all(p < resized))
        rgb = Frame.SampleLevel(LinearClamp, (p + 0.5f) / resized, 0).rgb;
    Planar[uint2(id.x, id.y)] = rgb.r;
    Planar[uint2(id.x, id.y + size)] = rgb.g;
    Planar[uint2(id.x, id.y + 2 * size)] = rgb.b;
}
