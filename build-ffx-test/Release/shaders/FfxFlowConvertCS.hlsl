#include "Common.hlsli"

// FidelityFX Optical Flow emits signed pixel displacements on an 8x8 grid.
// Convert its R16G16_SINT result to the flow format consumed by InterpolateCS.
Texture2D<int2> FidelityFxFlow : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (Width + 7) / 8;
    uint flowHeight = (Height + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;
    // Confidence 1 on both axes: FFX does its own field regularization
    // internally and exposes no per-block measure, so every block is taken
    // as given. FillFlowCS never runs on this path.
    Output[id.xy] = float4(float2(FidelityFxFlow.Load(int3(id.xy, 0))) * FlowScale, 1.0f, 1.0f);
}
