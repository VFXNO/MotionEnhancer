#include "Common.hlsli"

// FidelityFX Optical Flow emits signed pixel displacements on an 8x8 grid.
// Convert its R16G16_SINT result to the float2 format consumed by InterpolateCS.
Texture2D<int2> FidelityFxFlow : register(t0);
RWTexture2D<float2> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (Width + 7) / 8;
    uint flowHeight = (Height + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;
    Output[id.xy] = float2(FidelityFxFlow.Load(int3(id.xy, 0))) * FlowScale;
}
