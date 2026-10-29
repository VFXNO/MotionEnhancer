#include "Common.hlsli"

// FidelityFX Optical Flow emits signed pixel displacements on an 8x8 grid
// (repaired by FfxRepairCS into R32G32_FLOAT). Convert it to the flow format
// consumed by InterpolateCS.
Texture2D<float4> FidelityFxFlow : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint flowWidth = (Width + 7) / 8;
    uint flowHeight = (Height + 7) / 8;
    if (id.x >= flowWidth || id.y >= flowHeight) return;
    // Confidence 1 on both axes: FFX does its own field regularization
    // internally and exposes no per-block measure, so every block is taken
    // as given. FillFlowCS never runs on this path: filling FFX's flat
    // blocks from their textured neighbours (with SAD's structure measure as
    // the confidence) did remove its zero-vector runs on a fast pan, but it
    // also filled the flat inside of moving objects and of a third-person
    // character with the background's motion (-1.2 dB, character -0.4 dB).
    Output[id.xy] = float4(FidelityFxFlow.Load(int3(id.xy, 0)).xy * FlowScale, 1.0f, 1.0f);
}
