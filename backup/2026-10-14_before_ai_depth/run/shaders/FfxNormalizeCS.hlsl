#include "Common.hlsli"

// The frame FidelityFX Optical Flow and FfxRepairCS see: the captured frame
// under the gain and offset FfxStatsCS chose for it - unchanged unless the
// whole picture's brightness jumped (see there).
//
// FFX matches raw brightness: on the rendered pair whose second frame is 25
// levels brighter its interpolation was 28 dB in every mode, against DIS's
// 44 dB (DIS removes a gain and offset between the frames first,
// Photometric.hlsli).
Texture2D<float4> Frame : register(t0);
Texture2D<float> State : register(t1);
RWTexture2D<float4> Normalized : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float gain = State.Load(int3(0, 0, 0));
    const float offset = State.Load(int3(1, 0, 0));
    const float3 c = Frame.Load(int3(id.xy, 0)).rgb;
    Normalized[id.xy] = float4(saturate(c * gain + offset / 255.0f), 1.0f);
}
