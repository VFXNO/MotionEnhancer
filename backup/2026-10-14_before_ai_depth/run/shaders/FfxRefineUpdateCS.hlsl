#include "FfxRefine.hlsli"

// End of one outer iteration of the refinement (FfxRefine.hlsli): the field
// plus the solved update, into the other field buffer.
Texture2D<float2> Field : register(t0);
Texture2D<uint> Delta : register(t1);
RWTexture2D<float2> Updated : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    Updated[id.xy] = Field.Load(int3(id.xy, 0)) + UnpackDelta(Delta.Load(int3(id.xy, 0)));
}
