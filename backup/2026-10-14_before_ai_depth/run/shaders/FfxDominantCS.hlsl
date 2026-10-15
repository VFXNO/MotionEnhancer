#include "Common.hlsli"

// The dominant motions of FFX's 8x8 field (whole pixels, current ->
// previous; DominantMotion.hlsli), offered to every block of FfxRepairCS.
Texture2D<int2> Field : register(t0);

uint2 FieldSize()
{
    uint w, h;
    Field.GetDimensions(w, h);
    return uint2(w, h);
}

float2 FieldAt(uint2 texel)
{
    return float2(Field.Load(int3(texel, 0)));
}

#include "DominantMotion.hlsli"

[numthreads(32, 32, 1)]
void CSMain(uint li : SV_GroupIndex)
{
    FindDominant(li);
}
