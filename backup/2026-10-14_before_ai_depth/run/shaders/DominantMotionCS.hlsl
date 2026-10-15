#include "Common.hlsli"

// The dominant motions of a float flow field (DominantMotion.hlsli): DIS's
// densified flow of the next coarser level, offered to every patch of the
// level below (DisPatchCS).
Texture2D<float2> Field : register(t0);

uint2 FieldSize()
{
    uint w, h;
    Field.GetDimensions(w, h);
    return uint2(w, h);
}

float2 FieldAt(uint2 texel)
{
    return Field.Load(int3(texel, 0));
}

#include "DominantMotion.hlsli"

[numthreads(32, 32, 1)]
void CSMain(uint li : SV_GroupIndex)
{
    FindDominant(li);
}
