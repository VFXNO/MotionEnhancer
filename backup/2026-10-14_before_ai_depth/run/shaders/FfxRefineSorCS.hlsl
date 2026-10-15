#include "FfxRefine.hlsli"

// One half-sweep of red-black successive over-relaxation for the update du
// of the refinement (FfxRefine.hlsli). Per pixel of the given colour, the
// 2x2 system
//
//   (A + sum_n w_n I) du = b + sum_n w_n (u_n + du_n - u)
//
// (data terms from FfxRefinePrepCS, smoothness towards each 4-neighbour n
// with the mean of the two pixels' weights) is solved with the neighbours'
// current du, and du moves there by kRefineOmega. Red pixels only read
// black neighbours and vice versa, so a half-sweep updates in place.
//
// SearchRadius carries flags: bit 0 = colour ((x + y) & 1) to update,
// bit 1 = this pixel's stored du is stale (the first sweep of an outer
// iteration: start from 0), bit 2 = the neighbours' is stale too (the first
// half-sweep of it).
Texture2D<float4> Coefficients : register(t0);
Texture2D<float2> Field : register(t1);
RWTexture2D<uint> Delta : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const uint flags = (uint)SearchRadius;
    if (((id.x + id.y) & 1u) != (flags & 1u)) return;
    const bool ownStale = (flags & 2u) != 0u;
    const bool neighboursStale = (flags & 4u) != 0u;

    const int2 size = int2(LevelWidth, LevelHeight);
    const int2 x = int2(id.xy);
    const float4 a = Coefficients.Load(int3(2 * x.x, x.y, 0));
    const float2 b = Coefficients.Load(int3(2 * x.x + 1, x.y, 0)).xy;
    const float2 u = Field.Load(int3(x, 0));

    float2 rhs = b;
    float weights = 0.0f;
    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    [unroll]
    for (int k = 0; k < 4; ++k)
    {
        const int2 n = x + offsets[k];
        if (any(n < 0) || any(n >= size)) continue;
        const float w = 0.5f * (a.w + Coefficients.Load(int3(2 * n.x, n.y, 0)).w);
        const float2 dn = neighboursStale ? float2(0.0f, 0.0f) : UnpackDelta(Delta[n]);
        rhs += w * (Field.Load(int3(n, 0)) + dn - u);
        weights += w;
    }
    const float m11 = a.x + weights;
    const float m12 = a.y;
    const float m22 = a.z + weights;
    const float det = m11 * m22 - m12 * m12;
    const float2 solved = det > 1e-12f ? float2(m22 * rhs.x - m12 * rhs.y, m11 * rhs.y - m12 * rhs.x) / det
                                       : float2(0.0f, 0.0f);
    const float2 own = ownStale ? float2(0.0f, 0.0f) : UnpackDelta(Delta[x]);
    Delta[x] = PackDelta(lerp(own, solved, kRefineOmega));
}
