#include "FfxRefine.hlsli"

// One outer iteration of the refinement (FfxRefine.hlsli), step 1: the data
// terms linearised around the current field, and the robust weights frozen
// for this iteration. Per half-resolution pixel x, with frame0 warped by the
// field (p = x + u):
//
//   brightness  Iz = I0(p) - I1(x),        d/du = (I0x, I0y)(p)
//   gradients   G  = grad I0(p) - grad I1(x),  d/du = Hessian of I0 at p
//
// and the normal equations of the weighted linearised data terms, A du = b.
// The smoothness weight alpha * psi'(|grad u|^2) is stored alongside: the
// solver blends it with each neighbour's.
//
// Output: a (2 * LevelWidth) x LevelHeight R32G32B32A32_FLOAT texture,
// texel (2x, y) = (A11, A12, A22, smoothness weight), (2x + 1, y) = (b1, b2).
// A sample whose warped position is outside frame0 has no data term: the
// smoothness term carries the field there.
Texture2D<float> RefLuma : register(t0);     // frame1 (current), half resolution, 0..255
Texture2D<float> CandLuma : register(t1);    // frame0 (previous)
Texture2D<float2> Field : register(t2);      // current field, half-resolution pixels
Texture2D<float2> RefStats : register(t3);   // whole-frame luma mean, deviation
Texture2D<float2> CandStats : register(t4);
RWTexture2D<float4> Coefficients : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= LevelWidth || id.y >= LevelHeight) return;
    const int2 size = int2(LevelWidth, LevelHeight);
    const int2 limit = size - 1;
    const int2 x = int2(id.xy);
    const float2 inverseSize = rcp(float2(size));

    float refGain, refOffset, candGain, candOffset;
    const float2 refStats = RefStats.Load(int3(0, 0, 0));
    const float2 candStats = CandStats.Load(int3(0, 0, 0));
    PairMapping(refStats, candStats, refGain, refOffset);
    PairMapping(candStats, refStats, candGain, candOffset);

    const float i1 = refGain * RefLuma.Load(int3(x, 0)) + refOffset;
    const float i1x = 0.5f * refGain * (RefLuma.Load(int3(min(x + int2(1, 0), limit), 0)) -
                                        RefLuma.Load(int3(max(x - int2(1, 0), int2(0, 0)), 0)));
    const float i1y = 0.5f * refGain * (RefLuma.Load(int3(min(x + int2(0, 1), limit), 0)) -
                                        RefLuma.Load(int3(max(x - int2(0, 1), int2(0, 0)), 0)));

    const float2 u = Field.Load(int3(x, 0));
    const float2 p = float2(x) + u;
    #define S0(q) (candGain * CandLuma.SampleLevel(LinearClamp, ((q) + 0.5f) * inverseSize, 0) + candOffset)
    const float c = S0(p);
    const float xp = S0(p + float2(1, 0)), xm = S0(p - float2(1, 0));
    const float yp = S0(p + float2(0, 1)), ym = S0(p - float2(0, 1));
    const float pp = S0(p + float2(1, 1)), pm = S0(p + float2(1, -1));
    const float mp = S0(p + float2(-1, 1)), mm = S0(p + float2(-1, -1));
    #undef S0
    const float ix = 0.5f * (xp - xm);
    const float iy = 0.5f * (yp - ym);
    const float ixx = xp - 2.0f * c + xm;
    const float iyy = yp - 2.0f * c + ym;
    const float ixy = 0.25f * (pp - pm - mp + mm);

    const float iz = c - i1;
    const float gx = ix - i1x;
    const float gy = iy - i1y;
    const bool inside = all(p >= 0.0f) && all(p <= float2(limit));
    const float wb = inside ? kRefineDelta * CharbonnierWeight(iz * iz) : 0.0f;
    const float wg = inside ? kRefineGamma * CharbonnierWeight(gx * gx + gy * gy) : 0.0f;

    const float a11 = wb * ix * ix + wg * (ixx * ixx + ixy * ixy);
    const float a12 = wb * ix * iy + wg * (ixx * ixy + ixy * iyy);
    const float a22 = wb * iy * iy + wg * (ixy * ixy + iyy * iyy);
    const float b1 = -(wb * iz * ix + wg * (gx * ixx + gy * ixy));
    const float b2 = -(wb * iz * iy + wg * (gx * ixy + gy * iyy));

    // Smoothness: forward differences of the current field.
    const float2 du = Field.Load(int3(min(x + int2(1, 0), limit), 0)) - u;
    const float2 dv = Field.Load(int3(min(x + int2(0, 1), limit), 0)) - u;
    const float smooth = kRefineAlpha * CharbonnierWeight(dot(du, du) + dot(dv, dv));

    Coefficients[uint2(2 * id.x, id.y)] = float4(a11, a12, a22, smooth);
    Coefficients[uint2(2 * id.x + 1, id.y)] = float4(b1, b2, 0.0f, 0.0f);
}
