#include "FfxRefine.hlsli"

// One outer iteration of the refinement (FfxRefine.hlsli) in one dispatch.
//
// 1. Linearise the data terms around the current field, per half-resolution
//    pixel x with frame0 warped by it (p = x + u):
//      brightness  Iz = I0(p) - I1(x),                d/du = grad I0(p)
//      gradients   G  = grad I0(p) - grad I1(x),      d/du = Hessian of I0 at p
//    into the normal equations A du = b, with the robust weights frozen for
//    this iteration; the smoothness weight alpha * psi'(|grad u|^2) from the
//    current field. A pixel whose warped position is outside frame0 has no
//    data term: the smoothness term carries the field there.
// 2. Solve for du by red-black successive over-relaxation, per pixel the
//    2x2 system (A + sum_n w_n I) du = b + sum_n w_n (u_n + du_n - u) with
//    each 4-neighbour n weighted by the mean of the two pixels' smoothness
//    weights, moving du there by kRefineOmega.
// 3. Write u + du.
//
// The solve runs in groupshared memory on a 16x16 tile, and only the inner
// 12x12 is written: the tiles overlap by a 2-pixel apron on every side, so
// the tile's own border - where neighbours outside it are left out - never
// reaches an output pixel. Separate passes per half-sweep cost 1.65 ms per
// 1440p pair, almost all of it going through memory between passes.
//
// Constants: LevelWidth/Height = the half-resolution field.
Texture2D<float> RefLuma : register(t0);     // the field's reference frame, half resolution, 0..255
Texture2D<float> CandLuma : register(t1);    // the frame it points into
Texture2D<float2> Field : register(t2);      // current field, half-resolution pixels
Texture2D<float2> RefStats : register(t3);   // whole-frame luma mean, deviation
Texture2D<float2> CandStats : register(t4);
RWTexture2D<float2> Updated : register(u0);

static const int kTile = 16;
static const int kApron = 2;
static const int kInterior = kTile - 2 * kApron;
static const int kSweeps = 5;

groupshared float4 SharedA[kTile * kTile];    // A11, A12, A22, smoothness weight
groupshared float2 SharedB[kTile * kTile];
groupshared float2 SharedU[kTile * kTile];
groupshared float2 SharedDu[kTile * kTile];

[numthreads(16, 16, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 tid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 size = int2(LevelWidth, LevelHeight);
    const int2 limit = size - 1;
    const int2 x = int2(groupId.xy) * kInterior - kApron + int2(tid.xy);
    const bool valid = all(x >= 0) && all(x < size);
    const int2 xc = clamp(x, int2(0, 0), limit);
    const float2 inverseSize = rcp(float2(size));

    float refGain, refOffset, candGain, candOffset;
    const float2 refStats = RefStats.Load(int3(0, 0, 0));
    const float2 candStats = CandStats.Load(int3(0, 0, 0));
    PairMapping(refStats, candStats, refGain, refOffset);
    PairMapping(candStats, refStats, candGain, candOffset);

    // --- 1. linearised data terms and smoothness weight --------------------
    const float i1 = refGain * RefLuma.Load(int3(xc, 0)) + refOffset;
    const float i1x = 0.5f * refGain * (RefLuma.Load(int3(min(xc + int2(1, 0), limit), 0)) -
                                        RefLuma.Load(int3(max(xc - int2(1, 0), int2(0, 0)), 0)));
    const float i1y = 0.5f * refGain * (RefLuma.Load(int3(min(xc + int2(0, 1), limit), 0)) -
                                        RefLuma.Load(int3(max(xc - int2(0, 1), int2(0, 0)), 0)));
    const float2 u = Field.Load(int3(xc, 0));
    const float2 p = float2(xc) + u;
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
    const float2 fu = Field.Load(int3(min(xc + int2(1, 0), limit), 0)) - u;
    const float2 fv = Field.Load(int3(min(xc + int2(0, 1), limit), 0)) - u;
    const float smooth = kRefineAlpha * CharbonnierWeight(dot(fu, fu) + dot(fv, fv));

    SharedA[li] = float4(wb * ix * ix + wg * (ixx * ixx + ixy * ixy),
                         wb * ix * iy + wg * (ixx * ixy + ixy * iyy),
                         wb * iy * iy + wg * (ixy * ixy + iyy * iyy), smooth);
    SharedB[li] = -float2(wb * iz * ix + wg * (gx * ixx + gy * ixy),
                          wb * iz * iy + wg * (gx * ixy + gy * iyy));
    SharedU[li] = u;
    SharedDu[li] = float2(0.0f, 0.0f);
    GroupMemoryBarrierWithGroupSync();

    // --- 2. red-black SOR in the tile --------------------------------------
    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    const int colour = (x.x + x.y) & 1;
    for (int sweep = 0; sweep < kSweeps; ++sweep)
    {
        [unroll]
        for (int pass = 0; pass < 2; ++pass)
        {
            if (valid && colour == pass)
            {
                const float4 a = SharedA[li];
                float2 rhs = SharedB[li];
                float weights = 0.0f;
                [unroll]
                for (int k = 0; k < 4; ++k)
                {
                    const int2 t = int2(tid.xy) + offsets[k];
                    const int2 n = x + offsets[k];
                    if (any(t < 0) || any(t >= kTile) || any(n < 0) || any(n >= size)) continue;
                    const uint ni = (uint)(t.y * kTile + t.x);
                    const float w = 0.5f * (a.w + SharedA[ni].w);
                    rhs += w * (SharedU[ni] + SharedDu[ni] - SharedU[li]);
                    weights += w;
                }
                const float m11 = a.x + weights;
                const float m22 = a.z + weights;
                const float det = m11 * m22 - a.y * a.y;
                const float2 solved = det > 1e-12f
                    ? float2(m22 * rhs.x - a.y * rhs.y, m11 * rhs.y - a.y * rhs.x) / det
                    : float2(0.0f, 0.0f);
                SharedDu[li] = lerp(SharedDu[li], solved, kRefineOmega);
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // --- 3. the tile's interior -------------------------------------------
    if (valid && all(int2(tid.xy) >= kApron) && all(int2(tid.xy) < kApron + kInterior))
        Updated[x] = SharedU[li] + SharedDu[li];
}
