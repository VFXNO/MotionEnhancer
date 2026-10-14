#include "Common.hlsli"

// Median filter of one flow-ring field (FlowEngine::recordFlowMedian), once
// per pair, after the estimator (DIS, or FFX with its refinement) and the
// occlusion fill.
//
// Per-block refinement leaves isolated outliers where the picture does not
// decide - beside an object's outline, on flat regions - and on the 2 px
// grid the interpolation turned each one into a block of wrong pixels:
// "pixelated noise" next to a moving character with every refined variant,
// and none with FFX's raw 8x8 field (blurrier, but bilinear between blocks).
// A median over the neighbourhood removes a lone outlier and keeps an edge
// (a step survives a median; a spike does not) - the step Sun, Roth and
// Black ("Secrets of Optical Flow Estimation", CVPR 2010) found mattered
// most after the data and smoothness terms themselves.
//
// Each component is the median of its kMedianSize x kMedianSize
// neighbourhood. Input and output on the ring grid (xy = vector).
Texture2D<float4> Field : register(t0);
RWTexture2D<float2> Filtered : register(u0);

#ifndef MEDIAN_RADIUS
#define MEDIAN_RADIUS 1
#endif
static const int kRadius = MEDIAN_RADIUS;
static const int kCount = (2 * kRadius + 1) * (2 * kRadius + 1);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 gridMax = int2(grid) - 1;
    float xs[kCount];
    float ys[kCount];
    int n = 0;
    [unroll]
    for (int dy = -kRadius; dy <= kRadius; ++dy)
        [unroll]
        for (int dx = -kRadius; dx <= kRadius; ++dx)
        {
            const float2 v = Field.Load(int3(clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), gridMax), 0)).xy;
            xs[n] = v.x;
            ys[n] = v.y;
            ++n;
        }
    // Partial selection sort up to the middle element.
    const int middle = kCount / 2;
    [unroll]
    for (int i = 0; i <= middle; ++i)
    {
        [unroll]
        for (int j = i + 1; j < kCount; ++j)
        {
            const float ax = min(xs[i], xs[j]), bx = max(xs[i], xs[j]);
            xs[i] = ax; xs[j] = bx;
            const float ay = min(ys[i], ys[j]), by = max(ys[i], ys[j]);
            ys[i] = ay; ys[j] = by;
        }
    }
    Filtered[id.xy] = float2(xs[middle], ys[middle]);
}
