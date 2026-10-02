#include "Common.hlsli"

// Interpolation consumes the estimator's forward and backward vector fields
// directly. No per-pixel matching, vector reselection, or refining pass.
//
// Field convention (FlowScale selects it; see FlowEngine::Entry::singleField):
//   FlowScale == 0  two independently estimated fields. ForwardFlow lives on
//                   frame0's grid (frame0 -> frame1); BackwardFlow lives on
//                   frame1's grid (frame1 -> frame0). Each needs its own
//                   trajectory solve. This is what the SAD and neural graphs
//                   produce - two separate searches with the roles swapped.
//   FlowScale != 0  one field, negated. FFX estimates current -> previous
//                   only, so "forward" is exactly -backward and both are
//                   indexed on frame1's grid. There is a single trajectory,
//                   and solving the frame0-grid fixed point for a field that
//                   is not on frame0's grid is simply the wrong equation -
//                   it cost ~1.3 dB on a pure global shift before this split.
Texture2D<float4> Frame0 : register(t0);
Texture2D<float4> Frame1 : register(t1);
Texture2D<float4> ForwardFlow : register(t2);
Texture2D<float4> BackwardFlow : register(t3);
RWTexture2D<float4> Output : register(u0);

float2 SampleFlow(Texture2D<float4> flow, float2 pixel)
{
    const int2 gridMax = int2(FlowGridSize(Width, Height, BlockSize)) - 1;
    const float2 position = (pixel - (BlockSize - 1) * 0.5f) / BlockSize;
    const int2 base = int2(floor(position));
    const float2 fraction = frac(position);
    const float2 v00 = flow.Load(int3(clamp(base, int2(0, 0), gridMax), 0)).xy;
    const float2 v10 = flow.Load(int3(clamp(base + int2(1, 0), int2(0, 0), gridMax), 0)).xy;
    const float2 v01 = flow.Load(int3(clamp(base + int2(0, 1), int2(0, 0), gridMax), 0)).xy;
    const float2 v11 = flow.Load(int3(clamp(base + int2(1, 1), int2(0, 0), gridMax), 0)).xy;
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

// Flow of the scene point that is at `pixel` at time t, for a field defined
// on that field's own frame grid.
//
// ForwardFlow lives on frame0's grid: F(p) displaces p from frame0 to
// frame1, so the point visible at `pixel` at time t is the p with
// p + t*F(p) = pixel, i.e. p = pixel - t*F(p) - a fixed point, not a direct
// lookup. Reading F(pixel) and stopping there (what this pass used to do)
// asks the field for the motion at the point's *destination* instead of at
// the point: on a 16 px motion at t = 0.5 that samples 8 px away, so the
// vector an object's own block voted for gets applied to the background
// beside it and vice versa. That mismatch is what tears object edges.
//
// Two Picard iterations converge wherever the field is Lipschitz below
// 1/t, which covers everything except the boundary blocks themselves, and
// each one costs four loads of a (Width/8 x Height/8) texture that is
// resident in cache. Nothing here rejects, blends or smooths a vector: the
// estimate is only read at the right place.
float2 TrackFlow(Texture2D<float4> flow, float2 pixel, float scale)
{
    float2 v = SampleFlow(flow, pixel);
    v = SampleFlow(flow, pixel - scale * v);
    v = SampleFlow(flow, pixel - scale * v);
    return v;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    const float t = saturate(TimeT);
    const float2 pixel = float2(id.xy);
    const float2 inverseSize = rcp(float2(Width, Height));
    float2 from0, from1;
    if (FlowScale != 0.0f)
    {
        // One trajectory: q = pixel - (1-t)*B(q) is the point's position in
        // frame1, and its frame0 position is q + B(q) = pixel + t*B(q).
        const float2 backward = TrackFlow(BackwardFlow, pixel, 1.0f - t);
        from1 = pixel - (1.0f - t) * backward;
        from0 = pixel + t * backward;
    }
    else
    {
        from0 = pixel - t * TrackFlow(ForwardFlow, pixel, t);
        from1 = pixel - (1.0f - t) * TrackFlow(BackwardFlow, pixel, 1.0f - t);
    }
    const float3 c0 = Frame0.SampleLevel(LinearClamp, (from0 + 0.5f) * inverseSize, 0).rgb;
    const float3 c1 = Frame1.SampleLevel(LinearClamp, (from1 + 0.5f) * inverseSize, 0).rgb;
    Output[id.xy] = float4(lerp(c0, c1, t), 1.0f);
}
