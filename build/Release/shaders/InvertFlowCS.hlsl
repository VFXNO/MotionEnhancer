#include "Common.hlsli"

// Forward flow (frame0 -> frame1, indexed by frame0 block) -> backward flow
// (frame1 -> frame0, indexed by frame1 block), full-resolution grid.
//
// For frame1 block q we want the frame0 block p whose forward vector lands
// on q: centre(p) + F(p) ~= centre(q); then B(q) = -F(p). A fixed-point
// iteration (p = q - F(p)) converges for smooth flow; its 3x3 neighbourhood
// is then scored by landing error so the best actual source is picked
// rather than trusting the iteration blindly. Blocks nothing lands on
// (disocclusions) fall back to the negated local forward vector, which is
// what the pixel shader would have assumed anyway.
Texture2D<int2> ForwardFlow : register(t0);
RWTexture2D<int2> BackwardFlow : register(u0);

static const int kIterations = 4;
// Landing error (pixels) beyond which no source block maps here.
static const float kMaxLandingError = 8.0f;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(Width, Height);
    if (id.x >= grid.x || id.y >= grid.y) return;
    int2 gridMax = int2(grid) - 1;

    float2 target = float2(id.xy) * kFlowBlock + (kFlowBlock - 1) * 0.5f;

    // Fixed-point search for the source block.
    int2 source = int2(id.xy);
    [unroll]
    for (int i = 0; i < kIterations; ++i)
    {
        float2 f = float2(ForwardFlow.Load(int3(source, 0)));
        source = clamp(int2(floor((target - f) / kFlowBlock)), int2(0, 0), gridMax);
    }

    // Verify against the 3x3 neighbourhood of the estimate.
    float bestError = 1e30f;
    int2 bestVector = int2(0, 0);
    [unroll]
    for (int n = 0; n < 9; ++n)
    {
        int2 p = clamp(source + int2(n % 3 - 1, n / 3 - 1), int2(0, 0), gridMax);
        int2 f = ForwardFlow.Load(int3(p, 0));
        float2 landing = float2(p) * kFlowBlock + (kFlowBlock - 1) * 0.5f + float2(f);
        float error = length(landing - target);
        if (error < bestError)
        {
            bestError = error;
            bestVector = -f;
        }
    }

    if (bestError > kMaxLandingError)
    {
        bestVector = -ForwardFlow.Load(int3(id.xy, 0));
    }
    BackwardFlow[id.xy] = bestVector;
}
