#include "Common.hlsli"

Texture2D<float2> CurrentFlow  : register(t0);
Texture2D<float2> OppositeFlow : register(t1);
RWTexture2D<float2> OutputFlow : register(u0);

static const float kErrorThreshold = 4.0f;
static const float kImprovement = 1.0f;

float2 SampleFlowAtPixel(Texture2D<float2> flow, float2 pixel)
{
    uint flowWidth, flowHeight;
    flow.GetDimensions(flowWidth, flowHeight);
    int2 maxCoord = int2(flowWidth, flowHeight) - 1;
    float2 gridPos = (pixel - (kFlowBlock - 1) * 0.5f) / kFlowBlock;
    int2 base = int2(floor(gridPos));
    float2 f = gridPos - float2(base);
    int2 c00 = clamp(base, int2(0, 0), maxCoord);
    int2 c10 = clamp(base + int2(1, 0), int2(0, 0), maxCoord);
    int2 c01 = clamp(base + int2(0, 1), int2(0, 0), maxCoord);
    int2 c11 = clamp(base + int2(1, 1), int2(0, 0), maxCoord);
    float2 a = lerp(flow.Load(int3(c00, 0)), flow.Load(int3(c10, 0)), f.x);
    float2 b = lerp(flow.Load(int3(c01, 0)), flow.Load(int3(c11, 0)), f.x);
    return lerp(a, b, f.y);
}

float ConsistencyError(Texture2D<float2> opposite, float2 vector, float2 center)
{
    float2 reverse = SampleFlowAtPixel(opposite, center + vector);
    return length(vector + reverse);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint2 grid = FlowGridSize(LevelWidth, LevelHeight);
    if (id.x >= grid.x || id.y >= grid.y) return;

    float2 own = CurrentFlow.Load(int3(id.xy, 0));
    float2 center = float2(id.xy * kFlowBlock) + (kFlowBlock - 1) * 0.5f;
    float ownError = ConsistencyError(OppositeFlow, own, center);
    float bestError = ownError;
    float2 best = own;
    int2 gridMax = int2(grid) - 1;

    [unroll]
    for (int i = 0; i < 9; ++i)
    {
        if (i == 4) continue;
        int2 coord = clamp(int2(id.xy) + int2(i % 3 - 1, i / 3 - 1), int2(0, 0), gridMax);
        float2 candidate = CurrentFlow.Load(int3(coord, 0));
        float error = ConsistencyError(OppositeFlow, candidate, center);
        if (error < bestError)
        {
            bestError = error;
            best = candidate;
        }
    }

    OutputFlow[id.xy] = (ownError > kErrorThreshold &&
                         ownError - bestError > kImprovement) ? best : own;
}
