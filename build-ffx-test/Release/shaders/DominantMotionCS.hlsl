#include "Synthesis.hlsli"

// The camera's motion for one flow field: the mean vector of the blocks
// along the frame border (two blocks deep) whose motion is confirmed by the
// other field. The border is where the background almost always is - a
// character the camera orbits, a target in the crosshair, the viewmodel all
// sit inside the frame - so this is the motion an occluded background is
// filled with (OcclusionFillCS). One 256-thread group; writes texel
// (LevelIndex, 0) of the pair's 2x1 R32G32_FLOAT camera texture - 0 for the
// forward field, 1 for the backward. Constants: the interpolation constants
// of the pair (block size, field convention) and LevelIndex.
Texture2D<float4> Field : register(t0);    // the field being described
Texture2D<float4> Other : register(t1);    // the opposite direction's field
RWTexture2D<float2> Dominant : register(u0);

static const uint kThreads = 256;
static const int kBorderBlocks = 2;
static const float kConsistent = 2.0f;      // round-trip pixels
groupshared float3 Partial[kThreads];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_GroupThreadID)
{
    const uint2 grid = FlowGridSize(Width, Height, BlockSize);
    const uint ringBlocks = grid.x * kBorderBlocks * 2 + (grid.y - 2 * kBorderBlocks) * kBorderBlocks * 2;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (uint i = tid.x; i < ringBlocks; i += kThreads)
    {
        // Enumerate the border ring: top rows, bottom rows, then the left
        // and right columns between them.
        uint2 b;
        const uint rows = grid.x * kBorderBlocks;
        if (i < rows) b = uint2(i % grid.x, i / grid.x);
        else if (i < 2 * rows) b = uint2((i - rows) % grid.x, grid.y - 1 - (i - rows) / grid.x);
        else
        {
            const uint j = i - 2 * rows;
            const uint column = j % (kBorderBlocks * 2);
            b = uint2(column < kBorderBlocks ? column : grid.x - 1 - (column - kBorderBlocks),
                      kBorderBlocks + j / (kBorderBlocks * 2));
        }
        const float2 v = Field.Load(int3(b, 0)).xy;
        const float2 centre = float2(b) * BlockSize + 0.5f * (BlockSize - 1);
        const float2 back = SampleFlow(Other, centre + v);
        if (length(v + back) < kConsistent) sum += float3(v, 1.0f);
    }
    Partial[tid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (tid.x < stride) Partial[tid.x] += Partial[tid.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0)
        Dominant[uint2(LevelIndex, 0)] = Partial[0].z > 0.0f ? Partial[0].xy / Partial[0].z : float2(0.0f, 0.0f);
}
