cbuffer PipelineConstants : register(b0)
{
    uint Width;
    uint Height;
    uint LevelWidth;
    uint LevelHeight;

    int SearchRadius;
    int BlockSize;
    int LevelIndex;
    int TotalLevels;

    float TimeT;
    float SmoothnessWeight;
    uint Pad0;
    uint Pad1;
};

SamplerState LinearClamp : register(s0);
SamplerState PointClamp : register(s1);

float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}
