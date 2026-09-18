#pragma once

#include "D3D11Context.h"
#include <array>
#include <memory>
#include <vector>

struct PyramidLevelResources {
    uint32_t width = 0;
    uint32_t height = 0;
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
};

struct FlowLevelResources {
    uint32_t width = 0;
    uint32_t height = 0;
    ComPtr<ID3D11Texture2D> flowTexture;
    ComPtr<ID3D11ShaderResourceView> flowSRV;
    ComPtr<ID3D11UnorderedAccessView> flowUAV;

    ComPtr<ID3D11Texture2D> upscaledTexture;
    ComPtr<ID3D11ShaderResourceView> upscaledSRV;
    ComPtr<ID3D11UnorderedAccessView> upscaledUAV;

};

struct GPUInterpolationSettings {
    int pyramidLevels = 7;
    int minRefineLevel = 0;
    int coarseSearchRadius = 8;
    // Per-level correction around the coarse predictor. The coarsest level
    // (+-8 at 1/32 scale) already reaches large motions; each extra pixel of
    // refine radius mostly adds chance matches on smooth content, and every
    // wrong vector makes the two warps disagree so that region falls back to
    // source cadence (seen as stutter). 2 was confirmed smoother than 4 on
    // real anime content. Objects too small for the coarse levels to see
    // need 4 (--gpu-refine-radius).
    int refineSearchRadius = 2;
    // Same scale as ZNCCMatcher: cost = zncc - w * (du^2 + dv^2).
    float smoothnessWeight = 0.0005f;
};

class GPUInterpolator {
public:
    static const int MAX_PYRAMID_LEVELS = 8;
    // Levels whose short side would drop below this are not searched; see
    // prepareFramePair. 1080p therefore uses at most 6 levels (60x33 coarsest).
    static const uint32_t kMinCoarsestExtent = 24;

    GPUInterpolator() = default;
    ~GPUInterpolator() = default;

    bool initialize(std::shared_ptr<D3D11Context> context);
    bool resizeBuffers(uint32_t width, uint32_t height);
    void setSettings(const GPUInterpolationSettings& settings);

    bool prepareFramePair(
        ID3D11Texture2D* frame0,
        ID3D11Texture2D* frame1,
        uint64_t frame0Index,
        uint64_t frame1Index
    );
    bool synthesize(ID3D11Texture2D* outputTarget, float timeT);
    bool presentSourceFrame(ID3D11Texture2D* frame, ID3D11Texture2D* outputTarget);
    void invalidateFramePair();

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    double getLastGpuTimeMs() const { return m_lastGpuTimeMs; }

    // Full-resolution flow grids (R32G32_FLOAT, one float2 per 16x16 block) of the
    // last prepared pair; for offline debugging / readback.
    ID3D11Texture2D* forwardFlowTexture() const { return m_fwdFlow[0].flowTexture.Get(); }
    ID3D11Texture2D* backwardFlowTexture() const { return m_bwdFlow[0].flowTexture.Get(); }

private:
    std::shared_ptr<D3D11Context> m_context;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    int m_totalLevels = 7;
    int m_minRefineLevel = 0;
    int m_coarseSearchRadius = 8;
    int m_refineSearchRadius = 2;
    float m_smoothnessWeight = 0.0005f;

    // Compute Shaders
    ComPtr<ID3D11ComputeShader> m_luminanceCS;
    ComPtr<ID3D11ComputeShader> m_pyramidCS;
    ComPtr<ID3D11ComputeShader> m_blockMatchCS;
    ComPtr<ID3D11ComputeShader> m_filterFlowCS;
    ComPtr<ID3D11ComputeShader> m_interpolateCS;
    ComPtr<ID3D11ComputeShader> m_presentFrameCS;

    // Pyramid Levels (0 to 7)
    PyramidLevelResources m_pyr0[MAX_PYRAMID_LEVELS];
    PyramidLevelResources m_pyr1[MAX_PYRAMID_LEVELS];

    // Flow Fields (0 to 7)
    FlowLevelResources m_fwdFlow[MAX_PYRAMID_LEVELS];
    FlowLevelResources m_bwdFlow[MAX_PYRAMID_LEVELS];

    // Final synthesis output texture and views
    ComPtr<ID3D11Texture2D> m_outputTexture;
    ComPtr<ID3D11ShaderResourceView> m_outputSRV;
    ComPtr<ID3D11UnorderedAccessView> m_outputUAV;

    struct GpuTimingSlot {
        ComPtr<ID3D11Query> disjoint;
        ComPtr<ID3D11Query> start;
        ComPtr<ID3D11Query> end;
        bool pending = false;
    };
    std::array<GpuTimingSlot, 4> m_gpuTiming;
    double m_lastGpuTimeMs = 0.0;
    ComPtr<ID3D11ShaderResourceView> m_cachedFrame0SRV;
    ComPtr<ID3D11ShaderResourceView> m_cachedFrame1SRV;
    uint64_t m_cachedFrame0Index = UINT64_MAX;
    uint64_t m_cachedFrame1Index = UINT64_MAX;
};
