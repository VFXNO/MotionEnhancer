#include "GPUInterpolator.h"
#include <iostream>
#include <algorithm>

void GPUInterpolator::setSettings(const GPUInterpolationSettings& settings) {
    m_totalLevels = std::clamp(settings.pyramidLevels, 1, MAX_PYRAMID_LEVELS);
    m_minRefineLevel = std::clamp(settings.minRefineLevel, 0, m_totalLevels - 1);
    m_coarseSearchRadius = std::clamp(settings.coarseSearchRadius, 0, 8);
    m_refineSearchRadius = std::clamp(settings.refineSearchRadius, 0, 8);
    m_smoothnessWeight = std::clamp(settings.smoothnessWeight, 0.0f, 0.1f);
    invalidateFramePair();
}

bool GPUInterpolator::initialize(std::shared_ptr<D3D11Context> context) {
    m_context = context;
    if (!m_context || !m_context->device) return false;

    // Compile compute shaders
    m_luminanceCS   = m_context->compileComputeShader("LuminanceCS.hlsl");
    m_pyramidCS     = m_context->compileComputeShader("PyramidCS.hlsl");
    m_blockMatchCS  = m_context->compileComputeShader("MotionSearchCS.hlsl");
    m_filterFlowCS  = m_context->compileComputeShader("FilterFlowCS.hlsl");
    m_upscaleFlowCS = m_context->compileComputeShader("UpscaleFlowCS.hlsl");
    m_invertFlowCS  = m_context->compileComputeShader("InvertFlowCS.hlsl");
    m_interpolateCS = m_context->compileComputeShader("InterpolateCS.hlsl");
    m_presentFrameCS = m_context->compileComputeShader("PresentFrameCS.hlsl");

    if (!m_luminanceCS || !m_pyramidCS || !m_blockMatchCS || !m_filterFlowCS || !m_upscaleFlowCS || !m_invertFlowCS ||
        !m_interpolateCS || !m_presentFrameCS) {
        std::cerr << "Error: Failed to compile one or more GPU compute shaders.\n";
        return false;
    }

    D3D11_QUERY_DESC queryDesc = {};
    for (auto& timing : m_gpuTiming) {
        queryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        if (FAILED(m_context->device->CreateQuery(&queryDesc, timing.disjoint.GetAddressOf()))) break;
        queryDesc.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(m_context->device->CreateQuery(&queryDesc, timing.start.GetAddressOf()))) break;
        if (FAILED(m_context->device->CreateQuery(&queryDesc, timing.end.GetAddressOf()))) break;
    }

    return true;
}

bool GPUInterpolator::presentSourceFrame(
    ID3D11Texture2D* frame,
    ID3D11Texture2D* outputTarget
) {
    if (!m_context || !frame || !outputTarget || !m_presentFrameCS || !m_outputUAV) {
        return false;
    }

    ComPtr<ID3D11ShaderResourceView> frameSRV;
    HRESULT hr = m_context->device->CreateShaderResourceView(
        frame, nullptr, frameSRV.GetAddressOf());
    if (FAILED(hr)) return false;

    ShaderConstants constants = {};
    constants.width = m_width;
    constants.height = m_height;
    constants.levelWidth = m_width;
    constants.levelHeight = m_height;
    m_context->updateConstants(constants);

    auto context = m_context->context.Get();
    ID3D11ShaderResourceView* srvs[] = { frameSRV.Get() };
    ID3D11UnorderedAccessView* uavs[] = { m_outputUAV.Get() };
    context->CSSetShader(m_presentFrameCS.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, srvs);
    context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    context->Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);

    ID3D11ShaderResourceView* nullSRV[] = { nullptr };
    ID3D11UnorderedAccessView* nullUAV[] = { nullptr };
    context->CSSetShaderResources(0, 1, nullSRV);
    context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    context->CopyResource(outputTarget, m_outputTexture.Get());
    return true;
}

bool GPUInterpolator::resizeBuffers(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return true;
    m_width = width;
    m_height = height;

    // Allocate internal full-resolution output texture
    if (!m_context->createTexture2D(
        width, height,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        m_outputTexture,
        m_outputSRV,
        m_outputUAV,
        true // isRenderTarget
    )) {
        std::cerr << "Error: Failed to create GPU interpolation output texture ("
                  << width << "x" << height << ").\n";
        m_width = 0;
        m_height = 0;
        return false;
    }

    for (int l = 0; l < MAX_PYRAMID_LEVELS; ++l) {
        uint32_t levelW = std::max(2u, width >> l);
        uint32_t levelH = std::max(2u, height >> l);

        m_pyr0[l].width = levelW;
        m_pyr0[l].height = levelH;
        m_context->createTexture2D(levelW, levelH, DXGI_FORMAT_R32_FLOAT, m_pyr0[l].texture, m_pyr0[l].srv, m_pyr0[l].uav);

        m_pyr1[l].width = levelW;
        m_pyr1[l].height = levelH;
        m_context->createTexture2D(levelW, levelH, DXGI_FORMAT_R32_FLOAT, m_pyr1[l].texture, m_pyr1[l].srv, m_pyr1[l].uav);

        m_fwdFlow[l].width = levelW;
        m_fwdFlow[l].height = levelH;
        uint32_t flowW = (levelW + 7) / 8;
        uint32_t flowH = (levelH + 7) / 8;
        m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R16G16_SINT, m_fwdFlow[l].flowTexture, m_fwdFlow[l].flowSRV, m_fwdFlow[l].flowUAV);
        m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R16G16_SINT, m_fwdFlow[l].upscaledTexture, m_fwdFlow[l].upscaledSRV, m_fwdFlow[l].upscaledUAV);

        if (l == 0) {
            m_bwdFlow[l].width = levelW;
            m_bwdFlow[l].height = levelH;
            m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R16G16_SINT, m_bwdFlow[l].flowTexture, m_bwdFlow[l].flowSRV, m_bwdFlow[l].flowUAV);
        }
    }

    return true;
}

bool GPUInterpolator::prepareFramePair(
    ID3D11Texture2D* frame0,
    ID3D11Texture2D* frame1,
    uint64_t frame0Index,
    uint64_t frame1Index
) {
    if (!m_context || !frame0 || !frame1) return false;
    if (frame0Index == m_cachedFrame0Index && frame1Index == m_cachedFrame1Index) {
        return true;
    }

    D3D11_TEXTURE2D_DESC desc;
    frame0->GetDesc(&desc);
    if (desc.Width != m_width || desc.Height != m_height) {
        if (!resizeBuffers(desc.Width, desc.Height)) return false;
    }

    auto d3dContext = m_context->context.Get();

    int timingSlot = -1;
    for (size_t i = 0; i < m_gpuTiming.size(); ++i) {
        auto& timing = m_gpuTiming[i];
        if (timing.pending) {
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint = {};
            UINT64 startTimestamp = 0;
            UINT64 endTimestamp = 0;
            if (d3dContext->GetData(timing.disjoint.Get(), &disjoint, sizeof(disjoint), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK &&
                d3dContext->GetData(timing.start.Get(), &startTimestamp, sizeof(startTimestamp), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK &&
                d3dContext->GetData(timing.end.Get(), &endTimestamp, sizeof(endTimestamp), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK) {
                if (!disjoint.Disjoint && disjoint.Frequency > 0 && endTimestamp >= startTimestamp) {
                    m_lastGpuTimeMs = 1000.0 * static_cast<double>(endTimestamp - startTimestamp) /
                                      static_cast<double>(disjoint.Frequency);
                }
                timing.pending = false;
            }
        }
        if (timingSlot < 0 && !timing.pending && timing.disjoint && timing.start && timing.end) {
            timingSlot = static_cast<int>(i);
        }
    }

    if (timingSlot >= 0) {
        auto& timing = m_gpuTiming[static_cast<size_t>(timingSlot)];
        d3dContext->Begin(timing.disjoint.Get());
        d3dContext->End(timing.start.Get());
    }

    m_cachedFrame0SRV.Reset();
    m_cachedFrame1SRV.Reset();
    if (FAILED(m_context->device->CreateShaderResourceView(
            frame0, nullptr, m_cachedFrame0SRV.GetAddressOf())) ||
        FAILED(m_context->device->CreateShaderResourceView(
            frame1, nullptr, m_cachedFrame1SRV.GetAddressOf()))) {
        return false;
    }

    ShaderConstants constants = {};
    constants.width = m_width;
    constants.height = m_height;
    constants.totalLevels = m_totalLevels;
    constants.smoothnessWeight = m_smoothnessWeight;

    // =========================================================================
    // Step 1: Convert Frame 0 and Frame 1 to Luminance (Level 0)
    // =========================================================================
    constants.levelWidth = m_width;
    constants.levelHeight = m_height;
    m_context->updateConstants(constants);

    d3dContext->CSSetShader(m_luminanceCS.Get(), nullptr, 0);

    // Frame 0 -> Luminance
    ID3D11ShaderResourceView* srvList0[] = { m_cachedFrame0SRV.Get() };
    ID3D11UnorderedAccessView* uavList0[] = { m_pyr0[0].uav.Get() };
    d3dContext->CSSetShaderResources(0, 1, srvList0);
    d3dContext->CSSetUnorderedAccessViews(0, 1, uavList0, nullptr);
    d3dContext->Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);

    // Frame 1 -> Luminance
    ID3D11ShaderResourceView* nullSRV[] = { nullptr, nullptr, nullptr, nullptr };
    ID3D11UnorderedAccessView* nullUAV[] = { nullptr, nullptr };
    d3dContext->CSSetShaderResources(0, 1, nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

    ID3D11ShaderResourceView* srvList1[] = { m_cachedFrame1SRV.Get() };
    ID3D11UnorderedAccessView* uavList1[] = { m_pyr1[0].uav.Get() };
    d3dContext->CSSetShaderResources(0, 1, srvList1);
    d3dContext->CSSetUnorderedAccessViews(0, 1, uavList1, nullptr);
    d3dContext->Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);

    d3dContext->CSSetShaderResources(0, 1, nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

    // =========================================================================
    // Step 2: Build 8-Level Image Pyramids (Levels 1 to 7)
    // =========================================================================
    d3dContext->CSSetShader(m_pyramidCS.Get(), nullptr, 0);

    for (int l = 1; l < m_totalLevels; ++l) {
        constants.levelWidth = m_pyr0[l].width;
        constants.levelHeight = m_pyr0[l].height;
        constants.levelIndex = l;
        m_context->updateConstants(constants);

        // Pyr0 level l-1 -> level l
        ID3D11ShaderResourceView* pyr0Src[] = { m_pyr0[l - 1].srv.Get() };
        ID3D11UnorderedAccessView* pyr0Dst[] = { m_pyr0[l].uav.Get() };
        d3dContext->CSSetShaderResources(0, 1, pyr0Src);
        d3dContext->CSSetUnorderedAccessViews(0, 1, pyr0Dst, nullptr);
        d3dContext->Dispatch((m_pyr0[l].width + 15) / 16, (m_pyr0[l].height + 15) / 16, 1);

        d3dContext->CSSetShaderResources(0, 1, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

        // Pyr1 level l-1 -> level l
        ID3D11ShaderResourceView* pyr1Src[] = { m_pyr1[l - 1].srv.Get() };
        ID3D11UnorderedAccessView* pyr1Dst[] = { m_pyr1[l].uav.Get() };
        d3dContext->CSSetShaderResources(0, 1, pyr1Src);
        d3dContext->CSSetUnorderedAccessViews(0, 1, pyr1Dst, nullptr);
        d3dContext->Dispatch((m_pyr1[l].width + 15) / 16, (m_pyr1[l].height + 15) / 16, 1);

        d3dContext->CSSetShaderResources(0, 1, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    }

    // =========================================================================
    // Step 3: Hierarchical Coarse-to-Fine Census/SAD Block Matching
    // =========================================================================
    int coarsestIdx = m_totalLevels - 1;

    // Coarsest Level 7
    {
        constants.levelWidth = m_pyr0[coarsestIdx].width;
        constants.levelHeight = m_pyr0[coarsestIdx].height;
        constants.levelIndex = coarsestIdx;
        constants.blockSize = 8;
        constants.searchRadius = m_coarseSearchRadius;
        m_context->updateConstants(constants);

        d3dContext->CSSetShader(m_blockMatchCS.Get(), nullptr, 0);

        // Forward flow Level 7
        ID3D11ShaderResourceView* fwdSRV[] = { m_pyr0[coarsestIdx].srv.Get(), m_pyr1[coarsestIdx].srv.Get(), nullptr };
        ID3D11UnorderedAccessView* fwdUAV[] = { m_fwdFlow[coarsestIdx].upscaledUAV.Get() };
        d3dContext->CSSetShaderResources(0, 3, fwdSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, fwdUAV, nullptr);
        uint32_t flowWidth = (m_pyr0[coarsestIdx].width + 7) / 8;
        uint32_t flowHeight = (m_pyr0[coarsestIdx].height + 7) / 8;
        d3dContext->Dispatch((flowWidth + 1) / 2, (flowHeight + 1) / 2, 1);

        d3dContext->CSSetShaderResources(0, 3, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

        d3dContext->CSSetShader(m_filterFlowCS.Get(), nullptr, 0);
        ID3D11ShaderResourceView* filterSRV[] = { m_fwdFlow[coarsestIdx].upscaledSRV.Get() };
        ID3D11UnorderedAccessView* filterUAV[] = { m_fwdFlow[coarsestIdx].flowUAV.Get() };
        d3dContext->CSSetShaderResources(0, 1, filterSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, filterUAV, nullptr);
        d3dContext->Dispatch((flowWidth + 15) / 16, (flowHeight + 15) / 16, 1);
        d3dContext->CSSetShaderResources(0, 1, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    }

    // Refine down to the user-selected minimum level.
    for (int l = coarsestIdx - 1; l >= m_minRefineLevel; --l) {
        // 3a. Bilinearly upscale flow from level l+1 to level l
        constants.levelWidth = m_fwdFlow[l].width;
        constants.levelHeight = m_fwdFlow[l].height;
        constants.levelIndex = l;
        m_context->updateConstants(constants);

        d3dContext->CSSetShader(m_upscaleFlowCS.Get(), nullptr, 0);

        // Upscale Forward Flow
        ID3D11ShaderResourceView* upFwdSrc[] = {
            m_fwdFlow[l + 1].flowSRV.Get(), m_pyr0[l].srv.Get(), m_pyr1[l].srv.Get()
        };
        ID3D11UnorderedAccessView* upFwdDst[] = { m_fwdFlow[l].flowUAV.Get() };
        d3dContext->CSSetShaderResources(0, 3, upFwdSrc);
        d3dContext->CSSetUnorderedAccessViews(0, 1, upFwdDst, nullptr);
        uint32_t flowWidth = (m_fwdFlow[l].width + 7) / 8;
        uint32_t flowHeight = (m_fwdFlow[l].height + 7) / 8;
        d3dContext->Dispatch((flowWidth + 15) / 16, (flowHeight + 15) / 16, 1);

        d3dContext->CSSetShaderResources(0, 3, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

        // 3b. Refine Flow at Level l
        constants.blockSize = 4;
        constants.searchRadius = m_refineSearchRadius;
        m_context->updateConstants(constants);

        d3dContext->CSSetShader(m_blockMatchCS.Get(), nullptr, 0);

        // Refine Forward
        ID3D11ShaderResourceView* refFwdSRV[] = { m_pyr0[l].srv.Get(), m_pyr1[l].srv.Get(), m_fwdFlow[l].flowSRV.Get() };
        ID3D11UnorderedAccessView* refFwdUAV[] = { m_fwdFlow[l].upscaledUAV.Get() };
        d3dContext->CSSetShaderResources(0, 3, refFwdSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, refFwdUAV, nullptr);
        d3dContext->Dispatch((flowWidth + 1) / 2, (flowHeight + 1) / 2, 1);

        d3dContext->CSSetShaderResources(0, 3, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

        d3dContext->CSSetShader(m_filterFlowCS.Get(), nullptr, 0);
        ID3D11ShaderResourceView* filterSRV[] = { m_fwdFlow[l].upscaledSRV.Get() };
        ID3D11UnorderedAccessView* filterUAV[] = { m_fwdFlow[l].flowUAV.Get() };
        d3dContext->CSSetShaderResources(0, 1, filterSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, filterUAV, nullptr);
        d3dContext->Dispatch((flowWidth + 15) / 16, (flowHeight + 15) / 16, 1);
        d3dContext->CSSetShaderResources(0, 1, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

    }

    // Upscale the refined flow to full native resolution.
    for (int l = m_minRefineLevel - 1; l >= 0; --l) {
        constants.levelWidth = m_fwdFlow[l].width;
        constants.levelHeight = m_fwdFlow[l].height;
        constants.levelIndex = l;
        m_context->updateConstants(constants);

        d3dContext->CSSetShader(m_upscaleFlowCS.Get(), nullptr, 0);

        ID3D11ShaderResourceView* upFwdSrc[] = {
            m_fwdFlow[l + 1].flowSRV.Get(), m_pyr0[l].srv.Get(), m_pyr1[l].srv.Get()
        };
        ID3D11UnorderedAccessView* upFwdDst[] = { m_fwdFlow[l].flowUAV.Get() };
        d3dContext->CSSetShaderResources(0, 3, upFwdSrc);
        d3dContext->CSSetUnorderedAccessViews(0, 1, upFwdDst, nullptr);
        uint32_t flowWidth = (m_fwdFlow[l].width + 7) / 8;
        uint32_t flowHeight = (m_fwdFlow[l].height + 7) / 8;
        d3dContext->Dispatch((flowWidth + 15) / 16, (flowHeight + 15) / 16, 1);
        d3dContext->CSSetShaderResources(0, 3, nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

    }

    // Approximate backward flow from the completed forward field.
    constants.levelWidth = m_width;
    constants.levelHeight = m_height;
    m_context->updateConstants(constants);
    d3dContext->CSSetShader(m_invertFlowCS.Get(), nullptr, 0);
    ID3D11ShaderResourceView* invertSRV[] = { m_fwdFlow[0].flowSRV.Get() };
    ID3D11UnorderedAccessView* invertUAV[] = { m_bwdFlow[0].flowUAV.Get() };
    d3dContext->CSSetShaderResources(0, 1, invertSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, invertUAV, nullptr);
    d3dContext->Dispatch(((m_width + 7) / 8 + 15) / 16, ((m_height + 7) / 8 + 15) / 16, 1);
    d3dContext->CSSetShaderResources(0, 1, nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

    if (timingSlot >= 0) {
        auto& timing = m_gpuTiming[static_cast<size_t>(timingSlot)];
        d3dContext->End(timing.end.Get());
        d3dContext->End(timing.disjoint.Get());
        timing.pending = true;
    }
    m_cachedFrame0Index = frame0Index;
    m_cachedFrame1Index = frame1Index;
    return true;
}

bool GPUInterpolator::synthesize(ID3D11Texture2D* outputTarget, float timeT) {
    if (!m_context || !outputTarget || !m_cachedFrame0SRV || !m_cachedFrame1SRV) {
        return false;
    }

    ShaderConstants constants = {};
    constants.width = m_width;
    constants.height = m_height;
    constants.levelWidth = m_width;
    constants.levelHeight = m_height;
    constants.timeT = std::clamp(timeT, 0.0f, 1.0f);
    m_context->updateConstants(constants);

    auto context = m_context->context.Get();
    ID3D11ShaderResourceView* resources[] = {
        m_cachedFrame0SRV.Get(), m_cachedFrame1SRV.Get(),
        m_fwdFlow[0].flowSRV.Get(), m_bwdFlow[0].flowSRV.Get()
    };
    ID3D11UnorderedAccessView* output[] = { m_outputUAV.Get() };
    context->CSSetShader(m_interpolateCS.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 4, resources);
    context->CSSetUnorderedAccessViews(0, 1, output, nullptr);
    context->Dispatch((m_width + 15) / 16, (m_height + 15) / 16, 1);

    ID3D11ShaderResourceView* nullSRV[] = { nullptr, nullptr, nullptr, nullptr };
    ID3D11UnorderedAccessView* nullUAV[] = { nullptr };
    context->CSSetShaderResources(0, 4, nullSRV);
    context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    context->CopyResource(outputTarget, m_outputTexture.Get());
    return true;
}

void GPUInterpolator::invalidateFramePair() {
    m_cachedFrame0SRV.Reset();
    m_cachedFrame1SRV.Reset();
    m_cachedFrame0Index = UINT64_MAX;
    m_cachedFrame1Index = UINT64_MAX;
}
