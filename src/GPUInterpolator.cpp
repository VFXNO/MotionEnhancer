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
    m_interpolateCS = m_context->compileComputeShader("InterpolateCS.hlsl");
    m_presentFrameCS = m_context->compileComputeShader("PresentFrameCS.hlsl");

    if (!m_luminanceCS || !m_pyramidCS || !m_blockMatchCS || !m_filterFlowCS ||
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
    invalidateFramePair();
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
        uint32_t levelW = std::max(1u, width >> l);
        uint32_t levelH = std::max(1u, height >> l);

        // R16_FLOAT, not R32_FLOAT: the D3D11 spec does not guarantee linear
        // filtering support for 32-bit float formats on FL11.0 hardware (many
        // GPUs silently fall back to point sampling), but PyramidCS.hlsl's
        // 9-tap trick depends on true bilinear blending between texels. Half
        // float has ample precision for 0-255 luma and is universally
        // guaranteed to support full linear filtering.
        m_pyr0[l].width = levelW;
        m_pyr0[l].height = levelH;
        if (!m_context->createTexture2D(levelW, levelH, DXGI_FORMAT_R16_FLOAT, m_pyr0[l].texture, m_pyr0[l].srv, m_pyr0[l].uav)) return false;

        m_pyr1[l].width = levelW;
        m_pyr1[l].height = levelH;
        if (!m_context->createTexture2D(levelW, levelH, DXGI_FORMAT_R16_FLOAT, m_pyr1[l].texture, m_pyr1[l].srv, m_pyr1[l].uav)) return false;

        m_fwdFlow[l].width = levelW;
        m_fwdFlow[l].height = levelH;
        uint32_t flowW = (levelW + 15) / 16;
        uint32_t flowH = (levelH + 15) / 16;
        if (!m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R32G32_FLOAT, m_fwdFlow[l].flowTexture, m_fwdFlow[l].flowSRV, m_fwdFlow[l].flowUAV) ||
            !m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R32G32_FLOAT, m_fwdFlow[l].upscaledTexture, m_fwdFlow[l].upscaledSRV, m_fwdFlow[l].upscaledUAV)) return false;

        m_bwdFlow[l].width = levelW;
        m_bwdFlow[l].height = levelH;
        if (!m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R32G32_FLOAT, m_bwdFlow[l].flowTexture, m_bwdFlow[l].flowSRV, m_bwdFlow[l].flowUAV) ||
            !m_context->createTexture2D(flowW, flowH, DXGI_FORMAT_R32G32_FLOAT, m_bwdFlow[l].upscaledTexture, m_bwdFlow[l].upscaledSRV, m_bwdFlow[l].upscaledUAV)) return false;
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

    D3D11_TEXTURE2D_DESC desc0 = {};
    D3D11_TEXTURE2D_DESC desc1 = {};
    frame0->GetDesc(&desc0);
    frame1->GetDesc(&desc1);
    if (desc0.Width != desc1.Width || desc0.Height != desc1.Height) {
        std::cerr << "Error: GPU frame pair dimensions do not match.\n";
        return false;
    }
    if (desc0.Width != m_width || desc0.Height != m_height) {
        if (!resizeBuffers(desc0.Width, desc0.Height)) return false;
    }

    if (frame0Index == m_cachedFrame0Index && frame1Index == m_cachedFrame1Index &&
        m_cachedFrame0SRV && m_cachedFrame1SRV) {
        return true;
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

    // Do not let the coarsest level shrink below ~24 px on its short side: a 16x16 block
    // there already spans hundreds of screen pixels, so retain the pyramid
    // for the coarsest search; parent vectors are propagated in the matcher.
    int totalLevels = m_totalLevels;
    while (totalLevels > 1 && std::min(m_width, m_height) >> (totalLevels - 1) < kMinCoarsestExtent) {
        --totalLevels;
    }

    ShaderConstants constants = {};
    constants.width = m_width;
    constants.height = m_height;
    constants.totalLevels = totalLevels;
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

    for (int l = 1; l < totalLevels; ++l) {
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
    // Step 3: Bidirectional coarse-to-fine block matching. Parent flow is
    // consumed directly by MotionSearchCS as the fine-level predictor.
    // =========================================================================
    int coarsestIdx = totalLevels - 1;
    auto estimateDirection = [&](FlowLevelResources* flow,
                                 PyramidLevelResources* reference,
                                 PyramidLevelResources* candidate) {
        auto dispatchFilter = [&](int level) {
            constants.levelWidth = flow[level].width;
            constants.levelHeight = flow[level].height;
            constants.levelIndex = level;
            m_context->updateConstants(constants);
            d3dContext->CSSetShader(m_filterFlowCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* inputs[] = {
                flow[level].upscaledSRV.Get(), reference[level].srv.Get(), candidate[level].srv.Get() };
            d3dContext->CSSetShaderResources(0, 3, inputs);
            ID3D11UnorderedAccessView* output[] = { flow[level].flowUAV.Get() };
            d3dContext->CSSetUnorderedAccessViews(0, 1, output, nullptr);
            uint32_t flowWidth = (flow[level].width + 15) / 16;
            uint32_t flowHeight = (flow[level].height + 15) / 16;
            d3dContext->Dispatch((flowWidth + 15) / 16, (flowHeight + 15) / 16, 1);
            d3dContext->CSSetShaderResources(0, 3, nullSRV);
            d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
        };

        for (int level = coarsestIdx; level >= 0; --level) {
            constants.levelWidth = flow[level].width;
            constants.levelHeight = flow[level].height;
            constants.levelIndex = level;
            constants.totalLevels = totalLevels;
            constants.blockSize = 16;
            constants.searchRadius = level == coarsestIdx ? m_coarseSearchRadius : m_refineSearchRadius;
            m_context->updateConstants(constants);

            d3dContext->CSSetShader(m_blockMatchCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* inputs[] = {
                reference[level].srv.Get(), candidate[level].srv.Get(),
                level == coarsestIdx ? nullptr : flow[level + 1].flowSRV.Get() };
            d3dContext->CSSetShaderResources(0, 3, inputs);
            ID3D11UnorderedAccessView* output[] = { flow[level].upscaledUAV.Get() };
            d3dContext->CSSetUnorderedAccessViews(0, 1, output, nullptr);
            d3dContext->Dispatch((flow[level].width + 15) / 16,
                                 (flow[level].height + 15) / 16, 1);
            d3dContext->CSSetShaderResources(0, 3, nullSRV);
            d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
            dispatchFilter(level);
        }

    };

    estimateDirection(m_fwdFlow, m_pyr0, m_pyr1);
    estimateDirection(m_bwdFlow, m_pyr1, m_pyr0);

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
