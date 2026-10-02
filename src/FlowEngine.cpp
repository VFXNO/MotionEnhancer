#include "FlowEngine.h"

#include <algorithm>
#include <iostream>

namespace {
// Flow is four floats per block: xy = vector, zw = the per-axis confidence
// FillFlowCS needs to tell a measured component from an inherited one.
constexpr DXGI_FORMAT kFlowFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
constexpr UINT kFlowTexelBytes = 16;

// One place that copies the model into the constant buffer, for every pass
// that needs it. The previous code unpacked the weights into five separate
// constant arrays at two call sites with hand-written indices, and a layout
// change silently wrote the temporal weight into a hidden unit at both.
void uploadNeuralModel(ShaderConstants& c, const NeuralFlowModel& model) {
    static_assert(NeuralFlowLayout::kTotal <=
                      sizeof(ShaderConstants::neuralWeights) / sizeof(float),
                  "neuralWeights must hold the whole model");
    std::copy(model.weights.begin(), model.weights.end(), c.neuralWeights);
    c.neuralControl[0] = model.trained ? 1.0f : 0.0f;
}

constexpr DWORD kGpuWaitTimeoutMs = 5000;
// Worst case per submission: two cached luma/packed pyramids plus
// bidirectional SAD searches over every pyramid level.
constexpr UINT kDescriptorsPerRegion = 512;
constexpr UINT kConstantsPerRegion = 96;

// SAD block size per pyramid level: 8 px at every level, FFX's grid. A
// block straddling an object boundary is what decides motion on busy
// backgrounds, and it is four times less contaminated by background than a
// 16 px block (which in turn was far better than the original 32 px).
uint32_t sadBlockSize(int level) {
    (void)level;
    return 8u;
}
// Search radius per level: the refine radius at level 0, the coarse
// radius below it — tapering at the top of the pyramid. At the coarsest
// levels one pixel spans huge screen areas and every 8 px block mixes
// many textures, so the SAD there is the least trustworthy in the graph:
// a wide window at that scale is exactly where random wrong vectors spawn
// and then propagate down as predictors. The accumulated reach of the
// finer levels (~radius * 2^level per level) keeps total coverage nearly
// unchanged, while the candidate count at the most unreliable scale drops
// 16x. The taper trims where the measurements mean the least.
//
// Cost is a step function of the radius, not a smooth curve: see
// FlowSettings::coarseSearchRadius. Keep every level's span at or below 13.
int sadRadius(int level, int levels, int coarse, int refine, int minRefineLevel) {
    if (level < minRefineLevel) return 0;
    if (level == 0 && levels > 1) return refine;
    const int fromTop = (levels - 1) - level;   // 0 at the coarsest level
    if (fromTop == 0) return std::max(3, coarse >> 2);   // coarsest: +-3
    if (fromTop == 1) return std::max(4, coarse >> 1);   // next: +-4
    return coarse;
}
}

FlowEngine::~FlowEngine() {
    if (m_fence && m_fenceEvent && m_fence->GetCompletedValue() < m_fenceValue) {
        if (SUCCEEDED(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent)))
            WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs);
    }
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    destroyFfx();
#endif
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
}

bool FlowEngine::initialize(GraphicsDevice* device, const ComputeKernels* kernels,
                            ID3D12Fence* presentFence, const FlowSettings& settings) {
    m_device = device;
    m_kernels = kernels;
    m_presentFence = presentFence;
    m_settings = settings;
    m_neuralModelLoaded = false;
    if (settings.neuralFlow && settings.allowTrainedModel) {
        char modelPath[MAX_PATH] = {};
        const DWORD length = GetEnvironmentVariableA("MOTION_ENHANCER_NEURAL_MODEL", modelPath, sizeof(modelPath));
        if (length > 0 && length < sizeof(modelPath)) {
            m_neuralModelLoaded = m_neuralModel.load(modelPath);
            if (!m_neuralModelLoaded) std::cerr << "FlowEngine: invalid neural model " << modelPath << "\n";
        }
        if (!m_neuralModelLoaded) {
            std::cerr << "FlowEngine: neural flow requires a trained MNFN model in "
                         "MOTION_ENHANCER_NEURAL_MODEL (see --train-neural).\n";
            return false;
        }
    }
    ID3D12Device* d3d = device->device();

    for (auto& set : m_commands) {
        if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&set.allocator)))) return false;
    }
    if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commands[0].allocator.Get(),
                                      nullptr, IID_PPV_ARGS(&m_list))) ||
        FAILED(m_list->Close()) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        return false;
    }
    m_fence->SetName(L"FlowFence");
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;
    if (!m_dispatch.initialize(d3d, kListsInFlight, kDescriptorsPerRegion, kConstantsPerRegion)) return false;

    D3D12_QUERY_HEAP_DESC query = {};
    query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query.Count = kListsInFlight * 2;
    if (FAILED(d3d->CreateQueryHeap(&query, IID_PPV_ARGS(&m_queryHeap))) ||
        FAILED(device->createBuffer(sizeof(uint64_t) * kListsInFlight * 2, D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE,
                                    m_queryReadback))) {
        m_queryHeap.Reset();
        m_queryReadback.Reset();
    }
    device->flowQueue()->GetTimestampFrequency(&m_timestampFrequency);
    return true;
}

void FlowEngine::setSettings(const FlowSettings& settings) {
    const bool engineChanged = settings.preferFfxOpticalFlow != m_settings.preferFfxOpticalFlow ||
                               settings.neuralFlow != m_settings.neuralFlow;
    m_settings = settings;
    if (engineChanged) {
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
        m_ffxLastId = UINT64_MAX;
#endif
        m_lastSadCurrentId = UINT64_MAX;
        m_usingFfx = false;
    }
}

bool FlowEngine::waitFence(uint64_t value) {
    if (m_fence->GetCompletedValue() >= value) return true;
    if (FAILED(m_fence->SetEventOnCompletion(value, m_fenceEvent))) return false;
    if (WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs) != WAIT_OBJECT_0) {
        std::cerr << "FlowEngine: GPU fence timeout (device removed reason 0x" << std::hex
                  << m_device->device()->GetDeviceRemovedReason() << std::dec << ")\n";
        return false;
    }
    return true;
}

bool FlowEngine::resize(uint32_t width, uint32_t height, int slotCount) {
    if (width == 0 || height == 0) return false;
    slotCount = std::clamp(slotCount, 2, kMaxSlots);
    if (m_width == width && m_height == height && m_slotCount == slotCount) return true;
    // Every in-flight list may still reference the old resources.
    if (!waitFence(m_fenceValue)) return false;
    m_width = width;
    m_height = height;
    m_slotCount = slotCount;
    {
        std::lock_guard<std::mutex> lock(m_ringMutex);
        for (auto& entry : m_ring) entry = {};
    }
    m_lastSadCurrentId = UINT64_MAX;

    int levels = std::clamp(m_settings.pyramidLevels, 1, kMaxLevels);
    while (levels > 1 && (std::min(width, height) >> (levels - 1)) < kMinCoarsestExtent) --levels;
    m_levels = levels;

    for (auto& pyramid : m_pyramids) pyramid = {};
    for (int l = 0; l < kMaxLevels; ++l) {
        for (auto& dir : m_estimatedFlow) for (auto& parity : dir) parity[l].Reset();
        for (auto& dir : m_searchFlow) dir[l].Reset();
    }
    for (int l = 0; l < levels; ++l) {
        const uint32_t lw = std::max(1u, width >> l);
        const uint32_t lh = std::max(1u, height >> l);
        const uint32_t block = sadBlockSize(l);
        const uint32_t fw = (lw + block - 1) / block;
        const uint32_t fh = (lh + block - 1) / block;
        for (int s = 0; s < slotCount; ++s) {
            Pyramid& pyramid = m_pyramids[static_cast<size_t>(s)];
            if (FAILED(m_device->createTexture2D(lw, lh, DXGI_FORMAT_R16_FLOAT,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COMMON, pyramid.level[l], L"Pyramid"))) return false;
            if (m_neuralModelLoaded) {
                if (FAILED(m_device->createTexture2D(lw, lh, DXGI_FORMAT_R16G16_FLOAT,
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                     D3D12_RESOURCE_STATE_COMMON, pyramid.feature[l], L"LearnedFeature")))
                    return false;
            } else if (FAILED(m_device->createTexture2D((lw + 3) / 4, lh, DXGI_FORMAT_R32_UINT,
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                        D3D12_RESOURCE_STATE_COMMON, pyramid.packed[l], L"PackedLuma")))
                return false;
        }
        for (auto& dir : m_estimatedFlow) {
            for (auto& parity : dir) {
                if (FAILED(m_device->createTexture2D(fw, fh, kFlowFormat,
                                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                      D3D12_RESOURCE_STATE_COMMON, parity[l], L"EstimatedFlow"))) return false;
            }
        }
        for (auto& dir : m_searchFlow) {
            if (FAILED(m_device->createTexture2D(fw, fh, kFlowFormat,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COMMON, dir[l], L"SearchFlow"))) return false;
        }
    }

    // Ring entries hold the finest-level 8x8 grid (both engines); the
    // consumer indexes by the entry's block size.
    const uint32_t ringBlock = sadBlockSize(0);
    const uint32_t ringW = (width + ringBlock - 1) / ringBlock;
    const uint32_t ringH = (height + ringBlock - 1) / ringBlock;
    for (auto& entry : m_ring) {
        if (FAILED(m_device->createTexture2D(ringW, ringH, kFlowFormat,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_COMMON, entry.forward, L"RingForward")) ||
            FAILED(m_device->createTexture2D(ringW, ringH, kFlowFormat,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_COMMON, entry.backward, L"RingBackward"))) return false;
    }

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    destroyFfx();
    if (FAILED(m_device->createTexture2D(ringW, ringH, DXGI_FORMAT_R16G16_SINT,
                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                         D3D12_RESOURCE_STATE_COMMON, m_ffxVectors, L"FfxVectors")) ||
        FAILED(m_device->createTexture2D(3, 1, DXGI_FORMAT_R32_UINT,
                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                         D3D12_RESOURCE_STATE_COMMON, m_ffxScd, L"FfxScd"))) return false;
    if (!createFfx()) {
        std::cerr << "FlowEngine: AMD FidelityFX Optical Flow unavailable; using the SAD graph.\n";
    }
#endif
    return true;
}

// ------------------------------------------------------------------ FFX

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
bool FlowEngine::createFfx() {
    const FfxDevice device = ffxGetDeviceDX12(m_device->device());
    const size_t scratchSize = ffxGetScratchMemorySizeDX12(1);
    m_ffxScratch.resize(scratchSize);
    if (ffxGetInterfaceDX12(&m_ffxInterface, device, m_ffxScratch.data(), scratchSize, 1) != FFX_OK) {
        destroyFfx();
        return false;
    }
    FfxOpticalflowContextDescription desc = {};
    desc.backendInterface = m_ffxInterface;
    desc.resolution = {m_width, m_height};
    m_ffxContext = std::make_unique<FfxOpticalflowContext>();
    if (ffxOpticalflowContextCreate(m_ffxContext.get(), &desc) != FFX_OK) {
        destroyFfx();
        return false;
    }
    m_ffxReady = true;
    m_ffxLastId = UINT64_MAX;
    return true;
}

void FlowEngine::destroyFfx() {
    if (m_ffxReady && m_ffxContext) ffxOpticalflowContextDestroy(m_ffxContext.get());
    m_ffxContext.reset();
    m_ffxInterface = {};
    m_ffxScratch.clear();
    m_ffxReady = false;
    m_usingFfx = false;
    m_ffxLastId = UINT64_MAX;
}

bool FlowEngine::recordFfx(const FlowInput& previous, const FlowInput& current, Entry& out) {
    auto dispatch = [&](ID3D12Resource* color, bool reset) {
        FfxOpticalflowDispatchDescription desc = {};
        desc.commandList = ffxGetCommandListDX12(m_list.Get());
        // Registered as COMPUTE_READ, the state FFX reads it in, so the
        // backend records no barrier on it: the shared capture texture is
        // simultaneous-access and read through implicit promotion, possibly
        // by the present queue at the same time.
        desc.color = ffxGetResourceDX12(color, ffxGetResourceDescriptionDX12(color),
                                        L"MotionEnhancer OF Input", FFX_RESOURCE_STATE_COMPUTE_READ);
        desc.opticalFlowVector = ffxGetResourceDX12(
            m_ffxVectors.Get(), ffxGetResourceDescriptionDX12(m_ffxVectors.Get(), FFX_RESOURCE_USAGE_UAV),
            L"MotionEnhancer OF Vectors", FFX_RESOURCE_STATE_COMMON);
        desc.opticalFlowSCD = ffxGetResourceDX12(
            m_ffxScd.Get(), ffxGetResourceDescriptionDX12(m_ffxScd.Get(), FFX_RESOURCE_USAGE_UAV),
            L"MotionEnhancer OF SCD", FFX_RESOURCE_STATE_COMMON);
        desc.reset = reset;
        desc.backbufferTransferFunction = 0;
        desc.minMaxLuminance = {0.0f, 1.0f};
        // The backend returns every registered external resource to the
        // state it was registered in, so no barriers are needed after the
        // dispatch (the vector/SCD outputs end back in COMMON).
        return ffxOpticalflowContextDispatch(m_ffxContext.get(), &desc) == FFX_OK;
    };

    // FFX estimates flow between the last two frames it was given. When
    // `previous` is the frame it saw last, one dispatch continues the chain;
    // otherwise re-establish it with a reset dispatch of `previous` first.
    bool ok = true;
    if (m_ffxLastId != previous.id) ok = dispatch(previous.texture, true);
    if (ok) ok = dispatch(current.texture, false);
    if (!ok) return false;
    m_ffxLastId = current.id;

    // FFX vectors are current -> previous on an 8x8 grid: that is our
    // backward field; the forward field is its negation.
    ShaderConstants c = {};
    c.width = m_width;
    c.height = m_height;
    c.blockSize = c.parentBlockSize = 8;
    const uint32_t fw = (m_width + 7) / 8;
    const uint32_t fh = (m_height + 7) / 8;
    std::array<ID3D12Resource*, 6> in = {m_ffxVectors.Get(), nullptr, nullptr, nullptr, nullptr, nullptr};
    c.flowScale = 1.0f;
    ok = m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FfxFlowConvert, in, out.backward.Get(), c,
                              (fw + 15) / 16, (fh + 15) / 16);
    c.flowScale = -1.0f;
    ok = ok && m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FfxFlowConvert, in, out.forward.Get(), c,
                                    (fw + 15) / 16, (fh + 15) / 16);
    out.blockSize = 8;
    out.singleField = true;
    return ok;
}
#endif

// ------------------------------------------------------------------ SAD

bool FlowEngine::recordPyramid(const FlowInput& frame, Pyramid& pyramid) {
    ID3D12GraphicsCommandList* list = m_list.Get();
    ShaderConstants c = {};
    c.width = m_width;
    c.height = m_height;
    c.totalLevels = m_levels;
    if (m_neuralModelLoaded) uploadNeuralModel(c, m_neuralModel);
    c.levelWidth = m_width;
    c.levelHeight = m_height;
    // Capture textures are shared (simultaneous-access): no barriers.
    std::array<ID3D12Resource*, 6> in = {frame.texture, nullptr, nullptr, nullptr, nullptr, nullptr};
    bool ok = m_dispatch.dispatch(list, *m_kernels, Kernel::Luminance, in, pyramid.level[0].Get(), c,
                                  (m_width + 15) / 16, (m_height + 15) / 16, 0x1);
    for (int l = 0; ok && l < m_levels; ++l) {
        c.levelWidth = std::max(1u, m_width >> l);
        c.levelHeight = std::max(1u, m_height >> l);
        c.levelIndex = l;
        if (l > 0) {
            in = {pyramid.level[l - 1].Get(), nullptr, nullptr, nullptr, nullptr, nullptr};
            ok &= m_dispatch.dispatch(list, *m_kernels, Kernel::Pyramid, in, pyramid.level[l].Get(), c,
                                      (c.levelWidth + 15) / 16, (c.levelHeight + 15) / 16);
        }
        if (!m_neuralModelLoaded) {
            // Packed bytes keep the classical scalar SAD reads coalesced.
            in = {pyramid.level[l].Get(), nullptr, nullptr, nullptr, nullptr, nullptr};
            ok &= m_dispatch.dispatch(list, *m_kernels, Kernel::PackLuma, in, pyramid.packed[l].Get(), c,
                                      ((c.levelWidth + 3) / 4 + 15) / 16, (c.levelHeight + 15) / 16);
        }
    }
    // Learned features carry context from the next coarser level, so build
    // them coarse-to-fine after the entire luma pyramid exists.
    for (int l = m_levels - 1; ok && m_neuralModelLoaded && l >= 0; --l) {
        c.levelWidth = std::max(1u, m_width >> l);
        c.levelHeight = std::max(1u, m_height >> l);
        c.levelIndex = l;
        in = {pyramid.level[l].Get(), l + 1 < m_levels ? pyramid.feature[l + 1].Get() : nullptr,
              nullptr, nullptr, nullptr, nullptr};
        ok &= m_dispatch.dispatch(list, *m_kernels, Kernel::LearnedFeatures, in,
                                  pyramid.feature[l].Get(), c,
                                  (c.levelWidth + 15) / 16, (c.levelHeight + 15) / 16);
    }
    if (ok) pyramid.id = frame.id;
    return ok;
}

bool FlowEngine::recordSad(const FlowInput& previous, const FlowInput& current, Entry& out) {
    ID3D12GraphicsCommandList* list = m_list.Get();
    const int levels = m_levels;
    bool ok = true;

    // Luma pyramids are built once per frame and cached in the slot: in a
    // running stream only the new frame costs a pyramid, as in FFX.
    Pyramid& prevPyramid = m_pyramids[static_cast<size_t>(std::clamp(previous.slot, 0, m_slotCount - 1))];
    Pyramid& currPyramid = m_pyramids[static_cast<size_t>(std::clamp(current.slot, 0, m_slotCount - 1))];
    if (&prevPyramid == &currPyramid) return false;
    if (prevPyramid.id != previous.id) ok &= recordPyramid(previous, prevPyramid);
    if (ok && currPyramid.id != current.id) ok &= recordPyramid(current, currPyramid);

    // Temporal predictor: the previous pair's estimated flow at every level,
    // valid only while the chain is unbroken (that pair ended on this
    // pair's first frame).
    //
    // MOTION_ENHANCER_NO_TEMPORAL=1 unbinds it, which is the only way to A/B
    // what the chain contributes: every other offline path runs a single
    // pair, where it is unbound anyway.
    static const bool noTemporal = [] {
        char buffer[8] = {};
        return GetEnvironmentVariableA("MOTION_ENHANCER_NO_TEMPORAL", buffer, sizeof(buffer)) > 0 &&
               buffer[0] != '0';
    }();
    const bool temporalValid = !noTemporal && m_lastSadCurrentId == previous.id;
    const int parity = m_flowParity;
    const int previousParity = 1 - parity;

    ShaderConstants c = {};
    c.width = m_width;
    c.height = m_height;
    c.totalLevels = levels;
    if (m_neuralModelLoaded) uploadNeuralModel(c, m_neuralModel);
    std::array<ID3D12Resource*, 6> in = {};
    for (int direction = 0; ok && direction < 2; ++direction) {
        const Pyramid& ref = direction == 0 ? prevPyramid : currPyramid;
        const Pyramid& cand = direction == 0 ? currPyramid : prevPyramid;
        auto& estimated = m_estimatedFlow[static_cast<size_t>(direction)][static_cast<size_t>(parity)];
        auto& temporal = m_estimatedFlow[static_cast<size_t>(direction)][static_cast<size_t>(previousParity)];
        for (int l = levels - 1; ok && l >= 0; --l) {
            c.levelWidth = std::max(1u, m_width >> l);
            c.levelHeight = std::max(1u, m_height >> l);
            c.levelIndex = l;
            const uint32_t block = sadBlockSize(l);
            c.blockSize = static_cast<int32_t>(block);
            c.parentBlockSize = sadBlockSize(std::min(l + 1, levels - 1));
            // The learned window scores every displacement with a 16-sample
            // feature correlation over two textures, so it dominates the
            // whole graph and gets its own, narrower schedule. Reach does not
            // come from one level's window: it compounds down the pyramid.
            //
            // neuralWindowRadius() is the single source of truth, shared with
            // the CPU reference and the trainer. Applying the SAD taper here
            // instead left the coarsest level at radius 3 on the GPU and 4 on
            // the CPU, and the whole recursion then diverged from the
            // function the weights were fitted to: 5.4% of blocks picked a
            // different anchor. That is exactly what --neural-parity-check
            // is for.
            c.searchRadius = m_neuralModelLoaded
                ? neuralWindowRadius(l, levels)
                : sadRadius(l, levels, m_settings.coarseSearchRadius,
                            m_settings.refineSearchRadius, m_settings.minRefineLevel);
            c.smoothnessWeight = m_settings.smoothnessWeight;
            static const int sadDebug = [] {
                char buffer[8] = {};
                return GetEnvironmentVariableA("MOTION_ENHANCER_SAD_DEBUG", buffer, sizeof(buffer)) > 0
                    ? std::atoi(buffer) : 0;
            }();
            // MotionSearchCS diagnostics (1: edge counts, 3: parent flow, 5/6: skip stages).
            if (sadDebug) c.smoothnessWeight = -static_cast<float>(sadDebug);
            c.flowScale = temporalValid ? 1.0f : 0.0f;   // "temporal predictor bound"
            const uint32_t fw = (c.levelWidth + block - 1) / block;
            const uint32_t fh = (c.levelHeight + block - 1) / block;
            in = {ref.level[l].Get(), m_neuralModelLoaded ? nullptr : cand.packed[l].Get(),
                   l < levels - 1 ? estimated[l + 1].Get() : nullptr,
                   temporalValid ? temporal[l].Get() : nullptr,
                   m_neuralModelLoaded ? ref.feature[l].Get() : nullptr,
                   m_neuralModelLoaded ? cand.feature[l].Get() : nullptr};
            // One 8x8 group per flow block, into the scratch field.
            ID3D12Resource* raw = m_searchFlow[static_cast<size_t>(direction)][static_cast<size_t>(l)].Get();
            ok &= m_dispatch.dispatch(list, *m_kernels, Kernel::MotionSearch, in, raw, c, fw, fh);

            // Fill the components no block could measure from the ones that
            // could. A block only constrains the component of its vector
            // along which its own content changes; on a page of charts and
            // anti-aliased text 77% of blocks constrain neither, and they
            // emitted whatever the inheritance chain handed them, which
            // starts at zero. Bilinearly blending that zero against a
            // textured neighbour's real vector is what dragged the
            // background beside text and smeared glyph edges.
            //
            // Filling at *every* level is what gives it reach: the filled
            // field is the next level's predictor, so a flat region far
            // larger than the kernel is still covered, and any finer block
            // with real structure re-measures and overrides it. Doing it
            // only at level 0 was tried and costs 8-16 dB on exactly the
            // text cases it exists for.
            //
            // The fill is therefore part of the estimator, not a
            // post-process, so neuralFlow() on the CPU and the trainer both
            // run it too - otherwise the neural weights would be deployed
            // inside a recursion they were not fitted to, which is the
            // mismatch that sank the previous model.
            //
            // Skipped under the matcher diagnostics, which put edge counts
            // rather than vectors in the field.
            if (ok && !sadDebug) {
                in = {raw, nullptr, nullptr, nullptr, nullptr, nullptr};
                ok &= m_dispatch.dispatch(list, *m_kernels, Kernel::FillFlow, in,
                                          estimated[l].Get(), c, (fw + 7) / 8, (fh + 7) / 8);
            } else if (ok) {
                DispatchContext::transition(list, raw, D3D12_RESOURCE_STATE_COMMON,
                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
                DispatchContext::transition(list, estimated[l].Get(), D3D12_RESOURCE_STATE_COMMON,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
                list->CopyResource(estimated[l].Get(), raw);
                DispatchContext::transition(list, raw, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            D3D12_RESOURCE_STATE_COMMON);
                DispatchContext::transition(list, estimated[l].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                            D3D12_RESOURCE_STATE_COMMON);
            }
            if (ok && l == 0) {
                // The filled field goes to the presenter's ring unchanged.
                ID3D12Resource* finest = direction == 0 ? out.forward.Get() : out.backward.Get();
                DispatchContext::transition(list, estimated[l].Get(), D3D12_RESOURCE_STATE_COMMON,
                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
                DispatchContext::transition(list, finest, D3D12_RESOURCE_STATE_COMMON,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
                list->CopyResource(finest, estimated[l].Get());
                DispatchContext::transition(list, estimated[l].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            D3D12_RESOURCE_STATE_COMMON);
                DispatchContext::transition(list, finest, D3D12_RESOURCE_STATE_COPY_DEST,
                                            D3D12_RESOURCE_STATE_COMMON);
            }
        }
    }
    if (ok) {
        m_flowParity = previousParity;
        m_lastSadCurrentId = current.id;
    }
    out.blockSize = sadBlockSize(0);
    out.singleField = false;   // direction 0 and 1 are independent searches
    return ok;
}

// ------------------------------------------------------------ submission

bool FlowEngine::beginList() {
    const UINT idx = static_cast<UINT>(m_submitCount % kListsInFlight);
    CommandSet& set = m_commands[idx];
    if (set.fence && !waitFence(set.fence)) return false;
    lastGpuTimeMs();   // consume this set's timestamps before it is rewritten
    if (FAILED(set.allocator->Reset()) || FAILED(m_list->Reset(set.allocator.Get(), nullptr))) return false;
    m_dispatch.beginRegion(idx);
    if (m_queryHeap) m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, idx * 2);
    return true;
}

bool FlowEngine::endList(const FlowInput& previous, const FlowInput& current, uint64_t& fenceValue) {
    const UINT idx = static_cast<UINT>(m_submitCount % kListsInFlight);
    if (m_queryHeap) {
        // A timestamp does not wait for in-flight dispatches; flush all UAV
        // work first so the measured time covers the whole list.
        D3D12_RESOURCE_BARRIER flush = {};
        flush.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        flush.UAV.pResource = nullptr;
        m_list->ResourceBarrier(1, &flush);
        m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, idx * 2 + 1);
        m_list->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, idx * 2, 2,
                                 m_queryReadback.Get(), static_cast<UINT64>(idx) * 2 * sizeof(uint64_t));
    }
    if (FAILED(m_list->Close())) return false;
    ID3D12CommandQueue* queue = m_device->flowQueue();
    ID3D12Fence* ready = m_device->captureReadyFence12();
    // The capture copies are published through the shared fence.
    if (previous.readyValue && FAILED(queue->Wait(ready, previous.readyValue))) return false;
    if (current.readyValue && FAILED(queue->Wait(ready, current.readyValue))) return false;
    ID3D12CommandList* lists[] = {m_list.Get()};
    queue->ExecuteCommandLists(1, lists);
    fenceValue = ++m_fenceValue;
    if (FAILED(queue->Signal(m_fence.Get(), fenceValue))) return false;
    m_commands[idx].fence = fenceValue;
    ++m_submitCount;
    return true;
}

int FlowEngine::acquireEntry() {
    std::lock_guard<std::mutex> lock(m_ringMutex);
    const uint64_t flowDone = m_fence->GetCompletedValue();
    const uint64_t presentDone = m_presentFence ? m_presentFence->GetCompletedValue() : UINT64_MAX;
    for (int i = 0; i < kRingSize; ++i) {
        const int idx = (m_ringCursor + i) % kRingSize;
        Entry& e = m_ring[idx];
        if (e.flowFence <= flowDone && e.presentFence <= presentDone) {
            e.valid = false;
            e.flowFence = 0;
            m_ringCursor = (idx + 1) % kRingSize;
            return idx;
        }
    }
    return -1;
}

FlowEngine::Result FlowEngine::submitPair(const FlowInput& previous, const FlowInput& current) {
    Result result;
    if (!previous.texture || !current.texture || m_width == 0) return result;
    const int entryIndex = acquireEntry();
    if (entryIndex < 0) {
        std::cerr << "FlowEngine: flow ring exhausted (presenter far behind).\n";
        return result;
    }
    Entry scratch = m_ring[static_cast<size_t>(entryIndex)];  // textures only
    m_currentEntry = entryIndex;
    if (!beginList()) return result;

    bool ok = false;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    if (m_settings.preferFfxOpticalFlow && m_ffxReady) {
        ok = recordFfx(previous, current, scratch);
        if (!ok) {
            // Never execute a partially recorded FFX list; rebuild it with
            // the SAD graph and stop trying FFX for this stream.
            std::cerr << "FlowEngine: FFX dispatch failed; switching to the SAD graph.\n";
            m_list->Close();
            m_ffxReady = false;
            if (!beginList()) return result;
        }
    }
#endif
    m_usingFfx = ok;
    if (!ok) ok = recordSad(previous, current, scratch);
    if (!ok) {
        m_list->Close();
        return result;
    }
    uint64_t fenceValue = 0;
    if (!endList(previous, current, fenceValue)) return result;

    {
        std::lock_guard<std::mutex> lock(m_ringMutex);
        Entry& e = m_ring[static_cast<size_t>(entryIndex)];
        e.blockSize = scratch.blockSize;
        e.singleField = scratch.singleField;
        e.flowFence = fenceValue;
        e.valid = true;
    }
    result.entry = entryIndex;
    result.fence = fenceValue;
    return result;
}

FlowEngine::Entry FlowEngine::entry(int index) const {
    std::lock_guard<std::mutex> lock(m_ringMutex);
    if (index < 0 || index >= kRingSize) return {};
    return m_ring[static_cast<size_t>(index)];
}

void FlowEngine::noteEntryPresented(int index, uint64_t presentFenceValue) {
    std::lock_guard<std::mutex> lock(m_ringMutex);
    if (index < 0 || index >= kRingSize) return;
    Entry& e = m_ring[static_cast<size_t>(index)];
    e.presentFence = std::max(e.presentFence, presentFenceValue);
}

void FlowEngine::readTimestamps(UINT set) {
    if (!m_queryReadback || m_timestampFrequency == 0) return;
    D3D12_RANGE range = {static_cast<SIZE_T>(set) * 2 * sizeof(uint64_t),
                         static_cast<SIZE_T>(set + 1) * 2 * sizeof(uint64_t)};
    uint64_t* data = nullptr;
    if (FAILED(m_queryReadback->Map(0, &range, reinterpret_cast<void**>(&data)))) return;
    const uint64_t begin = data[set * 2];
    const uint64_t end = data[set * 2 + 1];
    D3D12_RANGE none = {0, 0};
    m_queryReadback->Unmap(0, &none);
    if (end > begin) {
        m_lastGpuTimeMs = static_cast<double>(end - begin) * 1000.0 / static_cast<double>(m_timestampFrequency);
    }
}

double FlowEngine::lastGpuTimeMs() {
    // Consume the timestamps of every submission that has completed since
    // the last call (newest wins). Called from both threads.
    std::lock_guard<std::mutex> lock(m_timestampMutex);
    const uint64_t submitted = m_submitCount.load();
    while (m_timestampsRead < submitted) {
        const UINT set = static_cast<UINT>(m_timestampsRead % kListsInFlight);
        if (m_fence->GetCompletedValue() < m_commands[set].fence) break;
        readTimestamps(set);
        ++m_timestampsRead;
    }
    return m_lastGpuTimeMs;
}
