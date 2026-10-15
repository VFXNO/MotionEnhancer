#include "DepthService.h"

#include "FlowEngine.h"

#include <chrono>
#include <iostream>

namespace {
constexpr UINT kRowBytes = DepthService::kSize * 4;   // one R32_FLOAT row, 256-byte aligned
constexpr DWORD kWaitMs = 5000;
}  // namespace

DepthService::~DepthService() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    if (m_thread.joinable()) m_thread.join();
    for (Slot& slot : m_slots) {
        m_estimator.releaseBuffer(slot.inputHandle);
        m_estimator.releaseBuffer(slot.outputHandle);
    }
    if (m_event) CloseHandle(m_event);
}

bool DepthService::initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* flowFence,
                              int slots) {
    m_device = device;
    m_kernels = kernels;
    m_flowFence = flowFence;
    m_slotCount = std::clamp(slots, 1, kMaxSlots);
    if (!kernels->pipeline(Kernel::DepthInput) || !kernels->pipeline(Kernel::DepthFinish)) return false;
    if (!m_estimator.initialize(device->device(), DepthEstimator::defaultModelPath())) return false;

    ID3D12Device* d3d = device->device();
    for (int s = 0; s < m_slotCount; ++s) {
        Slot& slot = m_slots[static_cast<size_t>(s)];
        if (FAILED(device->createTexture2D(kSize, 3 * kSize, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON, slot.planar, L"AiDepthInput")) ||
            FAILED(device->createTexture2D(kSize, kSize, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_COMMON, slot.raw, L"AiDepthRaw")) ||
            FAILED(device->createTexture2D(kSize, kSize, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON, slot.depth, L"AiDepth")) ||
            FAILED(device->createBuffer(DepthEstimator::kInputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, slot.input)) ||
            FAILED(device->createBuffer(DepthEstimator::kOutputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, slot.output)))
            return false;
        slot.inputHandle = m_estimator.registerBuffer(slot.input.Get());
        slot.outputHandle = m_estimator.registerBuffer(slot.output.Get());
        if (!slot.inputHandle || !slot.outputHandle) return false;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_networkFence))) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_finishFence))) ||
        FAILED(d3d->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_queue))) ||
        FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_allocator))) ||
        FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_allocator.Get(), nullptr, IID_PPV_ARGS(&m_list))) ||
        FAILED(m_list->Close()) ||
        !m_dispatch.initialize(d3d, 1, 8, 2))
        return false;
    m_queue->SetName(L"AiDepthFinish");
    m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_event) return false;
    m_thread = std::thread(&DepthService::run, this);
    return true;
}

bool DepthService::record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame,
                          uint32_t width, uint32_t height) {
    if (!frame.texture || frame.slot < 0 || frame.slot >= m_slotCount) return true;
    Slot& slot = m_slots[static_cast<size_t>(frame.slot)];
    for (const Job& job : m_recorded)
        if (job.frameId == frame.id) return true;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (slot.readyId == frame.id || slot.pendingId == frame.id) return true;
        // Each frame is offered twice (as a pair's second frame, then as the
        // next pair's first): only its first offer counts toward kStride.
        if (m_newFrames > 0 && frame.id == m_lastFrameId) return true;
        m_lastFrameId = frame.id;
        if (m_newFrames++ % kStride != 0) return true;
        slot.pendingId = frame.id;
    }

    ShaderConstants c = {};
    c.width = c.levelWidth = width;
    c.height = c.levelHeight = height;
    // The capture texture is shared: implicit promotion.
    std::array<ID3D12Resource*, 6> in = {frame.texture, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!dispatch.dispatch(list, *m_kernels, Kernel::DepthInput, in, slot.planar.Get(), c,
                           (kSize + 15) / 16, (kSize + 15) / 16, 0x1)) return false;
    // Planar rows of kSize floats are exactly the NCHW tensor.
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = slot.input.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(kSize), static_cast<UINT>(3 * kSize), 1, kRowBytes};
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = slot.planar.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    DispatchContext::transition(list, slot.planar.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    DispatchContext::transition(list, slot.planar.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    m_recorded.push_back({frame.slot, frame.id, 0});
    return true;
}

void DepthService::submitted(uint64_t flowFenceValue) {
    if (m_recorded.empty()) return;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (Job job : m_recorded) {
            job.flowFence = flowFenceValue;
            m_jobs.push_back(job);
        }
    }
    m_recorded.clear();
    m_wake.notify_one();
}

void DepthService::discarded() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const Job& job : m_recorded) {
        Slot& slot = m_slots[static_cast<size_t>(job.slot)];
        if (slot.pendingId == job.frameId) slot.pendingId = UINT64_MAX;
    }
    m_recorded.clear();
}

ID3D12Resource* DepthService::nearestLocked(uint64_t frameId) const {
    // Its own, else the frame before, else the one after, and so on.
    for (uint64_t distance = 0; distance < static_cast<uint64_t>(kStride); ++distance)
        for (int side = 0; side < (distance ? 2 : 1); ++side) {
            if (side == 0 && distance > frameId) continue;
            const uint64_t id = side == 0 ? frameId - distance : frameId + distance;
            for (int s = 0; s < m_slotCount; ++s)
                if (m_slots[static_cast<size_t>(s)].readyId == id) return m_slots[static_cast<size_t>(s)].depth.Get();
        }
    return nullptr;
}

ID3D12Resource* DepthService::depthFor(const FlowInput& frame) const {
    if (frame.slot < 0 || frame.slot >= m_slotCount) return nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    return nearestLocked(frame.id);
}

ID3D12Resource* DepthService::waitFor(const FlowInput& frame, unsigned timeoutMs) {
    if (frame.slot < 0 || frame.slot >= m_slotCount) return nullptr;
    std::unique_lock<std::mutex> lock(m_mutex);
    const Slot& slot = m_slots[static_cast<size_t>(frame.slot)];
    m_done.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                    [&] { return slot.readyId == frame.id || slot.pendingId != frame.id; });
    return nearestLocked(frame.id);
}

void DepthService::run() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
            if (m_stop) return;
            job = m_jobs.front();
            m_jobs.pop_front();
            // A newer frame already took this slot: this one is gone.
            if (m_slots[static_cast<size_t>(job.slot)].pendingId != job.frameId) continue;
        }
        const auto start = std::chrono::steady_clock::now();
        const bool ok = produce(job);
        static const bool report = GetEnvironmentVariableA("MOTION_ENHANCER_DEPTH_REPORT", nullptr, 0) > 0;
        if (report) {
            std::cout << "  [depth] frame " << job.frameId << ": "
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                      << " ms (network queued in " << m_estimator.lastTimings().network << " ms)"
                      << (ok ? "" : " FAILED") << std::endl;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            Slot& slot = m_slots[static_cast<size_t>(job.slot)];
            if (slot.pendingId == job.frameId) {
                slot.pendingId = UINT64_MAX;
                if (ok) slot.readyId = job.frameId;
            }
        }
        m_done.notify_all();
    }
}

bool DepthService::produce(const Job& job) {
    Slot& slot = m_slots[static_cast<size_t>(job.slot)];
    // The network: its queue waits on the GPU for the flow list that wrote
    // the input, then signals m_networkFence.
    if (!m_estimator.runGpu(slot.inputHandle, slot.outputHandle, m_flowFence.Get(), job.flowFence,
                            m_networkFence.Get(), ++m_networkValue)) return false;

    // Its output to nearness, on this service's queue once the network is done.
    if (FAILED(m_allocator->Reset()) || FAILED(m_list->Reset(m_allocator.Get(), nullptr))) return false;
    m_dispatch.beginRegion(0);
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = slot.raw.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = slot.output.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(kSize), static_cast<UINT>(kSize), 1, kRowBytes};
    DispatchContext::transition(m_list.Get(), slot.raw.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    m_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    DispatchContext::transition(m_list.Get(), slot.raw.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    ShaderConstants c = {};
    c.width = c.levelWidth = kSize;
    c.height = c.levelHeight = kSize;
    std::array<ID3D12Resource*, 6> in = {slot.raw.Get(), nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::DepthFinish, in, slot.depth.Get(), c, 1, 1) ||
        FAILED(m_list->Close())) return false;
    if (FAILED(m_queue->Wait(m_networkFence.Get(), m_networkValue))) return false;
    ID3D12CommandList* lists[] = {m_list.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    if (FAILED(m_queue->Signal(m_finishFence.Get(), ++m_finishValue)) ||
        FAILED(m_finishFence->SetEventOnCompletion(m_finishValue, m_event)) ||
        WaitForSingleObject(m_event, kWaitMs) != WAIT_OBJECT_0) return false;
    return true;
}
