#include "DepthService.h"

#include "FlowEngine.h"

#include <cstring>
#include <iostream>

namespace {
constexpr UINT kRowBytes = DepthService::kSize * 4;   // RGBA8 and R32_FLOAT rows alike, 256-byte aligned
constexpr DWORD kWaitMs = 5000;
}  // namespace

DepthService::~DepthService() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    if (m_thread.joinable()) m_thread.join();
    if (m_event) CloseHandle(m_event);
}

bool DepthService::initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* flowFence,
                              int slots) {
    m_device = device;
    m_kernels = kernels;
    m_flowFence = flowFence;
    m_slotCount = std::clamp(slots, 1, kMaxSlots);
    if (!kernels->pipeline(Kernel::DepthInput)) return false;
    if (!m_estimator.initialize(device->device(), DepthEstimator::defaultModelPath())) return false;

    ID3D12Device* d3d = device->device();
    for (int s = 0; s < m_slotCount; ++s) {
        Slot& slot = m_slots[static_cast<size_t>(s)];
        if (FAILED(device->createTexture2D(kSize, kSize, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON, slot.shrunk, L"DepthInput")) ||
            FAILED(device->createTexture2D(kSize, kSize, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_COMMON, slot.depth, L"AiDepth")) ||
            FAILED(device->createBuffer(static_cast<uint64_t>(kRowBytes) * kSize, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, slot.readback)) ||
            FAILED(device->createBuffer(static_cast<uint64_t>(kRowBytes) * kSize, D3D12_HEAP_TYPE_UPLOAD,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, slot.upload)))
            return false;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    if (FAILED(d3d->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_copyQueue))) ||
        FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_allocator))) ||
        FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_allocator.Get(), nullptr, IID_PPV_ARGS(&m_list))) ||
        FAILED(m_list->Close()) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_copyFence))))
        return false;
    m_copyQueue->SetName(L"AiDepthUpload");
    m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_event) return false;
    m_thread = std::thread(&DepthService::run, this);
    return true;
}

bool DepthService::record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame,
                          uint32_t width, uint32_t height) {
    if (!frame.texture || frame.slot < 0 || frame.slot >= m_slotCount) return true;
    Slot& slot = m_slots[static_cast<size_t>(frame.slot)];
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (slot.readyId == frame.id || slot.pendingId == frame.id) return true;
        slot.pendingId = frame.id;
    }
    for (const Job& job : m_recorded)
        if (job.frameId == frame.id) return true;

    ShaderConstants c = {};
    c.width = c.levelWidth = width;
    c.height = c.levelHeight = height;
    // The capture texture is shared: implicit promotion.
    std::array<ID3D12Resource*, 6> in = {frame.texture, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!dispatch.dispatch(list, *m_kernels, Kernel::DepthInput, in, slot.shrunk.Get(), c,
                           (kSize + 15) / 16, (kSize + 15) / 16, 0x1)) return false;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = slot.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, static_cast<UINT>(kSize), static_cast<UINT>(kSize), 1, kRowBytes};
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = slot.shrunk.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    DispatchContext::transition(list, slot.shrunk.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    DispatchContext::transition(list, slot.shrunk.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
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

ID3D12Resource* DepthService::depthFor(const FlowInput& frame) const {
    if (frame.slot < 0 || frame.slot >= m_slotCount) return nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    const Slot& slot = m_slots[static_cast<size_t>(frame.slot)];
    return slot.readyId == frame.id ? slot.depth.Get() : nullptr;
}

ID3D12Resource* DepthService::waitFor(const FlowInput& frame, unsigned timeoutMs) {
    if (frame.slot < 0 || frame.slot >= m_slotCount) return nullptr;
    std::unique_lock<std::mutex> lock(m_mutex);
    const Slot& slot = m_slots[static_cast<size_t>(frame.slot)];
    m_done.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                    [&] { return slot.readyId == frame.id || slot.pendingId != frame.id; });
    return slot.readyId == frame.id ? slot.depth.Get() : nullptr;
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
        const bool ok = produce(job);
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
    if (m_flowFence->GetCompletedValue() < job.flowFence) {
        if (FAILED(m_flowFence->SetEventOnCompletion(job.flowFence, m_event)) ||
            WaitForSingleObject(m_event, kWaitMs) != WAIT_OBJECT_0) return false;
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(kSize) * kSize * 3);
    uint8_t* mapped = nullptr;
    const D3D12_RANGE whole = {0, static_cast<SIZE_T>(kRowBytes) * kSize};
    if (FAILED(slot.readback->Map(0, &whole, reinterpret_cast<void**>(&mapped)))) return false;
    for (int y = 0; y < kSize; ++y)
        for (int x = 0; x < kSize; ++x)
            for (int c = 0; c < 3; ++c)
                rgb[(static_cast<size_t>(y) * kSize + x) * 3 + c] = mapped[static_cast<size_t>(y) * kRowBytes + x * 4 + c];
    const D3D12_RANGE nothing = {0, 0};
    slot.readback->Unmap(0, &nothing);

    std::vector<float> nearness(static_cast<size_t>(kSize) * kSize);
    if (!m_estimator.estimate(rgb.data(), nearness.data())) return false;
    static const bool report = GetEnvironmentVariableA("MOTION_ENHANCER_DEPTH_REPORT", nullptr, 0) > 0;
    if (report) {
        double sum = 0.0, rgbSum = 0.0;
        for (float v : nearness) sum += v;
        for (uint8_t v : rgb) rgbSum += v;
        std::cout << "  [depth] frame " << job.frameId << " slot " << job.slot << ": input mean "
                  << rgbSum / rgb.size() << ", nearness mean " << sum / nearness.size() << "\n";
    }

    void* upload = nullptr;
    if (FAILED(slot.upload->Map(0, &nothing, &upload))) return false;
    std::memcpy(upload, nearness.data(), nearness.size() * sizeof(float));
    slot.upload->Unmap(0, nullptr);

    if (FAILED(m_allocator->Reset()) || FAILED(m_list->Reset(m_allocator.Get(), nullptr))) return false;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = slot.depth.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = slot.upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(kSize), static_cast<UINT>(kSize), 1, kRowBytes};
    m_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(m_list->Close())) return false;
    ID3D12CommandList* lists[] = {m_list.Get()};
    m_copyQueue->ExecuteCommandLists(1, lists);
    if (FAILED(m_copyQueue->Signal(m_copyFence.Get(), ++m_copyValue)) ||
        FAILED(m_copyFence->SetEventOnCompletion(m_copyValue, m_event)) ||
        WaitForSingleObject(m_event, kWaitMs) != WAIT_OBJECT_0) return false;
    return true;
}
