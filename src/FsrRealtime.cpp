#include "FsrRealtime.h"

#include <iostream>

bool FsrRealtime::initialize(GraphicsDevice* device, const ComputeKernels* kernels, uint32_t width, uint32_t height,
                             std::string& error) {
    m_device = device;
    m_kernels = kernels;
    m_width = width;
    m_height = height;
    // FSR's render size, for motion and depth: the flow ring's grid.
    m_renderWidth = (width + 1) / 2;
    m_renderHeight = (height + 1) / 2;
    ID3D12Device* d3d = device->device();
    if (!m_fsr.initialize(d3d, width, height, m_renderWidth, m_renderHeight, DXGI_FORMAT_R8G8B8A8_UNORM, error))
        return false;
    for (auto& allocator : m_allocators)
        if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) {
            error = "command allocator";
            return false;
        }
    if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_allocators[0].Get(), nullptr,
                                      IID_PPV_ARGS(&m_list))) ||
        FAILED(m_list->Close()) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        error = "command list or fence";
        return false;
    }
    // Six dispatches per pair at most (two frames after a reset), seven
    // descriptors each.
    if (!m_dispatch.initialize(d3d, kLists, 64, 16)) {
        error = "descriptor heap";
        return false;
    }
    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES common = D3D12_RESOURCE_STATE_COMMON;
    for (int i = 0; i < 2; ++i) {
        if (FAILED(device->createTexture2D(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, uav, common, m_color[i],
                                           L"FsrColor")) ||
            FAILED(device->createTexture2D(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, uav, common, m_outputs[i],
                                           L"FsrMiddleFrame"))) {
            error = "colour textures";
            return false;
        }
    }
    if (FAILED(device->createTexture2D(m_renderWidth, m_renderHeight, DXGI_FORMAT_R16G16_FLOAT, uav, common, m_motion,
                                       L"FsrMotionVectors")) ||
        FAILED(device->createTexture2D(m_renderWidth, m_renderHeight, DXGI_FORMAT_R32_FLOAT, uav, common, m_depth,
                                       L"FsrDepth"))) {
        error = "motion or depth texture";
        return false;
    }
    return true;
}

bool FsrRealtime::feed(const FlowInput& input, ID3D12Resource* backward, uint32_t blockSize, bool reset,
                       ID3D12Resource* output, float frameTimeMs) {
    const uint64_t id = ++m_frameId;
    ID3D12Resource* color = m_color[id & 1].Get();
    // The game frame as RGBA in a texture of our own: the capture texture is
    // read through implicit promotion (as the presenter reads it), so FSR's
    // own barriers never touch a texture the flow queue may be reading.
    ShaderConstants c = {};
    c.width = c.levelWidth = m_width;
    c.height = c.levelHeight = m_height;
    std::array<ID3D12Resource*, 6> in = {input.texture, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::PresentFrame, in, color, c, (m_width + 15) / 16,
                             (m_height + 15) / 16, 0x1))
        return false;
    // Motion: the pair's backward field (none after a reset: no motion);
    // depth: one flat surface halfway between the planes.
    c.levelWidth = m_renderWidth;
    c.levelHeight = m_renderHeight;
    c.blockSize = c.parentBlockSize = static_cast<int32_t>(blockSize);
    c.flowScale = 0.5f;
    in = {backward, nullptr, nullptr, nullptr, nullptr, nullptr};
    const std::array<ID3D12Resource*, 6> none = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    const UINT gx = (m_renderWidth + 15) / 16, gy = (m_renderHeight + 15) / 16;
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FsrMotion, in, m_motion.Get(), c, gx, gy) ||
        !m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FsrDepth, none, m_depth.Get(), c, gx, gy))
        return false;
    std::string error;
    if (!m_fsr.frame(m_list.Get(), id, color, m_motion.Get(), m_depth.Get(), output, reset, frameTimeMs, error)) {
        if (!m_errorReported) {
            m_errorReported = true;
            std::cerr << "FSR frame generation: " << error << " (the normal interpolation continues)\n";
        }
        return false;
    }
    return true;
}

ID3D12Resource* FsrRealtime::middleFrame(const FlowInput& in0, uint64_t index0, const FlowInput& in1, uint64_t index1,
                                         const FlowEngine::Entry& entry, ID3D12Fence* flowFence,
                                         uint64_t flowFenceValue, float frameTimeMs) {
    if (index1 == m_pairIndex) return m_pairValid ? m_outputs[m_current].Get() : nullptr;
    m_pairIndex = index1;
    m_pairValid = false;
    if (!entry.valid || !entry.backward) {
        m_fedIndex = 0;
        return nullptr;
    }

    const uint32_t slot = m_submits % kLists;
    // Three lists deep: normally long finished.
    if (m_fence->GetCompletedValue() < m_allocatorFence[slot]) {
        HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!done || FAILED(m_fence->SetEventOnCompletion(m_allocatorFence[slot], done))) {
            if (done) CloseHandle(done);
            return nullptr;
        }
        WaitForSingleObject(done, 1000);
        CloseHandle(done);
        if (m_fence->GetCompletedValue() < m_allocatorFence[slot]) return nullptr;
    }
    if (FAILED(m_allocators[slot]->Reset()) || FAILED(m_list->Reset(m_allocators[slot].Get(), nullptr))) return nullptr;
    m_dispatch.beginRegion(slot);

    // FSR makes the frame between the previous frame it was given and this
    // one. When frame0 is that previous frame the sequence continues;
    // otherwise (start, a skipped or dropped frame, a cut) frame0 is given
    // first as a new sequence.
    const bool continues = m_fedIndex != 0 && m_fedIndex == index0;
    const uint32_t next = m_current ^ 1;
    ID3D12Resource* output = m_outputs[next].Get();
    bool ok = true;
    if (!continues) {
        ++m_stats.resets;
        ok = feed(in0, nullptr, entry.blockSize, true, output, frameTimeMs);
    }
    ok = ok && feed(in1, entry.backward.Get(), entry.blockSize, false, output, frameTimeMs);
    if (FAILED(m_list->Close())) ok = false;
    if (!ok) {
        ++m_stats.failures;
        m_fedIndex = 0;
        return nullptr;
    }

    ID3D12CommandQueue* queue = m_device->presentQueue();
    ID3D12Fence* ready = m_device->captureReadyFence12();
    if ((!continues && in0.readyValue && FAILED(queue->Wait(ready, in0.readyValue))) ||
        (in1.readyValue && FAILED(queue->Wait(ready, in1.readyValue))) ||
        (flowFence && flowFenceValue && FAILED(queue->Wait(flowFence, flowFenceValue)))) {
        ++m_stats.failures;
        m_fedIndex = 0;
        return nullptr;
    }
    ID3D12CommandList* lists[] = {m_list.Get()};
    queue->ExecuteCommandLists(1, lists);
    m_allocatorFence[slot] = ++m_fenceValue;
    queue->Signal(m_fence.Get(), m_fenceValue);
    ++m_submits;

    m_fedIndex = index1;
    m_current = next;
    m_pairValid = true;
    ++m_stats.generated;
    return m_outputs[m_current].Get();
}

FsrRealtime::Stats FsrRealtime::takeStats() {
    const Stats s = m_stats;
    m_stats = {};
    return s;
}
