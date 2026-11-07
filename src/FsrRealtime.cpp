#include "FsrRealtime.h"

#include <algorithm>
#include <cmath>
#include <iostream>

FsrRealtime::~FsrRealtime() {
    if (!m_device || !m_fence) return;
    // A signal after everything already submitted to the present queue:
    // FSR's own lists and the presenter's lists that showed its frames.
    const uint64_t value = ++m_fenceValue;
    if (FAILED(m_device->presentQueue()->Signal(m_fence.Get(), value))) return;
    if (m_fence->GetCompletedValue() >= value) return;
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done) return;
    // Bounded: a removed device never signals.
    if (SUCCEEDED(m_fence->SetEventOnCompletion(value, done))) WaitForSingleObject(done, 2000);
    CloseHandle(done);
}

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
    if (FAILED(device->createTexture2D(1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, uav, common, m_cameraResult,
                                       L"FsrCameraMotion"))) {
        error = "camera texture";
        return false;
    }
    for (auto& buffer : m_cameraReadback)
        if (FAILED(device->createBuffer(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, buffer))) {
            error = "camera readback";
            return false;
        }
    // Optional instrumentation. Read only fence-complete slots, before reuse;
    // failure to create it must not disable frame generation.
    D3D12_QUERY_HEAP_DESC query = {};
    query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query.Count = kLists * 2;
    if (FAILED(d3d->CreateQueryHeap(&query, IID_PPV_ARGS(&m_queryHeap))) ||
        FAILED(device->createBuffer(sizeof(uint64_t) * kLists * 2, D3D12_HEAP_TYPE_READBACK,
                                     D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, m_queryReadback)) ||
        FAILED(device->presentQueue()->GetTimestampFrequency(&m_timestampFrequency))) {
        m_queryHeap.Reset();
        m_queryReadback.Reset();
        m_timestampFrequency = 0;
    }
    return true;
}

bool FsrRealtime::waitFence(uint64_t value) {
    if (m_fence->GetCompletedValue() >= value) return true;
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done) return false;
    if (SUCCEEDED(m_fence->SetEventOnCompletion(value, done))) WaitForSingleObject(done, 1000);
    CloseHandle(done);
    return m_fence->GetCompletedValue() >= value;
}

bool FsrRealtime::beginList(uint32_t& slot) {
    slot = m_submits % kLists;
    // Three lists deep: normally long finished. Live must not lose an output
    // deadline by sleeping here behind earlier FSR/GPU work.
    if (m_nonBlocking && m_fence->GetCompletedValue() < m_allocatorFence[slot]) {
        ++m_stats.busy;
        return false;
    }
    if (!waitFence(m_allocatorFence[slot])) return false;
    collectGpuTimes();
    if (FAILED(m_allocators[slot]->Reset()) || FAILED(m_list->Reset(m_allocators[slot].Get(), nullptr))) return false;
    m_dispatch.beginRegion(slot);
    return true;
}

bool FsrRealtime::recordCamera(ID3D12Resource* backward, uint32_t blockSize, uint32_t slot) {
    ShaderConstants c = {};
    c.width = c.levelWidth = m_width;
    c.height = c.levelHeight = m_height;
    c.blockSize = c.parentBlockSize = static_cast<int32_t>(blockSize);
    const std::array<ID3D12Resource*, 6> in = {backward, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FsrCamera, in, m_cameraResult.Get(), c, 1, 1))
        return false;
    DispatchContext::transition(m_list.Get(), m_cameraResult.Get(), D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = m_cameraResult.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = m_cameraReadback[slot].Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
    m_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    DispatchContext::transition(m_list.Get(), m_cameraResult.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_COMMON);
    return true;
}

bool FsrRealtime::readCamera(uint32_t slot) {
    float* data = nullptr;
    const D3D12_RANGE range = {0, 4 * sizeof(float)};
    if (FAILED(m_cameraReadback[slot]->Map(0, &range, reinterpret_cast<void**>(&data)))) return false;
    const float x = data[0], y = data[1], agree = data[2];
    const D3D12_RANGE nothing = {0, 0};
    m_cameraReadback[slot]->Unmap(0, &nothing);
    // Too few samples agreeing on one motion (no camera move dominates the
    // picture): treat as no turn rather than chase a mix of objects.
    const bool usable = std::isfinite(x) && std::isfinite(y) && agree >= 0.25f;
    m_motionX = usable ? x : 0.0f;
    m_motionY = usable ? y : 0.0f;
    m_stats.panX = m_motionX;
    m_stats.panY = m_motionY;
    return true;
}

void FsrRealtime::turnCamera(float motionX, float motionY) {
    // A pinhole camera of vertical field of view fov: focal length in
    // full-resolution pixels. The backward field points from where a pixel
    // is now to where it was: content shifted left (vector +x) means the
    // camera turned right; content shifted up (vector +y, rows grow
    // downwards) means it turned down.
    const float focal = 0.5f * static_cast<float>(m_height) / std::tan(0.5f * m_camera.fovVertical);
    // Kept within one turn: a spinning camera turns forever.
    m_yaw = std::remainder(m_yaw + m_yawSign * std::atan(motionX / focal), 6.2831853f);
    m_pitch = std::clamp(m_pitch - m_pitchSign * std::atan(motionY / focal), -1.5f, 1.5f);
    const float cy = std::cos(m_yaw), sy = std::sin(m_yaw), cp = std::cos(m_pitch), sp = std::sin(m_pitch);
    const float forward[3] = {sy * cp, sp, cy * cp};
    const float right[3] = {cy, 0.0f, -sy};
    // Left-handed: up = forward x right ((0,0,1) x (1,0,0) = (0,1,0)).
    const float up[3] = {forward[1] * right[2] - forward[2] * right[1], forward[2] * right[0] - forward[0] * right[2],
                         forward[0] * right[1] - forward[1] * right[0]};
    for (int i = 0; i < 3; ++i) {
        m_camera.forward[i] = forward[i];
        m_camera.right[i] = right[i];
        m_camera.up[i] = up[i];
    }
}

bool FsrRealtime::feed(const FlowInput& input, ID3D12Resource* backward, uint32_t blockSize, ID3D12Resource* nearness,
                       bool reset, ID3D12Resource* output, float frameTimeMs) {
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
    // depth: the frame's AI depth spread over m_depthSpan around the middle
    // of the planes, or one flat surface there without it (FsrDepthCS).
    c.levelWidth = m_renderWidth;
    c.levelHeight = m_renderHeight;
    c.blockSize = c.parentBlockSize = static_cast<int32_t>(blockSize);
    c.flowScale = 0.5f;
    in = {backward, nullptr, nullptr, nullptr, nullptr, nullptr};
    const UINT gx = (m_renderWidth + 15) / 16, gy = (m_renderHeight + 15) / 16;
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FsrMotion, in, m_motion.Get(), c, gx, gy)) return false;
    const bool aiDepth = nearness && m_depthSpan > 0.0f;
    ShaderConstants depthConstants = c;
    depthConstants.timeT = aiDepth ? m_depthSpan : 0.0f;
    const std::array<ID3D12Resource*, 6> depthIn = {aiDepth ? nearness : nullptr, nullptr, nullptr, nullptr, nullptr,
                                                    nullptr};
    if (!m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::FsrDepth, depthIn, m_depth.Get(), depthConstants, gx, gy))
        return false;
    if (aiDepth) ++m_stats.aiDepth;
    std::string error;
    if (!m_fsr.frame(m_list.Get(), id, color, m_motion.Get(), m_depth.Get(), output, reset, frameTimeMs, error,
                     m_cameraMode == CameraMode::Fixed ? nullptr : &m_camera)) {
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
                                         uint64_t flowFenceValue, float frameTimeMs, ID3D12Resource* depth0,
                                         ID3D12Resource* depth1) {
    if (index1 == m_pairIndex) return m_pairValid ? m_outputs[m_current].Get() : nullptr;
    m_pairIndex = index1;
    m_pairValid = false;
    if (!entry.valid || !entry.backward) {
        m_fedIndex = 0;
        return nullptr;
    }

    ID3D12CommandQueue* queue = m_device->presentQueue();
    ID3D12Fence* ready = m_device->captureReadyFence12();
    // Camera from the flow. Live: the previous pair's motion, read once its
    // list has finished (never a wait; until then the last one read stays).
    // Exact (offline): this pair's own motion, submitted alone and waited for.
    if (m_cameraMode == CameraMode::FromFlow && m_cameraPendingFence &&
        m_fence->GetCompletedValue() >= m_cameraPendingFence) {
        readCamera(m_cameraPendingSlot);
        m_cameraPendingFence = 0;
    }
    uint32_t slot = 0;
    if (m_cameraMode == CameraMode::FromFlowExact) {
        if (!beginList(slot) || !recordCamera(entry.backward.Get(), entry.blockSize, slot) ||
            FAILED(m_list->Close()) ||
            (flowFence && flowFenceValue && FAILED(queue->Wait(flowFence, flowFenceValue))))
            return nullptr;
        ID3D12CommandList* cameraList[] = {m_list.Get()};
        queue->ExecuteCommandLists(1, cameraList);
        m_allocatorFence[slot] = ++m_fenceValue;
        queue->Signal(m_fence.Get(), m_fenceValue);
        ++m_submits;
        if (!waitFence(m_fenceValue) || !readCamera(slot)) return nullptr;
    }
    if (!beginList(slot)) return nullptr;
    if (m_queryHeap) m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2);

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
        m_yaw = m_pitch = 0.0f;
        turnCamera(0.0f, 0.0f);
        ok = feed(in0, nullptr, entry.blockSize, depth0, true, output, frameTimeMs);
    }
    if (m_cameraMode == CameraMode::Spin) turnCamera(kSpinPixels, 0.0f);
    else if (m_cameraMode != CameraMode::Fixed) turnCamera(m_motionX, m_motionY);
    ok = ok && feed(in1, entry.backward.Get(), entry.blockSize, depth1, false, output, frameTimeMs);
    const bool measure = m_cameraMode == CameraMode::FromFlow;
    if (ok && measure) ok = recordCamera(entry.backward.Get(), entry.blockSize, slot);
    if (ok && m_queryHeap) {
        D3D12_RESOURCE_BARRIER flush = {};
        flush.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        m_list->ResourceBarrier(1, &flush);
        m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
        m_list->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2,
                                 m_queryReadback.Get(), static_cast<UINT64>(slot) * 2 * sizeof(uint64_t));
    }
    if (FAILED(m_list->Close())) ok = false;
    if (!ok) {
        ++m_stats.failures;
        m_fedIndex = 0;
        return nullptr;
    }

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
    if (FAILED(queue->Signal(m_fence.Get(), m_fenceValue))) {
        ++m_stats.failures;
        m_fedIndex = 0;
        return nullptr;
    }
    if (m_queryHeap) m_timestampFence[slot] = m_fenceValue;
    ++m_submits;
    if (measure) {
        m_cameraPendingFence = m_fenceValue;
        m_cameraPendingSlot = slot;
    }

    m_fedIndex = index1;
    m_current = next;
    m_pairValid = true;
    ++m_stats.generated;
    return m_outputs[m_current].Get();
}

void FsrRealtime::collectGpuTimes() {
    if (!m_queryReadback || !m_timestampFrequency) return;
    const uint64_t completed = m_fence->GetCompletedValue();
    for (uint32_t slot = 0; slot < kLists; ++slot) {
        if (!m_timestampFence[slot] || m_timestampFence[slot] > completed) continue;
        const D3D12_RANGE range = {slot * 2 * sizeof(uint64_t), (slot + 1) * 2 * sizeof(uint64_t)};
        uint64_t* data = nullptr;
        if (FAILED(m_queryReadback->Map(0, &range, reinterpret_cast<void**>(&data)))) continue;
        const uint64_t begin = data[slot * 2], end = data[slot * 2 + 1];
        const D3D12_RANGE nothing = {0, 0};
        m_queryReadback->Unmap(0, &nothing);
        m_timestampFence[slot] = 0;
        if (end <= begin) continue;
        const double ms = static_cast<double>(end - begin) * 1000.0 / m_timestampFrequency;
        ++m_stats.gpuSamples;
        m_stats.gpuTotalMs += ms;
        m_stats.gpuMaxMs = std::max(m_stats.gpuMaxMs, ms);
    }
}

FsrRealtime::Stats FsrRealtime::takeStats() {
    collectGpuTimes();
    const Stats s = m_stats;
    m_stats = {};
    return s;
}
