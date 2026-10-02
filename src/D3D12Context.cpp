#include "D3D12Context.h"
#include "D3D11Context.h"

#include <d3dcompiler.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <array>

namespace {
constexpr DWORD kGpuWaitTimeoutMs = 5000;

std::filesystem::path ShaderPath(const std::string& name) {
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    const auto dir = std::filesystem::path(path).parent_path();
    for (const auto& candidate : {dir / "shaders" / name, dir / name,
                                  std::filesystem::path("shaders") / name}) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}
}

D3D12Context::~D3D12Context() {
    // Never destroy an allocator or resource while its queue work is live.
    // A device that has stopped making progress cannot be made safe by an
    // infinite destructor wait, so leave teardown to device removal in that
    // case rather than hanging the process forever.
    if (!m_failed && m_queue && m_fence && m_fence->GetCompletedValue() < m_fenceValue && m_fenceEvent) {
        if (SUCCEEDED(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent)))
            WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs);
    }
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    destroyFfxOpticalFlow();
#endif
    if (m_failed) {
        // A non-progressing device cannot safely release objects referenced by
        // the outstanding command list. Leak the native objects until process
        // teardown rather than releasing them while the queue may still read
        // them.
        for (auto& level : m_pyr0) {
            level.texture.Detach(); level.flow.Detach(); level.filteredFlow.Detach();
        }
        for (auto& level : m_pyr1) {
            level.texture.Detach(); level.flow.Detach(); level.filteredFlow.Detach();
        }
        for (auto& buffer : m_backBuffer) buffer.Detach();
        m_output.Detach();
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
        m_ffxBackwardFlow.Detach();
        m_ffxForwardConvertedFlow.Detach();
        m_ffxBackwardConvertedFlow.Detach();
        m_ffxScd.Detach();
#endif
        for (auto& pipeline : m_pipeline) pipeline.Detach();
        m_rootSignature.Detach();
        m_srvUavHeap.Detach();
        m_constantRing.Detach();
        m_commandList.Detach();
        m_allocator.Detach();
        m_presentAllocator1.Detach();
        m_ffxAllocator.Detach();
        m_fence.Detach();
        m_captureReady.Detach();
        m_swapChain.Detach();
        m_infoQueue.Detach();
        m_queue.Detach();
        m_device.Detach();
    }
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    if (m_frameLatencyWaitableObject) CloseHandle(m_frameLatencyWaitableObject);
    if (m_constantRing && m_constantRingCpu) m_constantRing->Unmap(0, nullptr);
}

bool D3D12Context::createConstantRing() {
    D3D12_HEAP_PROPERTIES upload = {};
    upload.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256ull * kConstantRingSlots;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(m_device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(&m_constantRing)))) {
        return false;
    }
    if (FAILED(m_constantRing->Map(0, nullptr,
                                   reinterpret_cast<void**>(&m_constantRingCpu)))) {
        m_constantRing.Reset();
        return false;
    }
    m_constantRingGpu = m_constantRing->GetGPUVirtualAddress();
    return true;
}

ID3D12Resource* D3D12Context::constantRingSlot(const ShaderConstants& constants,
                                               D3D12_GPU_VIRTUAL_ADDRESS& gpuAddress) {
    const UINT slot = m_constantRingCursor % kConstantRingSlots;
    ++m_constantRingCursor;
    memcpy(m_constantRingCpu + static_cast<size_t>(slot) * 256, &constants, sizeof(constants));
    gpuAddress = m_constantRingGpu + static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(slot) * 256;
    return m_constantRing.Get();
}

bool D3D12Context::initialize(ID3D12Device* device, ID3D12CommandQueue* queue) {
    m_device = device;
    m_queue = queue;
    m_device.As(&m_infoQueue);
    if (!m_device || !m_queue) return false;

    if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&m_allocator))) ||
        FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&m_presentAllocator1))) ||
        FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&m_ffxAllocator))) ||
        FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            m_allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&m_commandList))) ||
        FAILED(m_commandList->Close()) ||
        FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                     IID_PPV_ARGS(&m_fence)))) {
        return false;
    }
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap = {};
    heap.NumDescriptors = kDescriptorHeapSize;
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(m_device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&m_srvUavHeap)))) {
        return false;
    }
    m_descriptorSize = m_device->GetDescriptorHandleIncrementSize(heap.Type);
    if (!createConstantRing()) return false;
    return loadPipeline();
}

bool D3D12Context::createSwapChain(HWND hwnd, uint32_t width, uint32_t height) {
    if (!m_device || !m_queue || !hwnd || width == 0 || height == 0) return false;

    const LUID luid = m_device->GetAdapterLuid();
    ComPtr<IDXGIFactory4> factory4;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4)))) return false;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory4->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) return false;
    ComPtr<IDXGIFactory2> factory;
    if (FAILED(adapter->GetParent(IID_PPV_ARGS(&factory)))) return false;

    DXGI_SWAP_CHAIN_DESC1 scDesc = {};
    scDesc.Width = width;
    scDesc.Height = height;
    // Match the compute output format. BGRA typed UAVs are not supported on
    // all hardware, while RGBA8 is guaranteed at feature level 11.
    scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scDesc.SampleDesc.Count = 1;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    scDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    HRESULT hr = factory->CreateSwapChainForHwnd(
        m_queue.Get(), hwnd, &scDesc, nullptr, nullptr, m_swapChain.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create D3D12 swap chain for HWND (0x"
                  << std::hex << hr << std::dec << ")\n";
        return false;
    }

    ComPtr<IDXGISwapChain2> swapChain2;
    if (SUCCEEDED(m_swapChain.As(&swapChain2))) {
        swapChain2->SetMaximumFrameLatency(1);
        m_frameLatencyWaitableObject = swapChain2->GetFrameLatencyWaitableObject();
    }
    m_swapChain.As(&m_swapChain3);

    for (UINT i = 0; i < 2; ++i) {
        if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffer[i])))) {
            std::cerr << "Error: Failed to retrieve D3D12 back buffer " << i << ".\n";
            return false;
        }
    }
    return true;
}

void D3D12Context::resizeSwapChain(uint32_t width, uint32_t height) {
    if (m_swapChain && width > 0 && height > 0) {
        m_swapChain->ResizeBuffers(
            2, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        for (UINT i = 0; i < 2; ++i) {
            m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffer[i]));
        }
    }
}

bool D3D12Context::loadShaderBytecode(const std::string& name, std::vector<uint8_t>& bytes) {
    auto path = ShaderPath(name);
    if (path.empty()) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return !bytes.empty();
}

bool D3D12Context::loadPipeline() {
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    for (int i = 0; i < 2; ++i) {
        samplers[i].Filter = i == 0 ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
                                    : D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplers[i].AddressU = samplers[i].AddressV = samplers[i].AddressW =
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].ShaderRegister = static_cast<UINT>(i);
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC root = {};
    root.NumParameters = 3;
    root.pParameters = params;
    root.NumStaticSamplers = 2;
    root.pStaticSamplers = samplers;
    root.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized, errors;
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &serialized, &errors))) {
        return false;
    }
    if (FAILED(m_device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                             serialized->GetBufferSize(),
                                             IID_PPV_ARGS(&m_rootSignature)))) {
        return false;
    }
    static const char* names[] = {"LuminanceCS.cso", "PyramidCS.cso", "MotionSearchCS.cso",
                                   "FilterFlowCS.cso", "InterpolateCS.cso", "PresentFrameCS.cso",
                                   "FfxFlowConvertCS.cso"};
#ifndef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    constexpr size_t pipelineCount = 6;
#else
    constexpr size_t pipelineCount = 7;
#endif
    for (size_t i = 0; i < pipelineCount; ++i) {
        std::vector<uint8_t> shader;
        if (!loadShaderBytecode(names[i], shader)) {
            std::cerr << "D3D12: missing " << names[i] << "; native dispatch disabled.\n";
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_rootSignature.Get();
        desc.CS = {shader.data(), shader.size()};
        if (FAILED(m_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pipeline[i]))))
            return false;
    }
    return true;
}

bool D3D12Context::beginCommands(bool asyncPair, int presentationIndex) {
    m_allocatorKind = asyncPair ? 2 : presentationIndex == 1 ? 1 : 0;
    ID3D12CommandAllocator* allocator = asyncPair ? m_ffxAllocator.Get() :
        presentationIndex == 1 ? m_presentAllocator1.Get() : m_allocator.Get();
    if (!allocator) return false;
    const uint64_t previousFence = m_allocatorFence[m_allocatorKind];
    if (m_fence->GetCompletedValue() < previousFence) {
        // Pair preparation is polled by the presenter and must never block.
        // Presentation only waits here if the GPU has fallen a full two back
        // buffers behind; normal 144 Hz operation takes the nonblocking path.
        if (asyncPair) return false;
        // The wait itself is unavoidable back-pressure: the latency token
        // consumed by the main loop must always pair with a Present, or the
        // waitable swapchain stops producing slots. Log it throttled so GPU
        // starvation is visible instead of silent judder.
        LARGE_INTEGER t0 = {}, t1 = {}, freq = {};
        QueryPerformanceCounter(&t0);
        const bool ready = waitForFence(previousFence);
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&freq);
        const double stallMs =
            static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 /
            static_cast<double>(freq.QuadPart);
        if (stallMs > 0.5 &&
            (m_lastStallLogQpc.QuadPart == 0 ||
             (t1.QuadPart - m_lastStallLogQpc.QuadPart) * 1000.0 /
                 static_cast<double>(freq.QuadPart) > 1000.0)) {
            m_lastStallLogQpc = t1;
            std::cerr << "D3D12: presentation stalled " << stallMs
                      << " ms waiting for GPU backlog (allocator fence behind).\n";
        }
        if (!ready) return false;
    }
    if (FAILED(allocator->Reset()) || FAILED(m_commandList->Reset(allocator, nullptr))) {
        return false;
    }
    m_descriptorCursor = asyncPair ? kPairDescriptorBase :
        presentationIndex == 1 ? kPresent1DescriptorBase : 0;
    m_constantRingCursor = asyncPair ? kPairRingCursor :
        presentationIndex == 1 ? kPresent1RingCursor : 0;
    m_pendingReadyWaits.clear();
    return true;
}

bool D3D12Context::submit(uint64_t& fenceValue) {
    if (FAILED(m_commandList->Close())) return false;
    // Capture hand-off: D3D11 signals the shared ready fence after its copy.
    // Queue waits execute before the command list below consumes the frames.
    for (uint64_t value : m_pendingReadyWaits) {
        if (FAILED(m_queue->Wait(m_captureReady.Get(), value))) return false;
    }
    m_pendingReadyWaits.clear();
    ID3D12CommandList* lists[] = {m_commandList.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    fenceValue = ++m_fenceValue;
    if (FAILED(m_queue->Signal(m_fence.Get(), fenceValue))) return false;
    if (m_computeDone) m_queue->Signal(m_computeDone.Get(), fenceValue);
    m_allocatorFence[m_allocatorKind] = fenceValue;
    return true;
}

bool D3D12Context::pairOutputReady() const {
    return m_pairPendingFenceValue == 0 ||
           (m_fence && m_fence->GetCompletedValue() >= m_pairPendingFenceValue);
}

bool D3D12Context::waitPairOutput() {
    if (m_pairPendingFenceValue == 0) return true;
    if (!waitForFence(m_pairPendingFenceValue)) return false;
    m_pairPendingFenceValue = 0;
    return true;
}

bool D3D12Context::waitForFence(uint64_t fenceValue) {
    if (m_fence->GetCompletedValue() >= fenceValue) return true;
    if (FAILED(m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent)) ||
        WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs) != WAIT_OBJECT_0) {
        const HRESULT removedReason = m_device->GetDeviceRemovedReason();
        std::cerr << "D3D12: GPU fence did not complete within "
                  << kGpuWaitTimeoutMs << " ms; disabling native dispatch (device reason 0x"
                  << std::hex << removedReason << std::dec << ").\n";
        dumpInfoQueue("fence timeout");
        m_failed = true;
        return false;
    }
    return true;
}

void D3D12Context::dumpInfoQueue(const char* context) {
    if (!m_infoQueue) return;
    const UINT64 count = m_infoQueue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        m_infoQueue->GetMessage(i, nullptr, &size);
        if (size == 0) continue;
        std::vector<uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(m_infoQueue->GetMessage(i, message, &size)) && message->pDescription)
            std::cerr << "D3D12 validation (" << context << "): " << message->pDescription << "\n";
    }
    m_infoQueue->ClearStoredMessages();
}

void D3D12Context::transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                              D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

void D3D12Context::recordCaptureWaits(const SharedFrame& frame) {
    // The D3D11 capture copy is published through the shared ready fence;
    // queue waits execute before this command list in submit().
    if (m_captureReady && frame.readyValue) m_pendingReadyWaits.push_back(frame.readyValue);
}

bool D3D12Context::dispatchPresentFrame(const SharedFrame& source, ID3D12Resource* target,
                                         const ShaderConstants& constants) {
    if (!isReady() || !source.resource || !target) return false;
    const D3D12_RESOURCE_DESC sourceDesc = source.resource->GetDesc();
    const D3D12_RESOURCE_DESC targetDesc = target->GetDesc();
    if (sourceDesc.Width != constants.width || sourceDesc.Height != constants.height ||
        targetDesc.Width != constants.width || targetDesc.Height != constants.height) return false;
    int presentationIndex = -1;
    for (int i = 0; i < 2; ++i) {
        if (target == m_backBuffer[static_cast<size_t>(i)].Get()) presentationIndex = i;
    }
    if (!beginCommands(false, presentationIndex)) return false;
    recordCaptureWaits(source);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = sourceDesc.Format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(source.resource, &srv, cpu);
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpu = cpu;
    uavCpu.ptr += m_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = gpu;
    uavGpu.ptr += m_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = m_output ? m_output->GetDesc().Format : DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_output.Get(), nullptr, &uav, uavCpu);

    D3D12_GPU_VIRTUAL_ADDRESS cbAddress = 0;
    constantRingSlot(constants, cbAddress);


    ID3D12DescriptorHeap* heaps[] = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(1, heaps);
    m_commandList->SetPipelineState(m_pipeline[5].Get());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetComputeRootDescriptorTable(0, gpu);
    uavGpu.ptr = gpu.ptr + static_cast<UINT64>(m_descriptorSize);
    m_commandList->SetComputeRootDescriptorTable(1, uavGpu);
    m_commandList->SetComputeRootConstantBufferView(2, cbAddress);
    transition(m_commandList.Get(), source.resource, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->Dispatch((constants.width + 15) / 16, (constants.height + 15) / 16, 1);
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON);
    transition(m_commandList.Get(), source.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(m_commandList.Get(), target, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    m_commandList->CopyResource(target, m_output.Get());
    transition(m_commandList.Get(), target, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON);
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    uint64_t fenceValue = 0;
    if (!submit(fenceValue)) return false;
    if (presentationIndex < 0 && !waitForFence(fenceValue)) return false;
    m_outputValid = true;
    return true;
}

bool D3D12Context::runDiagnostics(D3D11Context* context) {
    if (!isReady() || !context) return false;
    ShaderConstants constants = {};
    constants.width = constants.height = constants.levelWidth = constants.levelHeight = 16;
    constants.blockSize = constants.parentBlockSize = 32;
    constants.totalLevels = 1;
    auto runNativePass = [&](const char* name, int pipeline, DXGI_FORMAT inputFormat,
                             DXGI_FORMAT outputFormat, bool secondInput) {
        std::cout << "D3D12 diagnostic: native one-group " << name << " dispatch...\n";
        ComPtr<ID3D12Resource> input0, input1, output;
        if (FAILED(createTexture2D(16, 16, inputFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COMMON, input0)) ||
            (secondInput && FAILED(createTexture2D(16, 16, inputFormat,
                                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                   D3D12_RESOURCE_STATE_COMMON, input1))) ||
            FAILED(createTexture2D(16, 16, outputFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COMMON, output)) || !beginCommands()) return false;
        std::array<ID3D12Resource*, 4> inputs = {input0.Get(), input1.Get(), nullptr, nullptr};
        if (!dispatchPass(pipeline, inputs, output.Get(), constants, 1, 1)) return false;
        uint64_t fenceValue = 0;
        return submit(fenceValue) && waitForFence(fenceValue);
    };
    if (!runNativePass("LuminanceCS", 0, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16_FLOAT, false) ||
        !runNativePass("PyramidCS", 1, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_FLOAT, false) ||
        !runNativePass("MotionSearchCS", 2, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R32G32_FLOAT, true) ||
        !runNativePass("PresentFrameCS", 5, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, false)) {
        return false;
    }
    dumpInfoQueue("native diagnostic");

    // FFX contexts have a minimum resolution, so the full-graph diagnostics
    // (which initialize the FFX context via resizeResources) run at 320x240.
    if (!resizeResources(320, 240, 2)) return false;
    ShaderConstants largeConstants = constants;
    largeConstants.width = largeConstants.levelWidth = 320;
    largeConstants.height = largeConstants.levelHeight = 240;
    std::cout << "D3D12 diagnostic: shared NT-handle texture hand-off...\n";
    ComPtr<ID3D11Texture2D> shared11;
    ComPtr<ID3D12Resource> shared12;
    if (!context->createSharedTexture2D(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        D3D11_BIND_SHADER_RESOURCE, shared11, shared12)) {
        std::cerr << "D3D12 diagnostic: shared texture creation failed.\n";
        return false;
    }
    std::vector<uint8_t> largePixels(static_cast<size_t>(320) * 240 * 4, 127);
    context->context->UpdateSubresource(shared11.Get(), 0, nullptr, largePixels.data(), 320 * 4, 0);
    context->context->Flush();
    SharedFrame frame{shared12.Get(), context->signalCaptureReady(shared11.Get())};
    ComPtr<ID3D12Resource> presentTarget;
    if (FAILED(createTexture2D(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_COMMON, presentTarget))) return false;
    const bool interopOk = dispatchPresentFrame(frame, presentTarget.Get(), largeConstants);
    if (interopOk) dumpInfoQueue("shared-texture diagnostic");
    if (!interopOk) {
        std::cerr << "D3D12 diagnostic: shared-texture dispatch failed.\n";
        return false;
    }

    std::cout << "D3D12 diagnostic: 320x240 two-level frame-pair command list...\n";
    ComPtr<ID3D11Texture2D> large0_11, large1_11;
    ComPtr<ID3D12Resource> large0, large1;
    if (!context->createSharedTexture2D(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        D3D11_BIND_SHADER_RESOURCE, large0_11, large0) ||
        !context->createSharedTexture2D(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        D3D11_BIND_SHADER_RESOURCE, large1_11, large1)) return false;
    context->context->UpdateSubresource(large0_11.Get(), 0, nullptr, largePixels.data(), 320 * 4, 0);
    context->context->UpdateSubresource(large1_11.Get(), 0, nullptr, largePixels.data(), 320 * 4, 0);
    context->context->Flush();
    SharedFrame largeFrame0{large0.Get(), context->signalCaptureReady(large0_11.Get())};
    SharedFrame largeFrame1{large1.Get(), context->signalCaptureReady(large1_11.Get())};
    bool graphOk = dispatchFramePair(largeFrame0, largeFrame1, 0, 1, 2, 0, 0, 0, 0.0f);
    if (graphOk) {
        graphOk = waitPairOutput();
    }
    if (graphOk) {
        ComPtr<ID3D12Resource> interpolateTarget;
        if (FAILED(createTexture2D(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COMMON, interpolateTarget))) return false;
        graphOk = dispatchInterpolate(interpolateTarget.Get(), 0.5f);
    }
    if (graphOk) dumpInfoQueue("frame-pair diagnostic");
    std::cout << "D3D12 diagnostic: " << (graphOk ? "PASS" : "FAIL") << "\n";
    return graphOk;
}

HRESULT D3D12Context::createBuffer(const D3D12_RESOURCE_DESC& desc,
                                   D3D12_RESOURCE_STATES initialState,
                                   ComPtr<ID3D12Resource>& resource) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    return m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                             initialState, nullptr,
                                             IID_PPV_ARGS(resource.ReleaseAndGetAddressOf()));
}

HRESULT D3D12Context::createTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                                      D3D12_RESOURCE_FLAGS flags,
                                      D3D12_RESOURCE_STATES initialState,
                                      ComPtr<ID3D12Resource>& resource) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return createBuffer(desc, initialState, resource);
}

bool D3D12Context::resizeResources(uint32_t width, uint32_t height, int totalLevels) {
    if (!isReady() || width == 0 || height == 0 || totalLevels < 1 || totalLevels > 8)
        return false;
    if (m_width == width && m_height == height && m_levels == totalLevels) return true;

    m_width = width;
    m_height = height;
    m_levels = totalLevels;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    destroyFfxOpticalFlow();
    m_ffxBackwardFlow.Reset();
    m_ffxForwardConvertedFlow.Reset();
    m_ffxBackwardConvertedFlow.Reset();
    m_ffxScd.Reset();
    m_usingFfxFlow = false;
#endif
    m_output.Reset();
    m_outputValid = false;
    m_pairFrame[0] = {};
    m_pairFrame[1] = {};
    for (auto& level : m_pyr0) level = {};
    for (auto& level : m_pyr1) level = {};
    if (FAILED(createTexture2D(width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_COMMON, m_output))) return false;

    for (int i = 0; i < totalLevels; ++i) {
        const uint32_t levelWidth = std::max(1u, width >> i);
        const uint32_t levelHeight = std::max(1u, height >> i);
        const uint32_t flowWidth = (levelWidth + 31) / 32;
        const uint32_t flowHeight = (levelHeight + 31) / 32;
        for (auto* pyramid : {&m_pyr0[i], &m_pyr1[i]}) {
            if (FAILED(createTexture2D(levelWidth, levelHeight, DXGI_FORMAT_R16_FLOAT,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, pyramid->texture))) return false;
            if (FAILED(createTexture2D(flowWidth, flowHeight, DXGI_FORMAT_R32G32_FLOAT,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, pyramid->flow))) return false;
            if (FAILED(createTexture2D(flowWidth, flowHeight, DXGI_FORMAT_R32G32_FLOAT,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, pyramid->filteredFlow))) return false;
        }
    }
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    const uint32_t flowWidth = (width + 7) / 8;
    const uint32_t flowHeight = (height + 7) / 8;
    if (FAILED(createTexture2D(flowWidth, flowHeight, DXGI_FORMAT_R16G16_SINT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COMMON, m_ffxBackwardFlow)) ||
        FAILED(createTexture2D(flowWidth, flowHeight, DXGI_FORMAT_R32G32_FLOAT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COMMON, m_ffxForwardConvertedFlow)) ||
        FAILED(createTexture2D(flowWidth, flowHeight, DXGI_FORMAT_R32G32_FLOAT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COMMON, m_ffxBackwardConvertedFlow)) ||
        FAILED(createTexture2D(3, 1, DXGI_FORMAT_R32_UINT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_COMMON, m_ffxScd))) return false;
    if (!initializeFfxOpticalFlow()) {
        std::cerr << "D3D12: AMD FidelityFX Optical Flow unavailable; using MSAD fallback.\n";
    }
#endif
    return true;
}

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
void D3D12Context::destroyFfxOpticalFlow() {
    if (m_ffxReady) {
        ffxOpticalflowContextDestroy(&m_ffxContext);
    }
    m_ffxContext = {};
    m_ffxInterface = {};
    m_ffxScratch.clear();
    m_ffxReady = false;
    m_ffxWarmed = false;
    m_ffxLastFrame1 = nullptr;
    m_usingFfxFlow = false;
}

bool D3D12Context::initializeFfxOpticalFlow() {
    if (!m_device || !m_width || !m_height) return false;
    FfxOpticalflowContextDescription desc = {};
    const FfxDevice device = ffxGetDeviceDX12(m_device.Get());
    const size_t scratchSize = ffxGetScratchMemorySizeDX12(1);
    m_ffxScratch.resize(scratchSize);
    if (ffxGetInterfaceDX12(&m_ffxInterface, device, m_ffxScratch.data(), scratchSize, 1) != FFX_OK) {
        destroyFfxOpticalFlow();
        return false;
    }
    desc.backendInterface = m_ffxInterface;
    desc.resolution = {m_width, m_height};
    if (ffxOpticalflowContextCreate(&m_ffxContext, &desc) != FFX_OK) {
        destroyFfxOpticalFlow();
        return false;
    }
    m_ffxReady = true;
    return true;
}

bool D3D12Context::dispatchFfxFramePair(const SharedFrame& frame0, const SharedFrame& frame1,
                                         uint64_t frame1Index) {
    if (!m_ffxReady || !frame0.resource || !frame1.resource || !beginCommands(true)) return false;
    recordCaptureWaits(frame0);
    recordCaptureWaits(frame1);

    // Prime the SCD pipeline only when the temporal chain is broken. FFX
    // computes flow between the last two dispatched frames, so a sequential
    // pair (previous == the chain's last frame1) needs a single frame1
    // dispatch. Anything else (first pair, seek, offline re-run, starve gap)
    // re-establishes the chain with a reset plus warm-up dispatches of frame0.
    const bool discontinuity = !m_ffxWarmed || frame0.resource != m_ffxLastFrame1;
    auto dispatch = [&](ID3D12Resource* color, bool reset) {
        FfxOpticalflowDispatchDescription desc = {};
        desc.commandList = ffxGetCommandListDX12(m_commandList.Get());
        desc.color = ffxGetResourceDX12(color, ffxGetResourceDescriptionDX12(color),
                                          L"MotionEnhancer Optical Flow Input", FFX_RESOURCE_STATE_COMMON);
        desc.opticalFlowVector = ffxGetResourceDX12(m_ffxBackwardFlow.Get(),
            ffxGetResourceDescriptionDX12(m_ffxBackwardFlow.Get(), FFX_RESOURCE_USAGE_UAV),
            L"MotionEnhancer Optical Flow Vectors", FFX_RESOURCE_STATE_COMMON);
        desc.opticalFlowSCD = ffxGetResourceDX12(m_ffxScd.Get(),
            ffxGetResourceDescriptionDX12(m_ffxScd.Get(), FFX_RESOURCE_USAGE_UAV),
            L"MotionEnhancer Optical Flow SCD", FFX_RESOURCE_STATE_COMMON);
        desc.reset = reset;
        desc.backbufferTransferFunction = 0;
        desc.minMaxLuminance = {0.0f, 1.0f};
        return ffxOpticalflowContextDispatch(&m_ffxContext, &desc) == FFX_OK;
    };

    bool ok = true;
    if (discontinuity) {
        // The bundled optical flow callbacks always emit vectors (the stock
        // frameIndex <= 5 post-reset suppression is patched out), so a single
        // reset dispatch re-establishes the temporal chain. Sequential pairs
        // still submit only frame1.
        ok = dispatch(frame0.resource, true);
    }
    if (ok) ok = dispatch(frame1.resource, false);
    if (ok) {
        // The DX12 backend's ExecuteGpuJobs returns registered exported
        // resources to COMMON. The shared frame inputs remain compute reads.
        transition(m_commandList.Get(), frame0.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
        transition(m_commandList.Get(), frame1.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
        ShaderConstants constants = {};
        constants.width = m_width;
        constants.height = m_height;
        constants.blockSize = constants.parentBlockSize = 8;
        std::array<ID3D12Resource*, 4> input = {m_ffxBackwardFlow.Get(), nullptr, nullptr, nullptr};
        const uint32_t flowWidth = (m_width + 7) / 8;
        const uint32_t flowHeight = (m_height + 7) / 8;
        // FFX's native field is current -> previous. It is our backward
        // field; negation gives the approximate frame0 -> frame1 field.
        constants.flowScale = 1.0f;
        ok = dispatchPass(6, input, m_ffxBackwardConvertedFlow.Get(), constants,
                          (flowWidth + 15) / 16, (flowHeight + 15) / 16);
        constants.flowScale = -1.0f;
        ok = ok && dispatchPass(6, input, m_ffxForwardConvertedFlow.Get(), constants,
                                 (flowWidth + 15) / 16, (flowHeight + 15) / 16);
    }
    uint64_t fenceValue = 0;
    if (ok) ok = submit(fenceValue);
    if (ok) {
        m_pairPendingFenceValue = fenceValue;
        m_ffxLastCurrentIndex = frame1Index;
        m_ffxLastFrame1 = frame1.resource;
        m_ffxWarmed = true;
        m_usingFfxFlow = true;
        return true;
    }
    // Do not execute a partially recorded FFX command list; the private
    // inputs remain in COMMON, so the MSAD path can retry safely.
    m_commandList->Close();
    m_ffxReady = false;
    m_ffxWarmed = false;
    m_ffxLastCurrentIndex = UINT64_MAX;
    m_ffxLastFrame1 = nullptr;
    m_usingFfxFlow = false;
    std::cerr << "D3D12: AMD FidelityFX Optical Flow dispatch failed; using MSAD fallback.\n";
    return false;
}
#endif

bool D3D12Context::dispatchPass(int pipelineIndex,
                                const std::array<ID3D12Resource*, 4>& inputs,
                                ID3D12Resource* output, const ShaderConstants& constants,
                                uint32_t groupsX, uint32_t groupsY) {
    if (pipelineIndex < 0 || pipelineIndex >= static_cast<int>(m_pipeline.size()) ||
        !m_pipeline[static_cast<size_t>(pipelineIndex)] || !output) return false;

    const UINT base = m_descriptorCursor;
    if (base + 5 > kDescriptorHeapSize) return false;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(base) * m_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += static_cast<UINT64>(base) * m_descriptorSize;
    for (size_t i = 0; i < 4; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu;
        handle.ptr += static_cast<SIZE_T>(i) * m_descriptorSize;
        if (inputs[i]) {
            m_device->CreateShaderResourceView(inputs[i], nullptr, handle);
        } else {
            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv = {};
            nullSrv.Format = DXGI_FORMAT_R32G32_FLOAT;
            nullSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            nullSrv.Texture2D.MipLevels = 1;
            m_device->CreateShaderResourceView(nullptr, &nullSrv, handle);
        }
        if (inputs[i]) transition(m_commandList.Get(), inputs[i],
                                  D3D12_RESOURCE_STATE_COMMON,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE uav = cpu;
    uav.ptr += static_cast<SIZE_T>(4) * m_descriptorSize;
    m_device->CreateUnorderedAccessView(output, nullptr, nullptr, uav);
    transition(m_commandList.Get(), output, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_descriptorCursor += 5;

    D3D12_GPU_VIRTUAL_ADDRESS cbAddress = 0;
    constantRingSlot(constants, cbAddress);

    ID3D12DescriptorHeap* heaps[] = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(1, heaps);
    m_commandList->SetPipelineState(m_pipeline[static_cast<size_t>(pipelineIndex)].Get());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetComputeRootDescriptorTable(0, gpu);
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = gpu;
    uavGpu.ptr += static_cast<UINT64>(m_descriptorSize) * 4;
    m_commandList->SetComputeRootDescriptorTable(1, uavGpu);
    m_commandList->SetComputeRootConstantBufferView(2, cbAddress);
    m_commandList->Dispatch(groupsX, groupsY, 1);

    transition(m_commandList.Get(), output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COMMON);
    for (ID3D12Resource* input : inputs) {
        if (input) transition(m_commandList.Get(), input,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                              D3D12_RESOURCE_STATE_COMMON);
    }
    return true;
}

bool D3D12Context::rePresentLastOutput() {
    if (!m_outputValid || !m_output || !m_swapChain) return false;
    const UINT index = currentBackBufferIndex();
    ID3D12Resource* back = m_backBuffer[index].Get();
    if (!back || back->GetDesc().Width != m_output->GetDesc().Width ||
        back->GetDesc().Height != m_output->GetDesc().Height) return false;
    if (!beginCommands(false, static_cast<int>(index))) return false;
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(m_commandList.Get(), back, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    m_commandList->CopyResource(back, m_output.Get());
    transition(m_commandList.Get(), back, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COMMON);
    transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    uint64_t fenceValue = 0;
    if (!submit(fenceValue)) return false;
    m_outputValid = true;
    return true;
}

bool D3D12Context::dispatchFramePair(const SharedFrame& frame0, const SharedFrame& frame1,
                                      uint64_t frame0Index, uint64_t frame1Index,
                                      int totalLevels, int minRefineLevel,
                                      int coarseRadius, int refineRadius,
                                      float smoothnessWeight) {
    (void)frame0Index;
    if (!isReady() || !frame0.resource || !frame1.resource ||
        totalLevels < 1 || totalLevels > m_levels) return false;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    if (m_preferFfx && m_ffxReady && dispatchFfxFramePair(frame0, frame1, frame1Index)) {
        m_pairFrame = {frame0, frame1};
        return true;
    }
#endif
    if (!beginCommands(true)) return false;
    recordCaptureWaits(frame0);
    recordCaptureWaits(frame1);
    // Zero-copy: bind the shared capture resources directly; they stay stable
    // for the pair's lifetime because the presenter never retires a slot
    // cached as previous/current.
    m_pairFrame = {frame0, frame1};
    bool ok = true;
    ShaderConstants c = {};
    c.width = m_width; c.height = m_height; c.totalLevels = totalLevels;
    std::array<ID3D12Resource*, 4> in = {};
    c.levelWidth = m_width; c.levelHeight = m_height;
    in[0] = frame0.resource;
    ok &= dispatchPass(0, in, m_pyr0[0].texture.Get(), c, (m_width + 15) / 16, (m_height + 15) / 16);
    in[0] = frame1.resource;
    ok &= dispatchPass(0, in, m_pyr1[0].texture.Get(), c, (m_width + 15) / 16, (m_height + 15) / 16);
    for (int level = 1; ok && level < totalLevels; ++level) {
        c.levelWidth = std::max(1u, m_width >> level);
        c.levelHeight = std::max(1u, m_height >> level);
        in = {};
        in[0] = m_pyr0[level - 1].texture.Get();
        ok &= dispatchPass(1, in, m_pyr0[level].texture.Get(), c,
                           (c.levelWidth + 15) / 16, (c.levelHeight + 15) / 16);
        in = {};
        in[0] = m_pyr1[level - 1].texture.Get();
        ok &= dispatchPass(1, in, m_pyr1[level].texture.Get(), c,
                           (c.levelWidth + 15) / 16, (c.levelHeight + 15) / 16);
    }
    for (int direction = 0; ok && direction < 2; ++direction) {
        for (int level = totalLevels - 1; level >= 0; --level) {
            c.levelWidth = std::max(1u, m_width >> level);
            c.levelHeight = std::max(1u, m_height >> level);
            c.levelIndex = level;
            c.blockSize = 32;
            c.parentBlockSize = 32;
            // Levels finer than minRefineLevel propagate the parent predictor
            // only (radius 0), skipping the per-block candidate search there.
            c.searchRadius = level == totalLevels - 1 ? coarseRadius
                             : level >= minRefineLevel ? refineRadius : 0;
            c.smoothnessWeight = smoothnessWeight;
            auto& ref = direction == 0 ? m_pyr0[level] : m_pyr1[level];
            auto& cand = direction == 0 ? m_pyr1[level] : m_pyr0[level];
            auto& dst = direction == 0 ? m_pyr0[level] : m_pyr1[level];
            in = {};
            in[0] = ref.texture.Get(); in[1] = cand.texture.Get();
            if (level < totalLevels - 1) {
                auto& parent = direction == 0 ? m_pyr0[level + 1] : m_pyr1[level + 1];
                in[2] = parent.filteredFlow.Get();
            }
            ok &= dispatchPass(2, in, dst.flow.Get(), c,
                               (c.levelWidth + 31) / 32, (c.levelHeight + 31) / 32);
            in = {};
            in[0] = dst.flow.Get(); in[1] = ref.texture.Get(); in[2] = cand.texture.Get();
            // FilterFlowCS uses one thread per flow vector; each 16x16 group
            // filters a tile of vectors. Dispatching the flow-grid dims as
            // GROUP counts launched 256x the needed threads.
            const uint32_t filterFlowWidth = (c.levelWidth + 31) / 32;
            const uint32_t filterFlowHeight = (c.levelHeight + 31) / 32;
            ok &= dispatchPass(3, in, dst.filteredFlow.Get(), c,
                               (filterFlowWidth + 15) / 16, (filterFlowHeight + 15) / 16);
        }
    }
    uint64_t fenceValue = 0;
    if (ok) ok = submit(fenceValue);
    if (ok) m_pairPendingFenceValue = fenceValue;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    m_usingFfxFlow = false;
#endif
    return ok;
}

bool D3D12Context::dispatchInterpolate(ID3D12Resource* target, float timeT) {
    if (!isReady() || !pairOutputReady() || !target || !m_output ||
        !m_pairFrame[0].resource || !m_pairFrame[1].resource) return false;
    const D3D12_RESOURCE_DESC targetDesc = target->GetDesc();
    if (targetDesc.Width != m_width || targetDesc.Height != m_height) return false;
    int presentationIndex = -1;
    for (int i = 0; i < 2; ++i) {
        if (target == m_backBuffer[static_cast<size_t>(i)].Get()) presentationIndex = i;
    }
    if (!beginCommands(false, presentationIndex)) return false;
    // The pair slots are fence-gated and pinned by the presenter; wait again
    // in case the interpolate submit lands on a fresh queue turn.
    recordCaptureWaits(m_pairFrame[0]);
    recordCaptureWaits(m_pairFrame[1]);
    ShaderConstants c = {};
    c.width = m_width; c.height = m_height; c.levelWidth = m_width; c.levelHeight = m_height;
    c.blockSize = 32; c.parentBlockSize = 32; c.timeT = timeT;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    if (m_usingFfxFlow) c.blockSize = c.parentBlockSize = 8;
#endif
    ID3D12Resource* forwardFlow = m_pyr0[0].filteredFlow.Get();
    ID3D12Resource* backwardFlow = m_pyr1[0].filteredFlow.Get();
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    if (m_usingFfxFlow) {
        forwardFlow = m_ffxForwardConvertedFlow.Get();
        backwardFlow = m_ffxBackwardConvertedFlow.Get();
    }
#endif
    std::array<ID3D12Resource*, 4> in = {m_pairFrame[0].resource, m_pairFrame[1].resource,
                                          forwardFlow, backwardFlow};
    bool ok = dispatchPass(4, in, m_output.Get(), c, (m_width + 15) / 16, (m_height + 15) / 16);
    if (ok) {
        transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(m_commandList.Get(), target, D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST);
        m_commandList->CopyResource(target, m_output.Get());
        transition(m_commandList.Get(), target, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON);
        transition(m_commandList.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
        uint64_t fenceValue = 0;
        ok = submit(fenceValue);
        if (ok && presentationIndex < 0) ok = waitForFence(fenceValue);
    }
    return ok;
}
