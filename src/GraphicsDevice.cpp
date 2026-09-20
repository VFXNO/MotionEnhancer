#include "GraphicsDevice.h"

#include <iostream>
#include <vector>

#ifndef D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_HIGH
#define D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_HIGH (100)
#endif

namespace {
bool envFlag(const char* name) {
    char buffer[8] = {};
    return GetEnvironmentVariableA(name, buffer, sizeof(buffer)) > 0 && buffer[0] != '0';
}
}

GraphicsDevice::~GraphicsDevice() {
    // Queues are released after every object that references them; callers
    // (FlowEngine / Presenter) drain their own fences before destruction.
}

bool GraphicsDevice::initialize(GraphicsAdapterPreference preference) {
    // The debug layer costs several ms per frame of CPU validation, which is
    // itself a source of stutter. It is opt-in (MOTION_ENHANCER_D3D_DEBUG=1
    // or a _DEBUG build).
#ifdef _DEBUG
    m_debugLayer = true;
#else
    m_debugLayer = envFlag("MOTION_ENHANCER_D3D_DEBUG");
#endif
    if (m_debugLayer) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            std::cout << "D3D12 debug layer enabled.\n";
            // MOTION_ENHANCER_D3D_DEBUG=2 adds GPU-based validation (slow):
            // catches out-of-bounds descriptor/resource access and
            // resource-state races that the CPU layer cannot see.
            char level[8] = {};
            GetEnvironmentVariableA("MOTION_ENHANCER_D3D_DEBUG", level, sizeof(level));
            ComPtr<ID3D12Debug1> debug1;
            if (level[0] == '2' && SUCCEEDED(debug.As(&debug1))) {
                debug1->SetEnableGPUBasedValidation(TRUE);
                std::cout << "D3D12 GPU-based validation enabled.\n";
            }
        }
    }

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cerr << "Error: CreateDXGIFactory1 failed.\n";
        return false;
    }

    // Adapter selection. Auto takes the high-performance adapter: on a
    // laptop the OS default is the integrated GPU that drives the panel,
    // and the flow search belongs on the discrete card even at the price of
    // a cross-adapter copy per frame.
    {
        const DXGI_GPU_PREFERENCE gpuPreference = preference == GraphicsAdapterPreference::Integrated
            ? DXGI_GPU_PREFERENCE_MINIMUM_POWER : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(
                 i, gpuPreference, IID_PPV_ARGS(&m_adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc = {};
            m_adapter->GetDesc1(&desc);
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
                SUCCEEDED(D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                            __uuidof(ID3D12Device), nullptr))) {
                break;
            }
            m_adapter.Reset();
        }
    }
    if (!m_adapter) {
        std::cerr << "Error: no D3D12-capable graphics adapter found.\n";
        return false;
    }

    HRESULT hr = D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&m_device));
    if (FAILED(hr)) {
        std::cerr << "Error: D3D12CreateDevice failed (0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }
    if (m_debugLayer) m_device.As(&m_infoQueue);

    DXGI_ADAPTER_DESC1 desc = {};
    m_adapter->GetDesc1(&desc);
    {
        std::wstring name(desc.Description);
        m_adapterName.assign(name.begin(), name.end());
    }
    std::cout << "Adapter: " << m_adapterName << " (dedicated VRAM: "
              << desc.DedicatedVideoMemory / (1024ull * 1024ull) << " MB)\n";

    if (!createQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, m_presentQueue, L"PresentQueue") ||
        !createQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, m_flowQueue, L"FlowQueue")) {
        return false;
    }
    if (!createCaptureDevice() || !createSharedFences()) return false;
    return true;
}

bool GraphicsDevice::createQueue(D3D12_COMMAND_LIST_TYPE type,
                                 ComPtr<ID3D12CommandQueue>& queue, const wchar_t* name) {
    // Global-high priority keeps our work ahead of the captured application's
    // GPU load, but needs SeIncreaseBasePriorityPrivilege; fall back
    // gracefully.
    const INT priorities[] = {D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_HIGH,
                              D3D12_COMMAND_QUEUE_PRIORITY_HIGH,
                              D3D12_COMMAND_QUEUE_PRIORITY_NORMAL};
    for (INT priority : priorities) {
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = type;
        qd.Priority = priority;
        if (SUCCEEDED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.ReleaseAndGetAddressOf())))) {
            queue->SetName(name);
            return true;
        }
    }
    std::cerr << "Error: failed to create D3D12 command queue.\n";
    return false;
}

bool GraphicsDevice::createCaptureDevice() {
    D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if (m_debugLayer) flags |= D3D11_CREATE_DEVICE_DEBUG;
    HRESULT hr = D3D11CreateDevice(m_adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
                                   levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                                   m_device11.GetAddressOf(), &level, m_context11.GetAddressOf());
    if (FAILED(hr) && (flags & D3D11_CREATE_DEVICE_DEBUG)) {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDevice(m_adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
                               levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
                               m_device11.GetAddressOf(), &level, m_context11.GetAddressOf());
    }
    if (FAILED(hr)) {
        std::cerr << "Error: D3D11 capture device creation failed (0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }
    if (FAILED(m_device11.As(&m_device11_5)) || FAILED(m_context11.As(&m_context11_4))) {
        std::cerr << "Error: D3D11.4 (shared fences) is required for capture interop.\n";
        return false;
    }
    // The capture device never presents; multithread protection is not
    // needed because only the capture thread touches its immediate context.
    return true;
}

bool GraphicsDevice::createSharedFences() {
    // D3D11 -> D3D12
    if (FAILED(m_device11_5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_captureReady11)))) {
        std::cerr << "Error: shared capture fence creation failed.\n";
        return false;
    }
    HANDLE handle = nullptr;
    if (FAILED(m_captureReady11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle))) return false;
    const HRESULT open = m_device->OpenSharedHandle(handle, IID_PPV_ARGS(&m_captureReady12));
    CloseHandle(handle);
    if (FAILED(open)) {
        std::cerr << "Error: D3D12 could not open the shared capture fence.\n";
        return false;
    }
    m_captureReady12->SetName(L"CaptureReadyFence");

    // D3D12 -> D3D11 (offline readback only).
    if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_computeDone12)))) return false;
    handle = nullptr;
    if (FAILED(m_device->CreateSharedHandle(m_computeDone12.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;
    const HRESULT open11 = m_device11_5->OpenSharedFence(handle, IID_PPV_ARGS(&m_computeDone11));
    CloseHandle(handle);
    if (FAILED(open11)) return false;
    return true;
}

uint64_t GraphicsDevice::signalCaptureReady() {
    const uint64_t value = ++m_captureReadyValue;
    if (FAILED(m_context11_4->Signal(m_captureReady11.Get(), value))) return 0;
    // The copy + signal must be submitted before D3D12 can observe them.
    m_context11_4->Flush();
    return value;
}

void GraphicsDevice::waitComputeDone11(uint64_t value) {
    if (value == 0 || !m_computeDone11) return;
    if (m_computeDone11->GetCompletedValue() >= value) return;
    m_context11_4->Wait(m_computeDone11.Get(), value);
    m_context11_4->Flush();
}

bool GraphicsDevice::createSharedTexture(uint32_t width, uint32_t height, DXGI_FORMAT format,
                                         SharedTexture& out) {
    // D3D11 owns the allocation (MISC_SHARED | NTHANDLE, no keyed mutex) and
    // D3D12 opens the NT handle. Access is ordered by the shared fences.
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    HRESULT hr = m_device11->CreateTexture2D(&desc, nullptr, out.texture11.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Shared texture: D3D11 CreateTexture2D failed (0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }
    ComPtr<IDXGIResource1> resource;
    if (FAILED(out.texture11.As(&resource))) return false;
    HANDLE handle = nullptr;
    if (FAILED(resource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle))) {
        out.texture11.Reset();
        return false;
    }
    hr = m_device->OpenSharedHandle(handle, IID_PPV_ARGS(out.texture12.ReleaseAndGetAddressOf()));
    CloseHandle(handle);
    if (FAILED(hr)) {
        std::cerr << "Shared texture: D3D12 OpenSharedHandle failed (0x" << std::hex << hr << std::dec << ")\n";
        out.texture11.Reset();
        return false;
    }
    return true;
}

HRESULT GraphicsDevice::createTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                                        D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                        ComPtr<ID3D12Resource>& out, const wchar_t* name) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    const HRESULT hr = m_device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
        IID_PPV_ARGS(out.ReleaseAndGetAddressOf()));
    if (SUCCEEDED(hr) && name) out->SetName(name);
    return hr;
}

HRESULT GraphicsDevice::createBuffer(uint64_t bytes, D3D12_HEAP_TYPE heapType,
                                     D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags,
                                     ComPtr<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heapType;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                             IID_PPV_ARGS(out.ReleaseAndGetAddressOf()));
}

void GraphicsDevice::dumpInfoQueue(const char* context) {
    if (!m_infoQueue) return;
    const UINT64 count = m_infoQueue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        m_infoQueue->GetMessage(i, nullptr, &size);
        if (size == 0) continue;
        std::vector<uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(m_infoQueue->GetMessage(i, message, &size)) && message->pDescription) {
            std::cerr << "D3D12 validation (" << context << "): " << message->pDescription << "\n";
        }
    }
    m_infoQueue->ClearStoredMessages();
}
