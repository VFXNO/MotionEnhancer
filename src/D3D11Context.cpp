#include "D3D11Context.h"
#include "D3D12Context.h"
#include <d3dcompiler.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

D3D11Context::~D3D11Context() {
    if (m_frameLatencyWaitableObject) {
        CloseHandle(m_frameLatencyWaitableObject);
        m_frameLatencyWaitableObject = nullptr;
    }
}

bool D3D11Context::initialize(GraphicsAdapterPreference preference) {
    // D3D12 owns compute and presentation. A separate native D3D11 device on
    // the SAME adapter serves Windows Graphics Capture, whose frame contract
    // is ID3D11Texture2D. Frames cross into D3D12 through NT-handle shared
    // textures and a shared timeline fence.
    ComPtr<ID3D12Debug> d3d12Debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3d12Debug)))) {
        d3d12Debug->EnableDebugLayer();
        std::cout << "D3D12 debug layer enabled.\n";
    }

    ComPtr<IDXGIAdapter1> requestedAdapter;
    if (preference != GraphicsAdapterPreference::Auto) {
        ComPtr<IDXGIFactory1> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            SIZE_T selectedDedicatedMemory = 0;
            bool haveSelection = false;
            for (UINT index = 0;; ++index) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
                    break;
                }
                DXGI_ADAPTER_DESC1 desc = {};
                candidate->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
                    SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                                __uuidof(ID3D12Device), nullptr))) {
                    const bool better = !haveSelection ||
                        (preference == GraphicsAdapterPreference::Discrete
                            ? desc.DedicatedVideoMemory > selectedDedicatedMemory
                            : desc.DedicatedVideoMemory < selectedDedicatedMemory);
                    if (better) {
                        requestedAdapter = candidate;
                        selectedDedicatedMemory = desc.DedicatedVideoMemory;
                        haveSelection = true;
                    }
                }
            }
        }
        if (!requestedAdapter) {
            std::cerr << "Error: requested graphics adapter is unavailable.\n";
            return false;
        }
    }

    HRESULT hr = D3D12CreateDevice(
        requestedAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(d3d12Device.GetAddressOf()));
    if (SUCCEEDED(hr)) {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = d3d12Device->CreateCommandQueue(
            &queueDesc, IID_PPV_ARGS(d3d12Queue.GetAddressOf()));
        if (SUCCEEDED(hr)) {
            const LUID luid = d3d12Device->GetAdapterLuid();
            ComPtr<IDXGIFactory4> factory4;
            ComPtr<IDXGIAdapter1> adapter;
            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))) &&
                SUCCEEDED(factory4->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
                D3D_FEATURE_LEVEL featureLevels[] = {
                    D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0
                };
                D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
                UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
                flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
                hr = D3D11CreateDevice(
                    adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
                    featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION,
                    device.GetAddressOf(), &featureLevel, context.GetAddressOf());
            } else {
                hr = E_FAIL;
            }
            if (SUCCEEDED(hr)) {
                device.As(&m_device1);
                device.As(&m_device5);
                context.As(&m_context4);
                nativeD3D12 = std::make_shared<D3D12Context>();
                if (!nativeD3D12->initialize(d3d12Device.Get(), d3d12Queue.Get())) {
                    nativeD3D12.reset();
                }
            }
        }
        if (!nativeD3D12) {
            d3d12Queue.Reset();
            d3d12Device.Reset();
        }
    }

    if (!nativeD3D12) {
        // Native D3D12 is unavailable; build a plain D3D11 device for the
        // fallback compute graph and presentation.
        UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };
        D3D_FEATURE_LEVEL featureLevel;

        hr = D3D11CreateDevice(
            requestedAdapter.Get(),
            requestedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            createDeviceFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            device.GetAddressOf(),
            &featureLevel,
            context.GetAddressOf()
        );

        if (FAILED(hr)) {
            std::cerr << "Error: Failed to create D3D11 device (0x" << std::hex << hr << ")\n";
            return false;
        }
        device.As(&m_device1);
        device.As(&m_device5);
        context.As(&m_context4);
    }

    // Shared capture-ready fence: D3D11 signals after a capture copy, D3D12
    // queue-waits before consuming the frame. Required for the native path.
    if (m_device5 && m_context4 && d3d12Device) {
        if (SUCCEEDED(m_device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                             IID_PPV_ARGS(&m_captureReady11)))) {
            HANDLE sharedFence = nullptr;
            if (SUCCEEDED(m_captureReady11->CreateSharedHandle(
                    nullptr, GENERIC_ALL, nullptr, &sharedFence))) {
                const HRESULT openHr = d3d12Device->OpenSharedHandle(
                    sharedFence, IID_PPV_ARGS(&m_captureReady12));
                CloseHandle(sharedFence);
                if (FAILED(openHr)) {
                    m_captureReady11.Reset();
                }
            } else {
                m_captureReady11.Reset();
            }
        }
        // Shared compute-done fence: D3D12 signals after a submission, D3D11
        // GPU-waits before reading native results (offline readback).
        if (SUCCEEDED(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                               IID_PPV_ARGS(&m_computeDone12)))) {
            HANDLE sharedFence = nullptr;
            if (SUCCEEDED(d3d12Device->CreateSharedHandle(
                    m_computeDone12.Get(), nullptr, GENERIC_ALL, nullptr, &sharedFence))) {
                if (FAILED(m_device5->OpenSharedFence(
                        sharedFence, IID_PPV_ARGS(&m_computeDone11)))) {
                    m_computeDone12.Reset();
                }
                CloseHandle(sharedFence);
            } else {
                m_computeDone12.Reset();
            }
        }
    }
    if (nativeD3D12) {
        if (m_captureReady12) {
            nativeD3D12->setCaptureReadyFence(m_captureReady12.Get());
            nativeD3D12->setComputeDoneFence(m_computeDone12.Get());
            m_isD3D12Backend = true;
        } else {
            std::cerr << "Warning: shared D3D11/D3D12 fence unavailable; "
                         "using the D3D11 fallback graph.\n";
            nativeD3D12.reset();
            d3d12Queue.Reset();
            d3d12Device.Reset();
        }
    }

    std::cout << "Graphics backend: " << backendName() << "\n";
    if (d3d12Device) {
        // Print which physical adapter the pipeline runs on. In hybrid-graphics
        // laptops, forcing the dGPU via Windows Graphics Settings moves every
        // captured frame across PCIe twice (capture in, present out), which
        // stutters even when the frame counter looks healthy.
        const LUID luid = d3d12Device->GetAdapterLuid();
        ComPtr<IDXGIFactory4> luidFactory;
        ComPtr<IDXGIAdapter1> luidAdapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&luidFactory))) &&
            SUCCEEDED(luidFactory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&luidAdapter)))) {
            DXGI_ADAPTER_DESC1 desc = {};
            if (SUCCEEDED(luidAdapter->GetDesc1(&desc))) {
                std::wstring name(desc.Description);
                std::cout << "Adapter: "
                          << std::string(name.begin(), name.end())
                          << " (dedicated VRAM: " << desc.DedicatedVideoMemory / (1024ull * 1024ull)
                          << " MB)\n";
            }
        }
    }

    ComPtr<IDXGIDevice1> dxgiDevice;
    if (SUCCEEDED(device.As(&dxgiDevice))) {
        // Keep at most one completed frame queued for presentation.
        dxgiDevice->SetMaximumFrameLatency(1);
    }

    // Create Samplers
    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampDesc.MinLOD = 0;
    sampDesc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = device->CreateSamplerState(&sampDesc, linearSampler.GetAddressOf());
    if (FAILED(hr)) return false;

    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    hr = device->CreateSamplerState(&sampDesc, pointSampler.GetAddressOf());
    if (FAILED(hr)) return false;

    // Create Constant Buffer
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = ((sizeof(ShaderConstants) + 15) / 16) * 16;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    hr = device->CreateBuffer(&cbDesc, nullptr, constantBuffer.GetAddressOf());
    if (FAILED(hr)) return false;

    return true;
}

bool D3D11Context::nativePresentation() const {
    return nativeD3D12 && nativeD3D12->isReady();
}

bool D3D11Context::createSwapChain(HWND hwnd, uint32_t width, uint32_t height) {
    if (nativePresentation()) {
        return nativeD3D12->createSwapChain(hwnd, width, height);
    }
    if (!device) return false;

    ComPtr<IDXGIDevice> dxgiDevice;
    device.As(&dxgiDevice);

    ComPtr<IDXGIAdapter> adapter;
    dxgiDevice->GetAdapter(adapter.GetAddressOf());

    ComPtr<IDXGIFactory2> factory;
    adapter->GetParent(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(factory.GetAddressOf()));

    DXGI_SWAP_CHAIN_DESC1 scDesc = {};
    scDesc.Width = width;
    scDesc.Height = height;
    // Match the compute output format. BGRA typed UAVs are not supported on
    // all D3D11 hardware, while RGBA8 is guaranteed at feature level 11.
    scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scDesc.SampleDesc.Count = 1;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    scDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    HRESULT hr = factory->CreateSwapChainForHwnd(
        device.Get(),
        hwnd,
        &scDesc,
        nullptr,
        nullptr,
        swapChain.GetAddressOf()
    );

    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create swap chain for HWND (0x" << std::hex << hr << ")\n";
        return false;
    }

    ComPtr<IDXGISwapChain2> swapChain2;
    if (SUCCEEDED(swapChain.As(&swapChain2))) {
        swapChain2->SetMaximumFrameLatency(1);
        m_frameLatencyWaitableObject = swapChain2->GetFrameLatencyWaitableObject();
    }

    return true;
}

void D3D11Context::resizeSwapChain(uint32_t width, uint32_t height) {
    if (nativePresentation()) {
        nativeD3D12->resizeSwapChain(width, height);
        return;
    }
    if (swapChain && width > 0 && height > 0) {
        swapChain->ResizeBuffers(
            2, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT
        );
    }
}

HANDLE D3D11Context::frameLatencyHandle() const {
    if (nativePresentation()) {
        return nativeD3D12->frameLatencyHandle();
    }
    return m_frameLatencyWaitableObject;
}

HRESULT D3D11Context::presentSwapChain() {
    if (nativePresentation()) {
        const HRESULT hr = nativeD3D12->present(0);
        if (FAILED(hr)) {
            std::cerr << "Presenter: D3D12 Present failed (0x" << std::hex << hr << std::dec << ")\n";
            if (d3d12Device) {
                std::cerr << "Presenter: device removed reason 0x"
                          << std::hex << d3d12Device->GetDeviceRemovedReason() << std::dec << "\n";
            }
        }
        return hr;
    }
    return swapChain ? swapChain->Present(0, 0) : E_FAIL;
}

bool D3D11Context::createSharedTexture2D(
    uint32_t width, uint32_t height,
    DXGI_FORMAT format,
    D3D11_BIND_FLAG bindFlags,
    ComPtr<ID3D11Texture2D>& texture,
    ComPtr<ID3D12Resource>& texture12
) {
    (void)bindFlags;
    if (!device || !d3d12Device || width == 0 || height == 0) return false;

    // Verified working sequence: D3D11 owns the allocation with
    // MISC_SHARED|SHARED_NTHANDLE (no keyed mutex; keyed-mutex textures
    // silently drop D3D11 GPU work that has not been AcquireSync'ed).
    // D3D12 opens the NT handle and both sides order access with the shared
    // fences. BIND_UNORDERED_ACCESS keeps the resource UAV-capable in D3D12.
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, texture.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Shared texture: D3D11 CreateTexture2D failed (0x"
                  << std::hex << hr << std::dec << ")\n";
        return false;
    }

    ComPtr<IDXGIResource1> resource1;
    if (FAILED(texture.As(&resource1))) return false;
    HANDLE shared = nullptr;
    if (FAILED(resource1->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &shared))) {
        std::cerr << "Shared texture: CreateSharedHandle failed\n";
        texture.Reset();
        return false;
    }
    hr = d3d12Device->OpenSharedHandle(shared, IID_PPV_ARGS(texture12.ReleaseAndGetAddressOf()));
    CloseHandle(shared);
    if (FAILED(hr)) {
        std::cerr << "Shared texture: D3D12 OpenSharedHandle failed (0x"
                  << std::hex << hr << std::dec << ")\n";
        texture.Reset();
        return false;
    }
    m_sharedRecords[texture.Get()] = {texture12, 0};
    return true;
}

uint64_t D3D11Context::signalCaptureReady(ID3D11Texture2D* sharedTexture) {
    if (!m_context4 || !m_captureReady11) return 0;
    const uint64_t value = ++m_captureReadyValue;
    if (FAILED(m_context4->Signal(m_captureReady11.Get(), value))) return 0;
    // The signal must reach the GPU before D3D12 consumes the frame.
    m_context4->Flush();
    auto it = m_sharedRecords.find(sharedTexture);
    if (it != m_sharedRecords.end()) it->second.readyValue = value;
    return value;
}

bool D3D11Context::resolveSharedFrame(ID3D11Texture2D* sharedTexture,
                                      ComPtr<ID3D12Resource>& texture12, uint64_t& readyValue) {
    auto it = m_sharedRecords.find(sharedTexture);
    if (it == m_sharedRecords.end() || !it->second.resource) return false;
    texture12 = it->second.resource;
    readyValue = it->second.readyValue;
    return true;
}

void D3D11Context::waitD3D12(uint64_t submittedValue) {
    if (!m_context4 || !m_computeDone11 || submittedValue == 0) return;
    if (m_computeDone11->GetCompletedValue() >= submittedValue) return;
    m_context4->Wait(m_computeDone11.Get(), submittedValue);
    m_context4->Flush();
}

void D3D11Context::updateConstants(const ShaderConstants& constants) {
    if (!context || !constantBuffer) return;

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context->Map(constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(mapped.pData, &constants, sizeof(ShaderConstants));
        context->Unmap(constantBuffer.Get(), 0);
    }

    ID3D11Buffer* cbs[] = { constantBuffer.Get() };
    context->CSSetConstantBuffers(0, 1, cbs);

    ID3D11SamplerState* samplers[] = { linearSampler.Get(), pointSampler.Get() };
    context->CSSetSamplers(0, 2, samplers);
}

bool D3D11Context::createTexture2D(
    uint32_t width, uint32_t height,
    DXGI_FORMAT format,
    ComPtr<ID3D11Texture2D>& texture,
    ComPtr<ID3D11ShaderResourceView>& srv,
    ComPtr<ID3D11UnorderedAccessView>& uav,
    bool isRenderTarget
) {
    if (!device || width == 0 || height == 0) return false;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (isRenderTarget) {
        desc.BindFlags |= D3D11_BIND_RENDER_TARGET;
    }

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, texture.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    hr = device->CreateShaderResourceView(texture.Get(), &srvDesc, srv.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return false;

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    hr = device->CreateUnorderedAccessView(texture.Get(), &uavDesc, uav.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return false;

    return true;
}

ComPtr<ID3D11ComputeShader> D3D11Context::compileComputeShader(
    const std::string& shaderPath,
    const std::string& entryPoint
) {
    if (!device) return nullptr;

    char exePathBuf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exePathBuf, MAX_PATH);
    std::filesystem::path exeDir = std::filesystem::path(exePathBuf).parent_path();

    // Search common paths for shader file
    std::vector<std::filesystem::path> searchPaths = {
        std::filesystem::path(shaderPath),
        exeDir / shaderPath,
        exeDir / "shaders" / shaderPath,
        exeDir / ".." / "shaders" / shaderPath,
        exeDir / ".." / ".." / "shaders" / shaderPath,
        std::filesystem::path("shaders") / shaderPath,
        std::filesystem::path("..") / "shaders" / shaderPath,
        std::filesystem::path("..") / ".." / "shaders" / shaderPath
    };

    std::filesystem::path foundPath;
    for (const auto& p : searchPaths) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec) && !std::filesystem::is_directory(p, ec)) {
            foundPath = std::filesystem::absolute(p, ec);
            break;
        }
    }

    if (foundPath.empty()) {
        std::cerr << "Error: Cannot find shader file '" << shaderPath << "'. Searched paths:\n";
        for (const auto& p : searchPaths) {
            std::cerr << "  - " << p.string() << "\n";
        }
        return nullptr;
    }

    std::ifstream file(foundPath);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open shader file '" << foundPath.string() << "'\n";
        return nullptr;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Release builds compile shaders with DXC in CMake. Loading the artifact
    // avoids a runtime compiler dependency and makes DXC the authoritative
    // HLSL compiler. The legacy compiler remains a source-tree fallback for
    // ad-hoc builds where DXC is unavailable.
    std::filesystem::path compiledPath = foundPath;
    compiledPath.replace_extension(".cso");
    std::ifstream compiled(compiledPath, std::ios::binary);
    if (compiled) {
        std::vector<char> bytecode((std::istreambuf_iterator<char>(compiled)), {});
        ComPtr<ID3D11ComputeShader> shader;
        if (SUCCEEDED(device->CreateComputeShader(
                bytecode.data(), bytecode.size(), nullptr, shader.GetAddressOf()))) {
            return shader;
        }
        // Some redistributed DXC builds promote cs_5_0 to DXIL. D3D11's
        // CreateComputeShader accepts DXBC, not that DXIL container. Keep the
        // DXC artifact for native D3D12 consumers, then use the source fallback
        // so the D3D11 compatibility path remains runnable.
        std::cerr << "Warning: DXC artifact is DXIL and was rejected by the "
                  << "D3D11 shader loader; compiling DXBC compatibility "
                  << "fallback for " << compiledPath.filename().string() << "\n";
    }

    UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    ComPtr<ID3DBlob> codeBlob;
    ComPtr<ID3DBlob> errorBlob;

    std::string foundPathStr = foundPath.string();
    HRESULT hr = D3DCompile(
        source.c_str(),
        source.length(),
        foundPathStr.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint.c_str(),
        "cs_5_0",
        compileFlags,
        0,
        codeBlob.GetAddressOf(),
        errorBlob.GetAddressOf()
    );

    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader compile error in " << foundPathStr << ":\n"
                      << static_cast<const char*>(errorBlob->GetBufferPointer()) << "\n";
        }
        return nullptr;
    }

    ComPtr<ID3D11ComputeShader> shader;
    hr = device->CreateComputeShader(
        codeBlob->GetBufferPointer(),
        codeBlob->GetBufferSize(),
        nullptr,
        shader.GetAddressOf()
    );

    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create compute shader from " << foundPathStr << "\n";
        return nullptr;
    }

    return shader;
}
