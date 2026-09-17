#include "D3D11Context.h"
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

bool D3D11Context::initialize() {
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D_FEATURE_LEVEL featureLevel;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
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

bool D3D11Context::createSwapChain(HWND hwnd, uint32_t width, uint32_t height) {
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
    if (swapChain && width > 0 && height > 0) {
        swapChain->ResizeBuffers(
            2, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT
        );
    }
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
