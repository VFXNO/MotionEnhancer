#pragma once

#include <d3d11.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

class D3D12Context;

enum class GraphicsAdapterPreference {
    Auto,
    Integrated,
    Discrete
};

using Microsoft::WRL::ComPtr;

struct ShaderConstants {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t levelWidth = 0;
    uint32_t levelHeight = 0;

    int32_t  searchRadius = 4;
    int32_t  blockSize = 8;
    int32_t  levelIndex = 0;
    int32_t  totalLevels = 8;

    float    timeT = 0.5f;
    float    smoothnessWeight = 0.0005f;
    uint32_t parentBlockSize = 8;
    float    flowScale = 1.0f;
};

// Top-level graphics owner. One native D3D12 device/queue (compute + present)
// and one native D3D11 device on the same adapter (WGC capture + capture
// copies). Frames cross the API boundary through NT-handle shared textures
// and a shared timeline fence; D3D11On12 is not used.
class D3D11Context {
public:
    ComPtr<ID3D11Device>        device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain1>     swapChain;   // D3D11 fallback presentation only
    ComPtr<ID3D11SamplerState>  linearSampler;
    ComPtr<ID3D11SamplerState>  pointSampler;
    ComPtr<ID3D11Buffer>        constantBuffer;

    ComPtr<ID3D12Device>        d3d12Device;
    ComPtr<ID3D12CommandQueue>  d3d12Queue;
    std::shared_ptr<D3D12Context> nativeD3D12;

    D3D11Context() = default;
    ~D3D11Context();

    bool initialize(GraphicsAdapterPreference preference = GraphicsAdapterPreference::Auto);
    bool isD3D12Backend() const { return m_isD3D12Backend; }
    bool nativePresentation() const;
    const char* backendName() const { return m_isD3D12Backend ? "D3D12 (shared-texture interop)" : "D3D11 fallback"; }
    bool createSwapChain(HWND hwnd, uint32_t width, uint32_t height);
    void resizeSwapChain(uint32_t width, uint32_t height);
    HANDLE frameLatencyHandle() const;
    HRESULT presentSwapChain();
    void updateConstants(const ShaderConstants& constants);

    // Creates a D3D11 texture shareable with D3D12 and opens it as an
    // ID3D12Resource on the native D3D12 device (same adapter required).
    bool createSharedTexture2D(
        uint32_t width, uint32_t height,
        DXGI_FORMAT format,
        D3D11_BIND_FLAG bindFlags,
        ComPtr<ID3D11Texture2D>& texture,
        ComPtr<ID3D12Resource>& texture12
    );
    // Signals the shared capture-ready fence after a D3D11-side copy into a
    // shared texture; returns the fence value D3D12 must wait on.
    uint64_t signalCaptureReady(ID3D11Texture2D* sharedTexture);
    // Resolves a previously shared texture into {resource, readyValue}.
    bool resolveSharedFrame(ID3D11Texture2D* sharedTexture,
                            ComPtr<ID3D12Resource>& texture12, uint64_t& readyValue);
    // GPU-orders the D3D11 queue after the given D3D12 submission value.
    void waitD3D12(uint64_t submittedValue);

    // Texture creation utilities (D3D11 fallback graph)
    bool createTexture2D(
        uint32_t width, uint32_t height,
        DXGI_FORMAT format,
        ComPtr<ID3D11Texture2D>& texture,
        ComPtr<ID3D11ShaderResourceView>& srv,
        ComPtr<ID3D11UnorderedAccessView>& uav,
        bool isRenderTarget = false
    );

    // Shader compilation
    ComPtr<ID3D11ComputeShader> compileComputeShader(
        const std::string& shaderPath,
        const std::string& entryPoint = "CSMain"
    );

private:
    struct SharedRecord {
        ComPtr<ID3D12Resource> resource;
        uint64_t readyValue = 0;
    };
    HANDLE m_frameLatencyWaitableObject = nullptr;
    bool m_isD3D12Backend = false;
    ComPtr<ID3D11Device1> m_device1;
    ComPtr<ID3D11Device5> m_device5;
    ComPtr<ID3D11DeviceContext4> m_context4;
    ComPtr<ID3D11Fence> m_captureReady11;
    ComPtr<ID3D12Fence> m_captureReady12;
    uint64_t m_captureReadyValue = 0;
    ComPtr<ID3D12Fence> m_computeDone12;
    ComPtr<ID3D11Fence> m_computeDone11;
    std::unordered_map<void*, SharedRecord> m_sharedRecords;
};
