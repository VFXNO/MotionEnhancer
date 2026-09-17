#pragma once

#include <d3d11.h>
#include <dxgi1_3.h>
#include <wrl/client.h>
#include <string>
#include <vector>

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
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
};

class D3D11Context {
public:
    ComPtr<ID3D11Device>        device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain1>     swapChain;
    ComPtr<ID3D11SamplerState>  linearSampler;
    ComPtr<ID3D11SamplerState>  pointSampler;
    ComPtr<ID3D11Buffer>        constantBuffer;

    D3D11Context() = default;
    ~D3D11Context();

    bool initialize();
    bool createSwapChain(HWND hwnd, uint32_t width, uint32_t height);
    void resizeSwapChain(uint32_t width, uint32_t height);
    HANDLE frameLatencyHandle() const { return m_frameLatencyWaitableObject; }
    void updateConstants(const ShaderConstants& constants);

    // Texture creation utilities
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
    HANDLE m_frameLatencyWaitableObject = nullptr;
};
