#pragma once

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

enum class GraphicsAdapterPreference {
    Auto,
    Integrated,
    Discrete
};

// Constant buffer layout shared by every compute pass. Must match
// shaders/Common.hlsli exactly (three 16-byte rows).
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

// A texture that lives in both APIs: written by the D3D11 capture device,
// read by D3D12. readyFence/readyValue publish the last D3D11 write.
struct SharedTexture {
    ComPtr<ID3D11Texture2D> texture11;
    ComPtr<ID3D12Resource>  texture12;
};

// Owns the D3D12 device (all compute + presentation) and a D3D11 device on
// the same adapter (Windows Graphics Capture only). The two are ordered by
// a shared timeline fence: D3D11 signals after each capture copy, D3D12
// queues wait before reading.
//
// Two D3D12 queues: presentQueue (interpolate + present, one command list per
// vsync) and flowQueue (pyramids and optical flow, submitted when a frame
// arrives). They are ordered through flowFence.
class GraphicsDevice {
public:
    ~GraphicsDevice();

    bool initialize(GraphicsAdapterPreference preference);

    ID3D12Device* device() const { return m_device.Get(); }
    ID3D12CommandQueue* presentQueue() const { return m_presentQueue.Get(); }
    ID3D12CommandQueue* flowQueue() const { return m_flowQueue.Get(); }
    ID3D11Device* captureDevice() const { return m_device11.Get(); }
    ID3D11DeviceContext* captureContext() const { return m_context11.Get(); }
    IDXGIAdapter1* adapter() const { return m_adapter.Get(); }
    const std::string& adapterName() const { return m_adapterName; }

    // Shared fence: D3D11 -> D3D12 hand-off of capture writes.
    ID3D12Fence* captureReadyFence12() const { return m_captureReady12.Get(); }
    // Signals the capture-ready fence on the D3D11 context after queued
    // work and flushes. Returns the value D3D12 must wait on.
    uint64_t signalCaptureReady();
    // Shared fence: D3D12 -> D3D11. A D3D12 queue signals computeDoneFence12
    // with a value; waitComputeDone11 GPU-orders the D3D11 context after it
    // (offline readback of native results through a shared texture).
    ID3D12Fence* computeDoneFence12() const { return m_computeDone12.Get(); }
    void waitComputeDone11(uint64_t value);

    bool createSharedTexture(uint32_t width, uint32_t height, DXGI_FORMAT format,
                             SharedTexture& out);
    HRESULT createTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                            D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                            ComPtr<ID3D12Resource>& out, const wchar_t* name = nullptr);
    HRESULT createBuffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                         D3D12_RESOURCE_FLAGS flags, ComPtr<ID3D12Resource>& out);

    // Prints and clears any stored debug-layer messages (no-op without the
    // debug layer).
    void dumpInfoQueue(const char* context);
    bool debugLayerEnabled() const { return m_debugLayer; }

private:
    bool createCaptureDevice();
    bool createSharedFences();
    bool createQueue(D3D12_COMMAND_LIST_TYPE type, ComPtr<ID3D12CommandQueue>& queue,
                     const wchar_t* name);

    ComPtr<IDXGIAdapter1>        m_adapter;
    ComPtr<ID3D12Device>         m_device;
    ComPtr<ID3D12InfoQueue>      m_infoQueue;
    ComPtr<ID3D12CommandQueue>   m_presentQueue;
    ComPtr<ID3D12CommandQueue>   m_flowQueue;

    ComPtr<ID3D11Device>         m_device11;
    ComPtr<ID3D11Device5>        m_device11_5;
    ComPtr<ID3D11DeviceContext>  m_context11;
    ComPtr<ID3D11DeviceContext4> m_context11_4;

    ComPtr<ID3D11Fence>          m_captureReady11;
    ComPtr<ID3D12Fence>          m_captureReady12;
    uint64_t                     m_captureReadyValue = 0;
    ComPtr<ID3D12Fence>          m_computeDone12;
    ComPtr<ID3D11Fence>          m_computeDone11;

    std::string                  m_adapterName;
    bool                         m_debugLayer = false;
};
