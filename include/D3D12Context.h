#pragma once

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#endif

using Microsoft::WRL::ComPtr;

struct ShaderConstants;

// A frame produced by the D3D11 capture device and shared into D3D12 through
// an NT-handle texture. readyValue is the value signaled on the shared
// capture-ready fence after the D3D11 copy; the D3D12 queue waits on it
// before reading resource. readyValue == 0 skips the fence wait (self-test).
struct SharedFrame {
    ID3D12Resource* resource = nullptr;
    uint64_t readyValue = 0;
};

// Native D3D12 submission context. WGC remains on a native D3D11 capture
// device; captured frames are shared via NT-handle textures and a shared
// timeline fence. Presentation uses a native D3D12 flip-model swapchain.
class D3D12Context {
public:
    D3D12Context() = default;
    ~D3D12Context();

    bool initialize(ID3D12Device* device, ID3D12CommandQueue* queue);
    bool isReady() const { return !m_failed && m_device && m_queue && m_commandList && m_pipeline[5]; }

    // Shared capture-ready fence (D3D11 signals, D3D12 queue waits).
    void setCaptureReadyFence(ID3D12Fence* fence) { m_captureReady = fence; }
    // Shared compute-done fence (D3D12 signals with the submit value; D3D11
    // GPU-waits before consuming results, e.g. offline readback).
    void setComputeDoneFence(ID3D12Fence* fence) { m_computeDone = fence; }
    uint64_t lastSubmittedValue() const { return m_fenceValue; }

    bool resizeResources(uint32_t width, uint32_t height, int totalLevels);

    // Binds the shared frames directly, then estimates flow (AMD FFX when
    // available, MSAD shader graph otherwise). Returns
    // after the prepare is SUBMITTED; call pairOutputReady()/waitPairOutput()
    // before dispatchInterpolate consumes the pair.
    bool dispatchFramePair(const SharedFrame& frame0, const SharedFrame& frame1,
                           uint64_t frame0Index, uint64_t frame1Index,
                           int totalLevels, int minRefineLevel,
                           int coarseRadius, int refineRadius,
                           float smoothnessWeight);
    // True once the submitted pair prepare has finished on the GPU. Never
    // blocks; polled by the presenter to defer a tick without CPU waits.
    bool pairOutputReady() const;
    // Blocks until the submitted pair prepare has finished (offline path).
    bool waitPairOutput();
    // Interpolates the previously prepared pair into m_output and copies the
    // result into target (a swap-chain back buffer or a shared texture).
    bool dispatchInterpolate(ID3D12Resource* target, float timeT);
    // Presents the retained last output again (latency-token Presents must
    // never show an unrendered back buffer).
    bool rePresentLastOutput();
    // Runtime flow-engine selection: false forces the MSAD graph even when
    // the FFX context exists. Switching re-primes the FFX temporal history.
    void setPreferFfx(bool preferFfx) {
        if (m_preferFfx == preferFfx) return;
        m_preferFfx = preferFfx;
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
        m_usingFfxFlow = false;
        if (m_ffxReady) {
            m_ffxWarmed = false;
            m_ffxLastCurrentIndex = UINT64_MAX;
            m_ffxLastFrame1 = nullptr;
        }
#endif
    }
    bool prefersFfx() const { return m_preferFfx; }

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    // True after the first successful FFX dispatch of the current stream.
    bool usingFfxOpticalFlow() const { return m_usingFfxFlow; }
    // True while the FFX context exists; the MSAD graph is the fallback.
    bool ffxAvailable() const { return m_ffxReady; }
#else
    bool usingFfxOpticalFlow() const { return false; }
    bool ffxAvailable() const { return false; }
#endif

    // Runs PresentFrameCS on D3D12 and copies the result into target.
    bool dispatchPresentFrame(const SharedFrame& source, ID3D12Resource* target,
                              const ShaderConstants& constants);
    // Runs bounded one-group dispatches plus the shared-texture interop path.
    bool runDiagnostics(class D3D11Context* context);

    // Native D3D12 presentation surface.
    bool createSwapChain(HWND hwnd, uint32_t width, uint32_t height);
    void resizeSwapChain(uint32_t width, uint32_t height);
    HANDLE frameLatencyHandle() const { return m_frameLatencyWaitableObject; }
    ComPtr<ID3D12Resource> backBuffer(UINT index) const {
        return index < 2 ? m_backBuffer[index] : nullptr;
    }
    // The buffer the next Present will flip to; write content there.
    UINT currentBackBufferIndex() const {
        return m_swapChain3 ? m_swapChain3->GetCurrentBackBufferIndex() : 0;
    }
    HRESULT present(UINT syncInterval) {
        return m_swapChain ? m_swapChain->Present(syncInterval, 0) : E_FAIL;
    }
    HRESULT createBuffer(const D3D12_RESOURCE_DESC& desc,
                         D3D12_RESOURCE_STATES initialState,
                         ComPtr<ID3D12Resource>& resource);
    HRESULT createTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                            D3D12_RESOURCE_FLAGS flags,
                            D3D12_RESOURCE_STATES initialState,
                            ComPtr<ID3D12Resource>& resource);
    void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    // Persistent upload ring for per-pass constants; one slot per dispatch.
    bool createConstantRing();
    ID3D12Resource* constantRingSlot(const ShaderConstants& constants,
                                     D3D12_GPU_VIRTUAL_ADDRESS& gpuAddress);

private:
    bool loadPipeline();
    bool loadShaderBytecode(const std::string& name, std::vector<uint8_t>& bytes);
    bool beginCommands(bool asyncPair = false, int presentationIndex = -1);
    bool submit(uint64_t& fenceValue);
    bool waitForFence(uint64_t fenceValue);
    void dumpInfoQueue(const char* context);
    bool copyInput(const SharedFrame& frame, ComPtr<ID3D12Resource>& input);
    void recordCaptureWaits(const SharedFrame& frame);
    bool dispatchPass(int pipelineIndex, const std::array<ID3D12Resource*, 4>& inputs,
                      ID3D12Resource* output, const ShaderConstants& constants,
                       uint32_t groupsX, uint32_t groupsY);
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    bool initializeFfxOpticalFlow();
    void destroyFfxOpticalFlow();
    bool dispatchFfxFramePair(const SharedFrame& frame0, const SharedFrame& frame1,
                              uint64_t frame1Index);
#endif
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12InfoQueue> m_infoQueue;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12Fence> m_captureReady;
    ComPtr<ID3D12Fence> m_computeDone;
    std::vector<uint64_t> m_pendingReadyWaits;
    ComPtr<ID3D12CommandAllocator> m_allocator;
    // Presentation has one allocator and descriptor/constant region per back
    // buffer. Pair preparation has a third isolated submission region.
    ComPtr<ID3D12CommandAllocator> m_presentAllocator1;
    ComPtr<ID3D12CommandAllocator> m_ffxAllocator;
    uint64_t m_allocatorFence[3] = {};
    int m_allocatorKind = 0;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;
    // Fence value of the in-flight (asynchronous) frame-pair prepare; 0 when
    // no pair output is pending. dispatchInterpolate waits for it GPU-side and
    // the presenter polls it to defer a tick without blocking the CPU.
    uint64_t m_pairPendingFenceValue = 0;
    LARGE_INTEGER m_lastStallLogQpc = {};
    bool m_failed = false;

    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    std::array<ComPtr<ID3D12PipelineState>, 7> m_pipeline;
    // Persistently mapped 256-byte-slot constant ring. Each in-flight command
    // class owns a disjoint region so CPU recording never overwrites GPU reads.
    static constexpr UINT kConstantRingSlots = 64;
    static constexpr UINT kPresent1RingCursor = 16;
    static constexpr UINT kPairRingCursor = 32;
    static constexpr UINT kPresent1DescriptorBase = 128;
    static constexpr UINT kPairDescriptorBase = 256;
    static constexpr UINT kDescriptorHeapSize = 512;
    ComPtr<ID3D12Resource> m_constantRing;
    uint8_t* m_constantRingCpu = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_constantRingGpu = 0;
    UINT m_constantRingCursor = 0;
    UINT m_descriptorSize = 0;
    UINT m_descriptorCursor = 0;

    struct NativeLevel {
        ComPtr<ID3D12Resource> texture;
        ComPtr<ID3D12Resource> flow;
        ComPtr<ID3D12Resource> filteredFlow;
    };
    std::array<NativeLevel, 8> m_pyr0;
    std::array<NativeLevel, 8> m_pyr1;
    ComPtr<ID3D12Resource> m_output;
    bool m_outputValid = false;
    // Zero-copy: the pair's shared capture resources are bound directly as
    // SRVs; they stay stable until the pair changes because the presenter
    // never retires a slot cached as previous/current.
    std::array<SharedFrame, 2> m_pairFrame;
    std::array<ComPtr<ID3D12Resource>, 2> m_frameInput;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    int m_levels = 0;

    ComPtr<IDXGISwapChain1> m_swapChain;
    ComPtr<IDXGISwapChain3> m_swapChain3;
    std::array<ComPtr<ID3D12Resource>, 2> m_backBuffer;
    HANDLE m_frameLatencyWaitableObject = nullptr;

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    FfxOpticalflowContext m_ffxContext = {};
    FfxInterface m_ffxInterface = {};
    std::vector<uint8_t> m_ffxScratch;
    // FFX emits current-to-previous vectors on its native 8x8 grid. The
    // backward field is converted directly and the forward field is its
    // explicit negated approximation.
    ComPtr<ID3D12Resource> m_ffxBackwardFlow;
    ComPtr<ID3D12Resource> m_ffxForwardConvertedFlow;
    ComPtr<ID3D12Resource> m_ffxBackwardConvertedFlow;
    ComPtr<ID3D12Resource> m_ffxScd;
    uint64_t m_ffxLastCurrentIndex = UINT64_MAX;
    // Borrowed reference to the last dispatched frame1 resource; the FFX
    // temporal chain continues only when the next frame0 is this resource.
    ID3D12Resource* m_ffxLastFrame1 = nullptr;
    bool m_ffxReady = false;
    bool m_ffxWarmed = false;
    bool m_usingFfxFlow = false;
#endif
    bool m_preferFfx = true;
};
