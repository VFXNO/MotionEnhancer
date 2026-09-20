#pragma once

#include "ComputeKernels.h"
#include "GraphicsDevice.h"

#include <array>
#include <cstdint>

// One interpolated (or pass-through) output frame.
struct RenderJob {
    ID3D12Resource* frame0 = nullptr;
    uint64_t ready0 = 0;
    ID3D12Resource* frame1 = nullptr;
    uint64_t ready1 = 0;
    ID3D12Resource* forwardFlow = nullptr;
    ID3D12Resource* backwardFlow = nullptr;
    uint32_t flowBlockSize = 32;
    ID3D12Fence* flowFence = nullptr;   // GPU-side wait before the interpolate pass
    uint64_t flowFenceValue = 0;
    float alpha = 0.0f;
    bool single = false;               // show frame0 (alpha < 0.5) or frame1 as-is
};

struct PresentStats {
    uint64_t presented = 0;
    uint64_t clockFallbacks = 0;  // waits the compositor clock refused
    uint64_t lateFrames = 0;      // wake-ups more than 1.5 refresh intervals apart
    uint64_t resyncs = 0;         // display-clock resyncs (missed vsync)
    double maxIntervalMs = 0.0;
    double meanIntervalMs = 0.0;
};

// Native D3D12 flip-model presentation. One command list per refresh; the
// loop is paced solely by the swap chain's frame-latency waitable object
// (Present(1), latency 1), so every output frame is phase-locked to vsync.
class Presenter {
public:
    static constexpr UINT kBackBuffers = 3;

    ~Presenter();

    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, HWND window,
                    uint32_t width, uint32_t height);

    // Blocks until the next compositor tick (vsync) or one of the handles
    // is signaled. Returns WAIT_OBJECT_0 + count on a tick, WAIT_OBJECT_0 +
    // index for a handle, WAIT_TIMEOUT otherwise.
    DWORD waitForVsync(DWORD count, const HANDLE* handles, DWORD timeoutMs);
    // True when the swap chain can accept a frame without queuing behind
    // one the compositor has not consumed yet.
    bool canPresent() const {
        return !m_latencyWaitable || WaitForSingleObject(m_latencyWaitable, 0) == WAIT_OBJECT_0;
    }
    ID3D12Fence* fence() const { return m_fence.Get(); }
    double refreshRate() const { return m_refreshHz; }
    int64_t refreshInterval() const { return m_refreshInterval; }

    // Called once per latency-object wake-up. Returns the predicted display
    // time (wall clock, 100 ns) of the frame that will be rendered now: a
    // phase-locked loop on the wake cadence, so consecutive frames are
    // exactly one refresh apart even when the wake-up itself jitters.
    int64_t beginFrame(int64_t wakeWallTime);

    // Records, submits and presents. presentFenceValue completes the GPU
    // reads of every resource in the job.
    bool render(const RenderJob& job, uint64_t& presentFenceValue);

    PresentStats takeStats();

private:
    struct CommandSet {
        ComPtr<ID3D12CommandAllocator> allocator;
        uint64_t fence = 0;
    };
    bool waitFence(uint64_t value);
    void revealWindow();

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    HWND m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    ComPtr<IDXGISwapChain3> m_swapChain;
    std::array<ComPtr<ID3D12Resource>, kBackBuffers> m_backBuffers;
    HANDLE m_latencyWaitable = nullptr;
    ComPtr<ID3D12Resource> m_output;

    ComPtr<ID3D12GraphicsCommandList> m_list;
    std::array<CommandSet, kBackBuffers> m_commands;
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;
    DispatchContext m_dispatch;
    uint64_t m_frameCount = 0;
    bool m_revealed = false;
    HRESULT m_lastPresentStatus = S_OK;
    // DCompositionWaitForCompositorClock (dcomp.dll, Windows 10 2004+).
    using CompositorClockWait = DWORD(WINAPI*)(UINT, const HANDLE*, DWORD);
    CompositorClockWait m_compositorWait = nullptr;
    HMODULE m_dcomp = nullptr;
    uint64_t m_clockFallbacks = 0;

    // display clock
    double m_refreshHz = 60.0;
    int64_t m_refreshInterval = 166667;
    double m_intervalEstimate = 166667.0;
    int64_t m_lastPredicted = 0;
    int64_t m_lastWake = 0;

    PresentStats m_stats;
    double m_intervalSum = 0.0;
    uint64_t m_intervalCount = 0;
};
