#pragma once

#include "ComputeKernels.h"
#include "GraphicsDevice.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Live debug view of the presented frames (see DebugView() in
// Common.hlsli): 0 = off, 1 = motion, 2 = static-UI evidence, 3 = occlusion
// trust, 4 = vector arrows. Ctrl+Alt+F3 in the overlay cycles it; read on
// every frame.
constexpr int kDebugViewCount = 5;
void cycleDebugView();
int debugView();
// Ctrl+Alt+F2 in the overlay: saves, into captures\<date-time>\ next to the
// executable, the last kReplayShown presented frames (shown_###_..._t<t>.png,
// exactly what was on screen, generated frames included) and the last
// kReplayGame game frames they were made from (game_###.png). The presenter
// keeps both in a replay buffer in video memory all the time (one copy per
// refresh, about 0.05 ms), so the key saves the moment just before it was
// pressed and nothing on the way to the screen waits for the save.
constexpr int kReplayShown = 20;
constexpr int kReplayGame = 8;
void requestCaptureDump();

// Records the interpolation of one output frame into `list`: for a
// two-field pair (constants.flowScale == 0) the motion projection to
// constants.timeT first (ProjectClearCS, ProjectSplatCS into `projection`),
// then InterpolateCS. `projection` comes from createProjection(). Bits of
// implicitInputs mark frame0, frame1 (0x1, 0x2) and uiMask (0x10) as read
// through implicit promotion, as DispatchContext::dispatch takes them.
bool recordInterpolate(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const ComputeKernels& kernels,
                       const ShaderConstants& constants, ID3D12Resource* frame0, ID3D12Resource* frame1,
                       ID3D12Resource* forward, ID3D12Resource* backward, ID3D12Resource* uiMask,
                       ID3D12Resource* projection, ID3D12Resource* output, uint32_t implicitInputs);
// The projection for frames of width x height, at any flow grid of 2 px or
// coarser.
bool createProjection(GraphicsDevice& device, uint32_t width, uint32_t height, ComPtr<ID3D12Resource>& projection);

// One interpolated (or pass-through) output frame.
struct RenderJob {
    ID3D12Resource* frame0 = nullptr;
    uint64_t ready0 = 0;
    ID3D12Resource* frame1 = nullptr;
    uint64_t ready1 = 0;
    ID3D12Resource* forwardFlow = nullptr;
    ID3D12Resource* backwardFlow = nullptr;
    uint32_t flowBlockSize = 32;
    bool singleFlowField = false;      // FlowEngine::Entry::singleField
    ID3D12Resource* uiMask = nullptr;  // FlowEngine::Entry::uiMaskInput(); null = no UI
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
    ComPtr<ID3D12Resource> m_projection;

    // Replay buffer for requestCaptureDump(): the last presented frames and
    // the game frames they were made from, in video memory. Frozen while a
    // save is reading it back (m_replayBusy, cleared by the save thread).
    struct ReplayFrame {
        ComPtr<ID3D12Resource> texture;
        float alpha = 0.0f;
        bool real = false;
        uint64_t ready0 = 0;   // the game frames it was made from (capture ready values)
        uint64_t ready1 = 0;
    };
    std::array<ReplayFrame, kReplayShown> m_replayShown;
    std::array<ReplayFrame, kReplayGame> m_replayGame;   // ready0 = this frame's ready value
    uint64_t m_replayShownCount = 0;
    uint64_t m_replayGameCount = 0;
    uint64_t m_replayLastGame = 0;
    std::shared_ptr<std::atomic<bool>> m_replayBusy;
    void recordReplay(const RenderJob& job);
    void startReplaySave(uint64_t presentValue);

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
