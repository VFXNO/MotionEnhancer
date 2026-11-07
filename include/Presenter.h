#pragma once

#include "ComputeKernels.h"
#include "GraphicsDevice.h"
#include "UiRegions.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Live debug view of the presented frames (see DebugView() in
// Common.hlsli): 0 = off, 1 = motion, 2 = static-UI evidence, 3 = occlusion
// trust, 4 = vector arrows, 5 = AI depth (DepthViewCS), 6 = raw AI blend mask
// (white = frame0, black = frame1, purple checkerboard = inactive/no pair).
// 7 = AI UI regions (cyan boxes, yellow evidence, green protected pixels).
// Ctrl+Alt+F3 in the overlay cycles it; read on every frame.
constexpr int kDebugViewCount = 8;
void cycleDebugView();
int debugView();

// Used only for debug view 7. Reuses otherwise-unused synthesis constants,
// not model weights, and leaves debug-view/blending controls untouched.
inline void setUiRegionDebugConstants(ShaderConstants& c, const UiRegionDebug& state) {
    c.neuralControl[0] = !state.maskEnabled ? 0.0f : !state.detectorEnabled ? 1.0f : !state.valid ? 2.0f : 3.0f;
    const uint32_t count = state.maskEnabled && state.detectorEnabled && state.valid
        ? std::min(state.count, static_cast<uint32_t>(kMaxUiRegions)) : 0;
    c.neuralControl[2] = static_cast<float>(count);
    for (uint32_t i = 0; i < count; ++i) {
        const UiRegion& r = state.regions[i];
        const float bounds[] = {r.left, r.top, r.right, r.bottom};
        std::copy_n(bounds, 4, c.neuralWeights + 4 * i);
    }
}
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
// depth0/depth1: the frames' AI depth (FlowEngine::depthFor), used by the
// projection only when both are given. dilated: two field-sized scratch
// textures (createDilatedFields); with them, each field whose frame has
// depth is first dilated at outlines (DepthDilateCS) and used dilated.
// aiBlendMask: optional pair-owned AI midpoint preference for frame0 (t6).
bool recordInterpolate(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const ComputeKernels& kernels,
                       const ShaderConstants& constants, ID3D12Resource* frame0, ID3D12Resource* frame1,
                       ID3D12Resource* forward, ID3D12Resource* backward, ID3D12Resource* uiMask,
                       ID3D12Resource* projection, ID3D12Resource* output, uint32_t implicitInputs,
                       ID3D12Resource* depth0 = nullptr, ID3D12Resource* depth1 = nullptr,
                       ID3D12Resource* const* dilated = nullptr, ID3D12Resource* aiBlendMask = nullptr,
                       ID3D12Resource* hudPersist = nullptr);
// The projection for frames of width x height, at any flow grid of 2 px or
// coarser.
bool createProjection(GraphicsDevice& device, uint32_t width, uint32_t height, ComPtr<ID3D12Resource>& projection);
// The two dilated fields for recordInterpolate, at any flow grid of 2 px or
// coarser.
bool createDilatedFields(GraphicsDevice& device, uint32_t width, uint32_t height, ComPtr<ID3D12Resource> (&fields)[2]);
// Debug view 5: draws the AI depth of depth0/depth1, blended at
// constants.timeT, over `output` (DepthViewCS).
bool recordDepthView(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const ComputeKernels& kernels,
                     const ShaderConstants& constants, ID3D12Resource* depth0, ID3D12Resource* depth1,
                     ID3D12Resource* output);

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
    ID3D12Resource* hudPersist = nullptr;  // Entry::hudPersistInput(), debug view 7 only
    UiRegionDebug uiRegionDebug;       // Same pair's allow-list/status, not latest detections.
    ID3D12Resource* aiBlendMask = nullptr;  // pair-owned network blend mask; null = disabled
    ID3D12Fence* flowFence = nullptr;   // GPU-side wait before the interpolate pass
    uint64_t flowFenceValue = 0;
    ID3D12Resource* depth0 = nullptr;  // AI depth of frame0 / frame1, when complete
    ID3D12Resource* depth1 = nullptr;
    float alpha = 0.0f;
    bool single = false;               // show frame0 (alpha < 0.5) or frame1 as-is
};

struct PresentStats {
    uint64_t presented = 0;
    uint64_t prepared = 0;
    double maxIntervalMs = 0.0;
    double meanIntervalMs = 0.0;
    double capacityWaitMs = 0.0;      // total CPU time in swap-chain back-pressure
    double maxCapacityWaitMs = 0.0;
    double allocatorWaitMs = 0.0;
    double maxAllocatorWaitMs = 0.0;
    double presentCallMs = 0.0;       // CPU time inside DXGI Present, not scanout
    double maxPresentCallMs = 0.0;
    uint64_t gpuPendingAtPresent = 0; // prepared list not complete when Present was called
    double maxDeadlineLateMs = 0.0;   // CPU call start vs output deadline, not scanout
};

// Native D3D12 flip-model presentation. The caller supplies FPS-clocked
// output; this class submits it with Present(0), without a display clock.
class Presenter {
public:
    static constexpr UINT kBackBuffers = 3;

    ~Presenter();

    // maxFrameLatency: frames the swap chain may hold before the compositor
    // takes them (1, the default, or 2; kBackBuffers allows at most 2).
    // allowTearing only controls the tearing flag. The sync interval is
    // always zero: display Hz must not replace the requested output FPS.
    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, HWND window,
                     uint32_t width, uint32_t height, UINT maxFrameLatency = 1, bool allowTearing = false);
    // Tearing was allowed and the system supports it.
    bool tearing() const { return m_tearing; }
    // How the last frames reached the screen (DXGI frame statistics):
    // "composed" (DWM draws the window at the display's refresh, tearing has
    // no effect), "independent flip" (the window has the display or a plane
    // to itself: tearing and FreeSync apply), or "unknown".
    const char* presentationMode() const;
    ID3D12Fence* fence() const { return m_fence.Get(); }
    double refreshRate() const { return m_refreshHz; }

    // Prepare immediately after reserving an output, BEFORE its deadline.
    // Records/submits GPU work and acquires queue capacity. Only present()
    // runs at the deadline. One prepared output at a time; its fence covers
    // all GPU reads, even if shutdown occurs before it is presented.
    bool prepare(const RenderJob& job, uint64_t& presentFenceValue);
    bool present(int64_t deadline);
    // FSR may submit on this same queue AFTER a real frame was presented.
    // Publish a later fence covering those reads before input is retired.
    bool protectQueuedWork(uint64_t& fenceValue);

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
    ComPtr<ID3D12Resource> m_dilated[2];   // DepthDilateCS

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
    bool m_tearing = false;
    bool m_prepared = false;
    uint64_t m_preparedFence = 0;
    HRESULT m_lastPresentStatus = S_OK;

    double m_refreshHz = 60.0;  // diagnostics and explicit Max mode only
    int64_t m_lastPresentTime = 0;

    PresentStats m_stats;
    double m_intervalSum = 0.0;
    uint64_t m_intervalCount = 0;
};
