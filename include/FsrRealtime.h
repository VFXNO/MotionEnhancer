#pragma once

#include "ComputeKernels.h"
#include "FlowEngine.h"
#include "FsrFrameGen.h"
#include "GraphicsDevice.h"

#include <array>
#include <cstdint>
#include <string>

// Experimental, optional (--fsr-fg, off by default): AMD FSR frame generation
// beside the program's own interpolation in the live overlay. Nothing of the
// normal path is replaced: when this is on, each new pair of game frames is
// also given to FSR (FsrFrameGen) with that pair's optical flow as motion
// vectors, and the frame FSR makes halfway between them is shown on refreshes
// whose blend position is close to the middle. Every other refresh, and every
// pair FSR cannot take (scene cuts, pairs without flow), is drawn by the
// normal interpolation exactly as without FSR.
//
// FSR 4.0.2 makes only the middle frame (it accepts more outputs but leaves
// them empty), so FSR covers every generated frame only at 2x output
// (--multiplier 2); at the display rate it covers the refreshes near the
// middle of each pair.
//
// All work is recorded into this class's own command list and submitted on
// the present queue before the presenter's list, so the presenter sees the
// finished frame through queue order. The GPU waits for the capture copies
// and the pair's flow; the CPU never does.
class FsrRealtime {
public:
    struct Stats {
        uint64_t generated = 0;   // pairs FSR made a middle frame for
        uint64_t resets = 0;      // pairs that restarted FSR's sequence
        uint64_t failures = 0;
    };

    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, uint32_t width, uint32_t height,
                    std::string& error);

    // The middle frame of the pair (in0, in1), made the first time a pair is
    // seen and returned again for the same pair. index0/index1 are the
    // frames' timeline indices; entry is the pair's flow (valid), ready on
    // flowFence at flowFenceValue. Null when FSR has no frame for this pair.
    ID3D12Resource* middleFrame(const FlowInput& in0, uint64_t index0, const FlowInput& in1, uint64_t index1,
                                const FlowEngine::Entry& entry, ID3D12Fence* flowFence, uint64_t flowFenceValue,
                                float frameTimeMs);

    // A pair FSR did not see (a scene cut, a pair without flow, a dropped
    // frame) needs nothing from the caller: frame0 is then not the last frame
    // FSR was given, and the next pair starts FSR's sequence again.

    Stats takeStats();

private:
    static constexpr uint32_t kLists = 3;
    bool feed(const FlowInput& input, ID3D12Resource* backward, uint32_t blockSize, bool reset,
              ID3D12Resource* output, float frameTimeMs);

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_renderWidth = 0, m_renderHeight = 0;
    FsrFrameGen m_fsr;
    DispatchContext m_dispatch;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    std::array<ComPtr<ID3D12CommandAllocator>, kLists> m_allocators;
    std::array<uint64_t, kLists> m_allocatorFence = {};
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    uint32_t m_submits = 0;
    // FSR's inputs: the game frame as RGBA (two, alternating, so the one FSR
    // holds as its previous frame is never overwritten), motion, depth.
    std::array<ComPtr<ID3D12Resource>, 2> m_color;
    ComPtr<ID3D12Resource> m_motion, m_depth;
    // The middle frames, alternating per pair.
    std::array<ComPtr<ID3D12Resource>, 2> m_outputs;
    uint32_t m_current = 0;
    uint64_t m_frameId = 0;
    uint64_t m_fedIndex = 0;     // the last game frame given to FSR (0 = none)
    uint64_t m_pairIndex = 0;    // index1 of the pair m_outputs[m_current] belongs to
    bool m_pairValid = false;
    Stats m_stats;
    bool m_errorReported = false;
};
