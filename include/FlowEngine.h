#pragma once

#include "ComputeKernels.h"
#include "GraphicsDevice.h"
#include "NeuralFlow.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#endif

struct FlowSettings {
    int pyramidLevels = 8;
    int minRefineLevel = 0;
    // Search radius (pixels at that level) above the finest level, and at
    // the finest level. Full dense windows everywhere: see sadRadius().
    //
    // Not eight. MotionSearchCS spreads a window's candidate columns over
    // its 64 threads, so a span of 13 (radius 6) is one pass of the inner
    // SAD loop per thread while a span of 17 (radius 8) is two - the cost
    // is a step function of the radius, and 8 sits just past the step for
    // no measured gain. At 1080p, 8/8 took 4.67 ms and 6/6 takes 3.06 ms
    // with equal or better PSNR everywhere.
    //
    // The coarse radius is narrower still, at four, and that is about
    // accuracy rather than speed. A coarse level's pixels each cover a large
    // area of screen, so a wide window there is a long way to look for a
    // match that the level cannot really resolve - and on periodic content
    // it finds one. A page of bar charts (bars on a 46 px pitch, text rows
    // on 20 px) locked onto those aliases and handed them down as
    // predictors: 18.80 dB at radius 6 against **47.69 dB** at radius 3,
    // and the live symptom was artifacts around text. Four also improved
    // the 160 px motion case and the pan case. It costs 4.6 dB on the
    // synthetic `circle` test, whose sinusoidal background is diagonally
    // translation-invariant and whose 21 px texture period makes -5 an
    // alias of +16 - that case genuinely needs the wide coarse window, and
    // it is the case the README already warns not to judge objects on. The
    // noise-background equivalents are unchanged.
    //
    // Reach still compounds down the pyramid: 6 + 4*(2+4+8+16+32) + 4*64 +
    // 3*128 = 896 px over eight levels, and a global -48,-24 px motion is
    // found exactly.
    int coarseSearchRadius = 4;
    int refineSearchRadius = 6;
    float smoothnessWeight = 0.0005f;
    bool preferFfxOpticalFlow = false;
    bool neuralFlow = false; // requires a validated learned-feature model
    bool allowTrainedModel = true; // false while gathering offline training examples
};

// A frame's contribution to the flow queue: the shared capture texture and
// the capture-ready fence value that publishes its pixels.
struct FlowInput {
    ID3D12Resource* texture = nullptr;
    uint64_t readyValue = 0;
    uint64_t id = 0;   // unique per captured frame (slots are recycled)
    int slot = 0;      // capture slot: selects the cached luma pyramid
};

// Computes bidirectional flow for consecutive frame pairs on the flow queue
// and keeps the results in a ring so the presenter can consume any recent
// pair without the producer and consumer ever blocking each other.
//
// Thread model: submitPair runs on the capture thread; entry accessors and
// fence queries run on the render thread. The ring is protected by a mutex.
class FlowEngine {
public:
    static constexpr int kMaxLevels = 8;
    // Levels whose short side would drop below this are not searched: half
    // a 16 px block, so 1080p gets the full 8 levels (coarsest 15x8).
    static constexpr uint32_t kMinCoarsestExtent = 8;
    static constexpr int kRingSize = 16;
    static constexpr UINT kListsInFlight = 4;
    static constexpr int kMaxSlots = 16;

    struct Entry {
        // R32G32B32A32_FLOAT per block: xy = vector in full-resolution
        // pixels, zw = how much the estimator could actually measure about
        // each axis (see FillFlowCS). The consumer only needs xy; zw is
        // carried so the fill pass and the next pyramid level can tell a
        // measured component from an inherited one.
        ComPtr<ID3D12Resource> forward;
        ComPtr<ID3D12Resource> backward;
        uint32_t blockSize = 8;            // 8x8 SAD/FFX flow grid
        // True when `forward` is exactly the negation of `backward` and both
        // are indexed on the second frame's block grid, because the estimator
        // only ran one direction (FFX). The warp then solves one trajectory
        // instead of two; see InterpolateCS.hlsl for why the distinction is
        // not cosmetic. False for the SAD and neural graphs, which search
        // both directions independently.
        bool singleField = false;
        uint64_t flowFence = 0;            // flow-queue value completing this entry
        uint64_t presentFence = 0;         // last present-queue value that read it
        bool valid = false;
    };

    struct Result {
        int entry = -1;
        uint64_t fence = 0;
    };

    ~FlowEngine();

    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels,
                    ID3D12Fence* presentFence, const FlowSettings& settings);
    // slotCount: number of capture slots whose luma pyramids are cached.
    bool resize(uint32_t width, uint32_t height, int slotCount = 2);
    void setSettings(const FlowSettings& settings);
    // Offline validation and the CPU/GPU parity check: inject weights
    // directly instead of going through a file on disk.
    void setNeuralModel(const NeuralFlowModel& model) {
        m_neuralModel = model;
        m_neuralModelLoaded = model.trained;
    }

    // Records and submits flow for (previous -> current). Returns the ring
    // entry and the fence value that completes it, or entry -1 on failure.
    Result submitPair(const FlowInput& previous, const FlowInput& current);

    ID3D12Fence* fence() const { return m_fence.Get(); }
    bool fenceReached(uint64_t value) const {
        return m_fence && m_fence->GetCompletedValue() >= value;
    }
    // Waits on the CPU (offline path only).
    bool waitFence(uint64_t value);
    Entry entry(int index) const;
    void noteEntryPresented(int index, uint64_t presentFenceValue);

    bool usingFfx() const { return m_usingFfx; }
    bool usingNeural() const { return m_neuralModelLoaded && !m_usingFfx; }
    bool ffxAvailable() const { return m_ffxReady; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    int searchedLevels() const { return m_levels; }
    double lastGpuTimeMs();
    // Offline trainer diagnostics: compare descriptor computation with its
    // CPU mirror before accepting a learned checkpoint.
    ID3D12Resource* trainingFeature(int slot, int level) const {
        if (!m_neuralModelLoaded || slot < 0 || slot >= m_slotCount ||
            level < 0 || level >= m_levels) return nullptr;
        return m_pyramids[static_cast<size_t>(slot)].feature[static_cast<size_t>(level)].Get();
    }
    ID3D12Resource* trainingFlow(int level) const {
        if (level < 0 || level >= m_levels) return nullptr;
        return m_estimatedFlow[0][1 - m_flowParity][static_cast<size_t>(level)].Get();
    }

private:
    struct CommandSet {
        ComPtr<ID3D12CommandAllocator> allocator;
        uint64_t fence = 0;
    };
    struct Pyramid {
        std::array<ComPtr<ID3D12Resource>, kMaxLevels> level;    // R16_FLOAT luma
        std::array<ComPtr<ID3D12Resource>, kMaxLevels> packed;   // R32_UINT, 4 bytes per texel
        std::array<ComPtr<ID3D12Resource>, kMaxLevels> feature;  // R16_FLOAT learned descriptor
        uint64_t id = UINT64_MAX;   // frame whose luma it holds
    };

    bool beginList();
    bool endList(const FlowInput& previous, const FlowInput& current, uint64_t& fenceValue);
    int acquireEntry();
    bool recordPyramid(const FlowInput& frame, Pyramid& pyramid);
    bool recordSad(const FlowInput& previous, const FlowInput& current, Entry& out);
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    bool createFfx();
    void destroyFfx();
    bool recordFfx(const FlowInput& previous, const FlowInput& current, Entry& out);
#endif
    void readTimestamps(UINT set);

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    ComPtr<ID3D12Fence> m_presentFence;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    std::array<CommandSet, kListsInFlight> m_commands;
    std::atomic<uint64_t> m_submitCount{0};
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;
    DispatchContext m_dispatch;

    mutable std::mutex m_ringMutex;
    std::array<Entry, kRingSize> m_ring;
    int m_ringCursor = 0;
    int m_currentEntry = -1;

    FlowSettings m_settings;
    NeuralFlowModel m_neuralModel;
    bool m_neuralModelLoaded = false;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    int m_levels = 0;
    // SAD resources. Luma pyramids are cached per capture slot so a frame's
    // pyramid is built once, when it first takes part in a pair (AMD's
    // optical flow keeps the previous frame's pyramid the same way). The
    // estimated flow of every level is double-buffered per submission: the
    // previous pair's flow at a level is that level's temporal predictor.
    std::array<Pyramid, kMaxSlots> m_pyramids;
    int m_slotCount = 0;
    std::array<std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2>, 2> m_estimatedFlow; // [dir][parity][level]
    // Raw matcher output, before FillFlowCS writes the filled field into
    // m_estimatedFlow. Single-buffered: produced and consumed inside one
    // command list, and the COMMON<->UAV transitions around each dispatch
    // order it against any other submission on the queue.
    std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2> m_searchFlow;  // [dir][level]

    int m_flowParity = 0;
    uint64_t m_lastSadCurrentId = UINT64_MAX;   // temporal chain: previous pair's current frame

    // GPU timestamps for the status line.
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    ComPtr<ID3D12Resource> m_queryReadback;
    uint64_t m_timestampFrequency = 0;
    std::mutex m_timestampMutex;
    uint64_t m_timestampsRead = 0;   // submissions whose timestamps were consumed
    double m_lastGpuTimeMs = 0.0;

#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    // 512 KB opaque context: heap-allocated so FlowEngine can live on a
    // thread stack.
    std::unique_ptr<FfxOpticalflowContext> m_ffxContext;
    FfxInterface m_ffxInterface = {};
    std::vector<uint8_t> m_ffxScratch;
    ComPtr<ID3D12Resource> m_ffxVectors;   // R16G16_SINT, 8x8 grid, current -> previous
    ComPtr<ID3D12Resource> m_ffxScd;
    uint64_t m_ffxLastId = UINT64_MAX;     // last frame fed to the temporal chain
#endif
    bool m_ffxReady = false;
    bool m_usingFfx = false;
};

// Fills everything InterpolateCS reads about a completed flow pair. Every
// consumer goes through this rather than assigning the three fields by hand:
// `flowScale` selects the field convention, and a site that sets the block
// size and the timestamp but forgets the convention silently warps a
// bidirectional pair with single-field math (or the reverse), which is a
// wrong-equation bug that still produces a plausible-looking frame.
inline void setInterpolationConstants(ShaderConstants& c, const FlowEngine::Entry& entry, float t) {
    c.blockSize = static_cast<int32_t>(entry.blockSize);
    c.parentBlockSize = entry.blockSize;
    c.timeT = std::clamp(t, 0.0f, 1.0f);
    c.flowScale = entry.singleField ? 1.0f : 0.0f;
}
