#pragma once

#include "ComputeKernels.h"
#include "GraphicsDevice.h"

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
    // the finest level. Full windows everywhere: see msadRadius().
    int coarseSearchRadius = 8;
    int refineSearchRadius = 8;
    float smoothnessWeight = 0.0005f;
    bool preferFfxOpticalFlow = true;
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
        ComPtr<ID3D12Resource> forward;    // R32G32_FLOAT, one vector per block
        ComPtr<ID3D12Resource> backward;
        uint32_t blockSize = 32;           // 8 for FFX, 32 for MSAD
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
    bool ffxAvailable() const { return m_ffxReady; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    int searchedLevels() const { return m_levels; }
    double lastGpuTimeMs();

private:
    struct CommandSet {
        ComPtr<ID3D12CommandAllocator> allocator;
        uint64_t fence = 0;
    };
    struct Pyramid {
        std::array<ComPtr<ID3D12Resource>, kMaxLevels> level;    // R16_FLOAT luma
        std::array<ComPtr<ID3D12Resource>, kMaxLevels> packed;   // R32_UINT, 4 bytes per texel
        uint64_t id = UINT64_MAX;   // frame whose luma it holds
    };

    bool beginList();
    bool endList(const FlowInput& previous, const FlowInput& current, uint64_t& fenceValue);
    int acquireEntry();
    bool recordPyramid(const FlowInput& frame, Pyramid& pyramid);
    bool recordMsad(const FlowInput& previous, const FlowInput& current, Entry& out);
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
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    int m_levels = 0;
    // MSAD resources. Luma pyramids are cached per capture slot so a frame's
    // pyramid is built once, when it first takes part in a pair (AMD's
    // optical flow keeps the previous frame's pyramid the same way). The
    // filtered flow of every level is double-buffered per submission: the
    // previous pair's flow at a level is that level's temporal predictor.
    std::array<Pyramid, kMaxSlots> m_pyramids;
    int m_slotCount = 0;
    std::array<ComPtr<ID3D12Resource>, kMaxLevels> m_rawFlow;
    std::array<std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2>, 2> m_filteredFlow; // [dir][parity][level]
    int m_flowParity = 0;
    uint64_t m_lastMsadCurrentId = UINT64_MAX;   // temporal chain: previous pair's current frame

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
