#pragma once

#include "AiFlowEstimator.h"
#include "ComputeKernels.h"
#include "DepthService.h"
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
    // DIS (Dense Inverse Search): overlapping patches refined by inverse
    // compositional Lucas-Kanade and densified per pixel, instead of the SAD
    // block search. See DisPatchCS.hlsl. The default: on the rendered test
    // pairs it measured 45.7 dB against SAD's 44.6, ahead on 9 of 10, and
    // 48.3 against 37.5 on the CS2-style sequence.
    bool disFlow = true;
    // The AI pyramid flow engine: the network trained and exported with
    // tools/ai-train (models\ai_flow.onnx), run by ONNX Runtime on the flow
    // queue (AiFlowEstimator), its two fields filling the 2x2 ring. Falls
    // back to DIS when there is no model.
    bool aiFlow = false;
    // AI pyramid only: use the network's learned midpoint blend mask during
    // synthesis. Experimental, off by default; transported in time by motion.
    bool aiBlendMask = false;
    // AI pyramid only: integrated neural correction INSTEAD OF DIS. Off
    // selects the paired backbone-only ONNX + DIS. Choose before capture;
    // switching requires a new session so in-flight tensor bindings stay valid.
    bool aiRefinement = false;
    // With FFX: refine its whole-pixel 8x8 field into a per-pixel one at half
    // resolution by variational refinement (FfxRefine.hlsli), stored on a
    // 2x2 grid. Off = FFX's field as it comes.
    bool ffxRefine = true;
    // With the refinement: refine a forward field as well, from FFX's
    // turned around, so the two directions are estimated separately and
    // InterpolateCS's occlusion weighting can tell which frame shows a
    // point (FfxRefineConsistCS). Off by default: on the third-person 4x
    // test it sharpens the character (+0.5 dB) but costs the background
    // (-1.0 dB) and step-to-step smoothness (+20% temporal error), because
    // FFX's field is not accurate enough at edges for the test to separate
    // occlusion from FFX's own errors. Off = FFX's single field, negated
    // for the forward direction.
    bool ffxOcclusion = false;
    // FFX: before anything reads FFX's field, let each block try zero motion
    // and its neighbours' vectors (FfxRepairCS) - FFX spreads one surface's
    // vector over its neighbours at large motion.
    bool ffxRepair = true;
    bool allowTrainedModel = true; // false while gathering offline training examples
    // Keep screen-fixed UI (HUD, subtitles, logos) out of the warp. Per
    // pixel and confirmed over several pairs; see UiMaskCS.hlsl.
    bool uiMask = true;
    // DIS: give the blocks beside a moving object that neither field
    // confirms one consistent vector - the visible surface's where the
    // picture says so, the background's where the block is hidden in the
    // other frame (OcclusionFillCS). Once per pair, on the ring grid.
    bool occlusionFill = true;
    // Refined FFX: a 3x3 median of both fields on the 2 px ring grid, once
    // per pair, removing isolated outlier vectors the refinement leaves
    // where the picture does not decide (FlowMedianCS).
    bool flowMedian = true;
    // AI depth for every frame (DepthService, DepthEstimator), used by the
    // motion projection to put the nearer of two equally plausible surfaces
    // in front. Two-field pairs only (DIS, FFX --ffx-occlusion). Needs the
    // ONNX Runtime / DirectML DLLs and the model next to the executable.
    bool aiDepth = false;
    // The network on every n-th new frame (1-4); the frames between borrow
    // a neighbour's depth. 1 = every frame (about 1.3-1.6 ms of GPU each).
    int aiDepthStride = 1;
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
    static constexpr int kMaxLevels = 10;
    // Do not search below a 4 px short side; smaller levels have too little
    // independent image content for the 8x8 SAD / DIS patches. This allows
    // 1080p to use 9 levels and 4K to use all 10.
    static constexpr uint32_t kMinCoarsestExtent = 4;
    static constexpr int kRingSize = 16;
    static constexpr UINT kListsInFlight = 4;
    static constexpr int kMaxSlots = 16;
    // DIS: 8x8 patches every kDisStride pixels at every level, kDisIterations
    // Lucas-Kanade steps each, and a ring grid of kDisRingBlock pixels - the
    // densified field is per pixel, so the ring can be finer than SAD's. At
    // 4 px, the mean over a block and the bilinear blend of neighbouring
    // blocks spread an object's motion 2-4 px into the background around
    // its outline; 2 px halves that (third-person 4x test: 1-2 px outside
    // the character +0.22 dB, the character +0.1 dB, rendered pairs +0.0 to
    // +0.2 dB, none worse).
    static constexpr int kDisStride = 4;
    static constexpr int kDisIterations = 12;
    static constexpr uint32_t kDisRingBlock = 2;
    // AI engine (recordAiFlow): the network's field seeds DIS's dense field
    // at kAiSeedLevel (160 x 90 at 1440p), and DIS refines it down to
    // kAiFinestLevel (half resolution, the ring's grid) with
    // kAiRefineIterations steps per patch.
    static constexpr int kAiSeedLevel = 4;
    static constexpr int kAiFinestLevel = 1;
    static constexpr int kAiRefineIterations = 2;

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
        // R8_UNORM at full resolution: the static-UI evidence after this
        // pair (UiMaskCS). Allocated only when FlowSettings::uiMask is on;
        // hasUiMask says whether this pair actually wrote it.
        ComPtr<ID3D12Resource> uiMask;
        bool hasUiMask = false;
        // What InterpolateCS binds at t4: null reads as "no UI anywhere",
        // which is exactly the plain warp.
        ID3D12Resource* uiMaskInput() const { return hasUiMask ? uiMask.Get() : nullptr; }
        // The AI pair's midpoint blend preference, at network resolution.
        // Each ring entry owns its copy: a later inference must not overwrite
        // a mask that the presenter is still reading for an earlier pair.
        ComPtr<ID3D12Resource> aiBlendMask;
        bool hasAiBlendMask = false;
        bool aiRefined = false;
        int aiDisPatchPasses = 0;   // Per-pair diagnostics; zero in integrated mode.
        ID3D12Resource* aiBlendMaskInput() const { return hasAiBlendMask ? aiBlendMask.Get() : nullptr; }
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
    // AI depth: the frame's nearness texture once complete, else null
    // (non-blocking); waitDepth blocks up to timeoutMs (offline paths).
    ID3D12Resource* depthFor(const FlowInput& frame) const;
    ID3D12Resource* waitDepth(const FlowInput& frame, unsigned timeoutMs);
    Entry entry(int index) const;
    void noteEntryPresented(int index, uint64_t presentFenceValue);

    bool usingFfx() const { return m_usingFfx; }
    bool usingNeural() const { return m_neuralModelLoaded && !m_usingFfx; }
    bool usingDis() const { return m_usingDis; }
    bool usingAi() const { return m_usingAi; }
    bool usingAiRefinement() const { return m_usingAi && m_aiIntegrated; }
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
        if (disActive()) return m_disDense[0][static_cast<size_t>(level)].Get();
        return m_estimatedFlow[0][static_cast<size_t>(level)].Get();
    }
    // The forward matcher's raw output at a level, before FillFlowCS: with
    // trainingFlow it separates what the search measured from what the fill
    // changed (--gpu-flow-eval).
    ID3D12Resource* searchFlow(int level) const {
        if (level < 0 || level >= m_levels) return nullptr;
        if (disActive()) return m_disPatch[0][static_cast<size_t>(level)].Get();
        return m_searchFlow[0][static_cast<size_t>(level)].Get();
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
        ComPtr<ID3D12Resource> stats;   // 1x1 R32G32_FLOAT: whole-frame luma mean, deviation
        uint64_t id = UINT64_MAX;   // frame whose luma it holds
    };

    bool beginList();
    bool endList(const FlowInput& previous, const FlowInput& current, uint64_t& fenceValue);
    int acquireEntry();
    bool recordPyramid(const FlowInput& frame, Pyramid& pyramid);
    bool recordSad(const FlowInput& previous, const FlowInput& current, Entry& out);
    bool recordPack(Pyramid& frame, const Pyramid& partner);
    bool recordDis(const FlowInput& previous, const FlowInput& current, Entry& out);
    int statsLevel() const { return std::max(0, m_levels - 3); }
    // DIS runs unless a learned model is loaded (the neural estimator is
    // built on the SAD graph). With FFX selected it is the fallback only when
    // FFX's field is refined: refined FFX fills DIS's 2x2 ring, plain FFX
    // fills an 8x8 one and falls back to SAD, which fills the same grid.
    bool disActive() const {
        if (m_neuralModelLoaded || aiFlowActive()) return false;
        return m_settings.preferFfxOpticalFlow ? m_settings.ffxRefine : m_settings.disFlow;
    }
#ifdef MOTION_ENHANCER_FFX_OPTICAL_FLOW
    bool createFfx();
    void destroyFfx();
    bool recordFfx(const FlowInput& previous, const FlowInput& current, Entry& out);
    bool recordFfxRefine(const FlowInput& previous, const FlowInput& current, Entry& out);
    bool ffxRefineActive() const { return m_settings.preferFfxOpticalFlow && disActive(); }
#endif
    bool recordUiMask(const FlowInput& previous, const FlowInput& current, int entryIndex, Entry& out);
    bool recordOcclusionFill(const FlowInput& previous, const FlowInput& current, Entry& out);
    // The AI pyramid flow engine (FlowSettings::aiFlow).
    bool aiFlowActive() const { return m_aiFlow && m_aiFlow->ready(); }
    bool recordAiFlow(const FlowInput& previous, const FlowInput& current, Entry& out);
    // Executes what the list holds so far (after the capture-ready waits)
    // and reopens it on the same allocator: work queued on the flow queue in
    // between - the AI network - runs after the first part, before the rest.
    bool splitList(const FlowInput& previous, const FlowInput& current);
    void releaseAiBuffers();
    bool recordFlowMedian(Entry& out);
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
    // optical flow keeps the previous frame's pyramid the same way). Each
    // pair's flow is estimated from that pair alone: no previous pair's
    // vectors are carried into the search.
    std::array<Pyramid, kMaxSlots> m_pyramids;
    std::unique_ptr<DepthService> m_depth;   // FlowSettings::aiDepth
    std::unique_ptr<AiFlowEstimator> m_aiFlow;   // FlowSettings::aiFlow
    bool m_aiIntegrated = false;   // Selected graph; immutable for this initialized session.
    ComPtr<ID3D12Resource> m_aiPlanar;    // network input, planar, W x 6H R32 (AiFlowInputCS)
    ComPtr<ID3D12Resource> m_aiInput;     // ... as the network's input buffer
    ComPtr<ID3D12Resource> m_aiFlowOut;   // network outputs (buffers)
    ComPtr<ID3D12Resource> m_aiMaskOut;
    ComPtr<ID3D12Resource> m_aiFlowTex;   // flow output as W x 4H R32, for AiFlowToRingCS
    // The network's field at the seed level, per direction; DIS's own coarse
    // field at that level is in m_disDense (recordAiFlow).
    std::array<ComPtr<ID3D12Resource>, 2> m_aiSeed;
    void* m_aiInputHandle = nullptr;
    void* m_aiFlowHandle = nullptr;
    void* m_aiMaskHandle = nullptr;
    int m_slotCount = 0;
    std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2> m_estimatedFlow; // [dir][level]
    // Raw matcher output, before FillFlowCS writes the filled field into
    // m_estimatedFlow. Single-buffered: produced and consumed inside one
    // command list, and the COMMON<->UAV transitions around each dispatch
    // order it against any other submission on the queue.
    std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2> m_searchFlow;  // [dir][level]

    // Static-UI chain: the entries holding the last two pairs' evidence,
    // each with the frame its pair ended on; [0] is the newest. A pair's
    // prior is the link that ended on the pair's first frame. Two deep
    // because the timeline can replace its newest frame (a repaint between
    // video frames): the replacement pair starts one frame back, and its
    // prior is the evidence from before the replaced pair, not nothing.
    struct UiLink {
        int entry = -1;
        uint64_t frameId = UINT64_MAX;
    };
    std::array<UiLink, 2> m_uiChain;

    // DIS, per direction and level: the patch grid and the densified
    // per-pixel field (both R32G32_FLOAT, in that level's pixels). Kept per
    // direction so the forward field survives the backward pass for
    // --gpu-flow-eval.
    std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2> m_disPatch;   // [dir][level]
    std::array<std::array<ComPtr<ID3D12Resource>, kMaxLevels>, 2> m_disDense;
    // Occlusion fill: one ring-sized field per direction, so both are filled
    // from the unfilled pair before either is written back.
    std::array<ComPtr<ID3D12Resource>, 2> m_fillScratch;
    // Flow median: one ring-sized field per direction, filtered into and
    // copied back.
    std::array<ComPtr<ID3D12Resource>, 2> m_medianScratch;
    bool m_usingDis = false;
    bool m_usingAi = false;


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
    ComPtr<ID3D12Resource> m_ffxRepaired;  // R32G32_FLOAT, the same field after FfxRepairCS
    // Refinement (FfxRefine.hlsli), at half resolution: the field, two
    // R32G32_FLOAT buffers ping-ponged by the outer iterations (one
    // FfxRefineIterateCS dispatch each).
    static constexpr int kFfxRefineOuter = 3;
    std::array<ComPtr<ID3D12Resource>, 2> m_refineField;
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
