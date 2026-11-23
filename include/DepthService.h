#pragma once

#include "ComputeKernels.h"
#include "DepthEstimator.h"
#include "GraphicsDevice.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

struct FlowInput;

// AI depth for every captured frame (FlowSettings::aiDepth), next to the
// flow: per capture slot, a kSize x kSize "nearness" texture (R32_FLOAT,
// 0 = far, 1 = near) that the motion projection uses to put the nearer of
// two equally plausible surfaces in front (ProjectSplatCS).
//
// Everything stays on the GPU. The flow list writes each new frame's network
// input (DepthInputCS: averaged down, normalised, planar) and copies it into
// the slot's input buffer. A worker thread then queues the network on those
// buffers on the estimator's own queue, which first waits on the GPU for
// that flow list (DepthEstimator::runGpu), and a compute queue of this
// service maps the output to nearness (DepthFinishCS) once the network is
// done. The CPU only orders the work. The presenter takes a frame's depth
// only once it is complete (depthFor), so nothing on the way to the screen
// waits for the network: a frame whose depth is not ready yet is
// interpolated without it.
//
// The network runs on every `stride`-th new frame (FlowSettings::
// aiDepthStride; 1 = every frame; about 1.3-1.6 ms of GPU a run, nearly all
// fixed per-layer cost). With a stride above 1, a frame without its own
// depth borrows its neighbour's (depthFor).
class DepthService {
public:
    static constexpr int kSize = DepthEstimator::kSize;
    static constexpr int kMaxSlots = 16;
    static constexpr int kMaxStride = 4;

    ~DepthService();

    // readFence: the present queue's fence, for release() (null offline).
    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* flowFence, int slots,
                    int stride = 1, ID3D12Fence* readFence = nullptr);

    // Records the input of `frame` into the flow list, unless this frame's
    // depth is already done or pending. Returns false only on a recording
    // failure.
    bool record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame,
                uint32_t width, uint32_t height);
    // The flow list holding everything record() added since the last call
    // was submitted with this fence value / was thrown away.
    void submitted(uint64_t flowFenceValue);
    void discarded();

    // The frame's nearness texture once complete - its own, else that of the
    // nearest frame (by id, within the stride) that has one - else null.
    // Non-blocking.
    ID3D12Resource* depthFor(const FlowInput& frame) const;
    // Offline: waits up to timeoutMs for the frame's depth.
    ID3D12Resource* waitFor(const FlowInput& frame, unsigned timeoutMs);
    // Live: depthFor, pinning the returned texture until release() gives the
    // readFence value that covers the GPU read (0 when it was not read).
    // With a stride above 1 a frame borrows its neighbour's texture, and the
    // neighbour's capture slot - which that texture belongs to - is not one
    // the reader protects: once it was retired and taken by a new frame, the
    // worker rewrote the texture while a presented frame could still be
    // reading it. The worker now stops handing out a texture it is about to
    // rewrite, waits for pinned readers to report, then makes its queue wait
    // on GPU for their reads.
    ID3D12Resource* acquire(const FlowInput& frame);
    void release(ID3D12Resource* depth, uint64_t readFenceValue);

private:
    struct Slot {
        ComPtr<ID3D12Resource> planar;     // kSize x 3 kSize R32_FLOAT, DepthInputCS output
        ComPtr<ID3D12Resource> input;      // the network's input tensor (buffer)
        ComPtr<ID3D12Resource> output;     // its output tensor (buffer)
        ComPtr<ID3D12Resource> raw;        // the output as a kSize^2 R32_FLOAT texture
        ComPtr<ID3D12Resource> depth;      // kSize^2 R32_FLOAT nearness, what shaders read
        void* inputHandle = nullptr;       // DirectML allocations of input / output
        void* outputHandle = nullptr;
        uint64_t pendingId = UINT64_MAX;   // frame being produced
        uint64_t readyId = UINT64_MAX;     // frame whose depth `depth` holds
        uint32_t pinned = 0;               // acquire()d and not yet release()d
        uint64_t readValue = 0;            // m_readFence value covering every released read
    };
    struct Job {
        int slot = 0;
        uint64_t frameId = 0;
        uint64_t flowFence = 0;
    };
    void run();
    bool produce(const Job& job, uint64_t readValue);
    int nearestSlotLocked(uint64_t frameId) const;           // m_mutex held; -1 if none
    ID3D12Resource* nearestLocked(uint64_t frameId) const;   // m_mutex held

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    ComPtr<ID3D12Fence> m_flowFence;
    ComPtr<ID3D12Fence> m_readFence;       // present queue: GPU reads of `depth` textures
    std::array<Slot, kMaxSlots> m_slots;
    int m_slotCount = 0;
    std::vector<Job> m_recorded;           // in the flow list being built
    int m_stride = 1;                      // network on every m_stride-th new frame
    uint64_t m_newFrames = 0;              // frames record() was given, for m_stride
    uint64_t m_lastFrameId = 0;            // the newest of them

    DepthEstimator m_estimator;
    ComPtr<ID3D12Fence> m_networkFence;    // signalled by the estimator's queue
    uint64_t m_networkValue = 0;
    ComPtr<ID3D12CommandQueue> m_queue;    // DepthFinishCS
    ComPtr<ID3D12CommandAllocator> m_allocator;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    DispatchContext m_dispatch;
    ComPtr<ID3D12Fence> m_finishFence;
    uint64_t m_finishValue = 0;
    HANDLE m_event = nullptr;

    mutable std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    std::condition_variable m_released;    // release(): a pinned texture was let go
    std::deque<Job> m_jobs;
    bool m_stop = false;
    std::thread m_thread;
};
