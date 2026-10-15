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
// 0 = far, 1 = near; DepthEstimator) that the motion projection uses to put
// the nearer of two equally plausible surfaces in front (ProjectSplatCS).
//
// The flow list shrinks each new frame to kSize squared (DepthInputCS) and
// copies it to a readback buffer; once that list has run, a worker thread
// runs the network and uploads the result on a copy queue of its own. The
// presenter only ever takes a frame's depth once it is complete (depthFor),
// so nothing on the way to the screen waits for the network: a frame whose
// depth is not ready yet is interpolated without it.
class DepthService {
public:
    static constexpr int kSize = DepthEstimator::kSize;
    static constexpr int kMaxSlots = 16;

    ~DepthService();

    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* flowFence, int slots);

    // Records the shrink and readback of `frame` into the flow list, unless
    // this frame's depth is already done or pending. Returns false only on a
    // recording failure.
    bool record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame,
                uint32_t width, uint32_t height);
    // The flow list holding everything record() added since the last call
    // was submitted with this fence value / was thrown away.
    void submitted(uint64_t flowFenceValue);
    void discarded();

    // The frame's nearness texture once complete, else null. Non-blocking.
    ID3D12Resource* depthFor(const FlowInput& frame) const;
    // Offline: waits up to timeoutMs for the frame's depth.
    ID3D12Resource* waitFor(const FlowInput& frame, unsigned timeoutMs);

private:
    struct Slot {
        ComPtr<ID3D12Resource> shrunk;     // kSize^2 RGBA8, DepthInputCS output
        ComPtr<ID3D12Resource> readback;   // the same, for the CPU
        ComPtr<ID3D12Resource> upload;     // nearness for the copy queue
        ComPtr<ID3D12Resource> depth;      // kSize^2 R32_FLOAT, what shaders read
        uint64_t pendingId = UINT64_MAX;   // frame being produced
        uint64_t readyId = UINT64_MAX;     // frame whose depth `depth` holds
    };
    struct Job {
        int slot = 0;
        uint64_t frameId = 0;
        uint64_t flowFence = 0;
    };
    void run();
    bool produce(const Job& job);

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    ComPtr<ID3D12Fence> m_flowFence;
    std::array<Slot, kMaxSlots> m_slots;
    int m_slotCount = 0;
    std::vector<Job> m_recorded;           // in the flow list being built

    DepthEstimator m_estimator;
    ComPtr<ID3D12CommandQueue> m_copyQueue;
    ComPtr<ID3D12CommandAllocator> m_allocator;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    ComPtr<ID3D12Fence> m_copyFence;
    uint64_t m_copyValue = 0;
    HANDLE m_event = nullptr;

    mutable std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    std::deque<Job> m_jobs;
    bool m_stop = false;
    std::thread m_thread;
};
