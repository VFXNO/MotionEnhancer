#pragma once

#include "ComputeKernels.h"
#include "UiRegions.h"

#include <memory>
#include <string>

struct FlowInput;

// Optional region detector, independent of the flow estimator and blend mask.
// GPU preprocessing is recorded with flow, at most 5Hz. One worker queues
// inference on its own queue and returns only small CPU box snapshots.
// Optional Vulkan matrix input transfers also belong to that worker.
// No network inference, readback wait or result wait on the capture thread.
// Current-proposal continuity only; missing proposals are never held over.
class UiDetector {
public:
    enum class Choice { Automatic, DirectMLIcons, MatrixIcons };
    struct Snapshot {
        bool valid = false;
        uint64_t frameId = 0;
        double milliseconds = 0;
        std::vector<UiRegion> regions;
        unsigned matrixDispatches = 0;
        unsigned verifiedWmmaDispatches = 0;
        uint64_t diagnosticChecksum = 0; // opt-in offline replay test only
    };
    UiDetector();
    ~UiDetector();
    static std::wstring defaultModelPath(Choice choice = Choice::Automatic);
    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* inputFence,
                     uint32_t width, uint32_t height, Choice choice = Choice::Automatic);
    bool record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame,
                bool force = false); // force bypasses the rate limit for offline tests only
    void submitted(uint64_t fenceValue);
    void discarded();
    void invalidate(); // drops cached regions and rejects an older in-flight job
    Snapshot snapshot(uint64_t currentFrameId) const; // empty/invalid after 600ms
    bool waitIdle(unsigned timeoutMs); // offline diagnostics ONLY
    std::string finishProfiling(); // Offline only, after waitIdle; empty when not requested.
    const char* backendName() const;
    const char* detectorName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
