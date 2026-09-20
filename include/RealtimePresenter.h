#pragma once

#include "D3D11Context.h"
#include "GPUInterpolator.h"

#include <array>
#include <memory>
#include <vector>

class RealtimePresenter {
public:
    ~RealtimePresenter();
    bool initialize(
        std::shared_ptr<D3D11Context> context,
        HWND outputWindow,
        uint32_t width,
        uint32_t height,
        uint32_t sourceFps,
        uint32_t outputMultiplier
    );

    ID3D11Texture2D* acquireCaptureTarget();
    bool commitCapturedFrame(int64_t timestamp100ns);
    void cancelCaptureTarget();
    void setSourceFps(uint32_t sourceFps);
    bool presentNext(GPUInterpolator& interpolator);
    bool prefetchPair(GPUInterpolator& interpolator);

    HANDLE frameLatencyHandle() const;
    HANDLE pacingHandle() const { return m_pacingTimer; }
    uint64_t renderedFrameIndex() const { return m_renderedFrameIndex; }
    uint64_t queuedFrameIndex() const { return m_queuedFrameIndex; }
    uint64_t presentedFrameIndex() const { return m_presentedFrameIndex; }
    size_t queueDepth() const;
    float interpolationFactor() const { return m_interpolationFactor; }
    double refreshRate() const { return m_refreshRate; }
    double outputRate() const { return m_outputRate; }

private:
    enum class SlotState { Free, Ready };
    struct FrameSlot {
        ComPtr<ID3D11Texture2D> texture;      // D3D11 capture side (shared)
        ComPtr<ID3D12Resource> texture12;     // same allocation opened on D3D12
        SlotState state = SlotState::Free;
        uint64_t index = 0;
        int64_t timestamp100ns = 0;
    };

    static constexpr size_t kSlotCount = 3;
    // Fills out with ready slot indices sorted by timestamp; returns the
    // count. Fixed storage: called on every pacing tick, so it must not
    // allocate.
    size_t sortedReadySlots(int (&out)[kSlotCount]) const;
    void retireConsumedFrames(const int* sorted, size_t count, int currentSlot);
    bool presentTexture(GPUInterpolator& interpolator, FrameSlot& slot);
    bool scheduleNextPresentation();
    void updateOutputCadence();
    void revealOutput();
    void logPacing(const char* mode);

    std::shared_ptr<D3D11Context> m_context;
    std::array<FrameSlot, 3> m_slots;
    int m_pendingWriteSlot = -1;
    int m_cachedPreviousSlot = -1;
    int m_cachedCurrentSlot = -1;
    int m_pendingPreviousSlot = -1;
    int m_pendingCurrentSlot = -1;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    uint64_t m_renderedFrameIndex = 0;
    uint64_t m_queuedFrameIndex = 0;
    uint64_t m_presentedFrameIndex = 0;
    int64_t m_previousFrameTimestamp = 0;
    int64_t m_currentFrameTimestamp = 0;
    int64_t m_targetPresentationTimestamp = 0;
    int64_t m_refreshInterval100ns = 166667;
    int64_t m_nominalSourceInterval100ns = 333333;
    float m_interpolationFactor = 0.0f;
    double m_refreshRate = 60.0;
    double m_outputRate = 60.0;
    uint32_t m_sourceFps = 30;
    uint32_t m_pendingSourceFps = 0;
    uint32_t m_pendingSourceFpsSamples = 0;
    uint32_t m_outputMultiplier = 2;

    HANDLE m_pacingTimer = nullptr;
    int64_t m_qpcFrequency = 0;
    int64_t m_nextPresentationQpc = 0;
    LARGE_INTEGER m_lastPresentQpc = {};
    HWND m_outputWindow = nullptr;
    bool m_outputVisible = false;
};
