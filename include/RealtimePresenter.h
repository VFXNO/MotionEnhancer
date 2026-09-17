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
        uint32_t sourceFps
    );

    ID3D11Texture2D* acquireCaptureTarget();
    bool commitCapturedFrame(int64_t timestamp100ns);
    void cancelCaptureTarget();
    void setSourceFps(uint32_t sourceFps);
    bool presentNext(GPUInterpolator& interpolator);

    HANDLE frameLatencyHandle() const;
    HANDLE pacingHandle() const { return m_pacingTimer; }
    uint64_t renderedFrameIndex() const { return m_renderedFrameIndex; }
    uint64_t queuedFrameIndex() const { return m_queuedFrameIndex; }
    uint64_t presentedFrameIndex() const { return m_presentedFrameIndex; }
    size_t queueDepth() const;
    float interpolationFactor() const { return m_interpolationFactor; }
    double refreshRate() const { return m_refreshRate; }

private:
    enum class SlotState { Free, CopyPending, Ready };
    struct FrameSlot {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<ID3D11Query> completion;
        SlotState state = SlotState::Free;
        uint64_t index = 0;
        int64_t timestamp100ns = 0;
    };

    void updateCompletions();
    std::vector<int> sortedReadySlots() const;
    void retireConsumedFrames(const std::vector<int>& sorted, int currentSlot);
    bool presentTexture(GPUInterpolator& interpolator, FrameSlot& slot);
    bool scheduleNextPresentation();
    void revealOutput();

    std::shared_ptr<D3D11Context> m_context;
    std::array<FrameSlot, 3> m_slots;
    int m_pendingWriteSlot = -1;
    int m_cachedPreviousSlot = -1;
    int m_cachedCurrentSlot = -1;
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

    HANDLE m_pacingTimer = nullptr;
    int64_t m_qpcFrequency = 0;
    int64_t m_nextPresentationQpc = 0;
    HWND m_outputWindow = nullptr;
    bool m_outputVisible = false;
};
