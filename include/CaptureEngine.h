#pragma once

#include "FlowEngine.h"
#include "FrameTimeline.h"
#include "GraphicsDevice.h"

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// Windows Graphics Capture -> slot ring -> timeline + flow queue.
//
// The WGC callback only queues the frame; a dedicated producer thread owns
// the D3D11 immediate context and does, per frame: copy into a free slot,
// GPU luma diff against the newest frame (exact-repaint rejection and
// scene-cut detection), cadence classification through the timeline, and
// flow submission. The render thread never touches D3D11.
class CaptureEngine {
public:
    static constexpr int kSlots = 8;

    struct Stats {
        uint64_t arrivals = 0;      // WGC frames delivered
        uint64_t duplicates = 0;    // identical repaints rejected
        uint64_t queueDrops = 0;    // no free slot / callback queue overflow
        uint64_t sizeMismatch = 0;
        uint64_t accepted = 0;
        uint64_t sceneCuts = 0;
        double lastDiffMs = 0.0;
    };

    CaptureEngine();
    ~CaptureEngine();

    bool initialize(GraphicsDevice* device, FlowEngine* flow, FrameTimeline* timeline);
    bool prepare(HWND target);
    // The present-queue fence gates slot recycling; bind it before start().
    void setPresentFence(ID3D12Fence* fence) { m_presentFence = fence; }
    bool start();
    void stop();

    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    // Signaled whenever a frame is appended to the timeline.
    HANDLE acceptedEvent() const { return m_acceptedEvent; }

    // ---- render thread ----
    FlowInput slotInput(int slot) const;
    void noteSlotPresented(int slot, uint64_t presentFenceValue);
    void releaseSlots(const std::vector<int>& slots);
    Stats stats() const;

private:
    enum class SlotState { Free, Filling, Live, Retiring };
    struct Slot {
        SharedTexture texture;
        SlotState state = SlotState::Free;
        uint64_t readyValue = 0;
        uint64_t captureId = 0;
        uint64_t flowFence = 0;
        uint64_t presentFence = 0;
    };
    struct QueuedFrame {
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame frame{nullptr};
        int64_t arrival = 0;
    };
    struct PendingFrame {
        int slot = -1;
        int64_t tsRaw = 0;
        int64_t arrival = 0;
        bool sceneCut = false;
    };
    struct DiffResult {
        float changedFraction = 1.0f;
        float meanAbsDiff = 255.0f;
    };

    void onFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
                        winrt::Windows::Foundation::IInspectable const&);
    void producerLoop();
    void processFrame(QueuedFrame& queued);
    void acceptFrame(const PendingFrame& pending);
    int acquireSlot();
    void releaseSlot(int slot);
    void recycleRetiringSlots();
    bool createDiffResources();
    bool runDiff(int slot, int referenceSlot, DiffResult& out);

    GraphicsDevice* m_device = nullptr;
    FlowEngine* m_flow = nullptr;
    FrameTimeline* m_timeline = nullptr;
    ComPtr<ID3D12Fence> m_presentFence;

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_pool{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker m_revoker;

    HWND m_target = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_cropX = 0;
    uint32_t m_cropY = 0;

    // callback -> producer queue
    std::mutex m_queueMutex;
    std::deque<QueuedFrame> m_queue;
    HANDLE m_frameEvent = nullptr;
    HANDLE m_acceptedEvent = nullptr;
    std::thread m_producer;
    std::atomic<bool> m_running{false};

    // slots
    mutable std::mutex m_slotMutex;
    std::array<Slot, kSlots> m_slots;
    uint64_t m_captureCounter = 0;
    int m_referenceSlot = -1;   // newest non-duplicate frame (diff reference)
    bool m_haveDeferred = false;
    PendingFrame m_deferred;
    int64_t m_deferDeadline = 0;

    // D3D11 luma diff
    ComPtr<ID3D11ComputeShader> m_diffShader;
    ComPtr<ID3D11Buffer> m_diffConstants;
    ComPtr<ID3D11Buffer> m_diffOutput;
    ComPtr<ID3D11UnorderedAccessView> m_diffOutputUav;
    ComPtr<ID3D11Buffer> m_diffStaging;
    std::array<ComPtr<ID3D11ShaderResourceView>, kSlots> m_slotSrv;

    mutable std::mutex m_statsMutex;
    Stats m_stats;
};
