#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <atomic>
#include <memory>
#include <mutex>

class WGCCapture {
public:
    WGCCapture();
    ~WGCCapture();

    bool initialize(ID3D11Device* d3d11Device);
    bool prepareCapture(HWND targetWindow);
    bool startCapture();
    void stopCapture();

    // Drains the WGC pool and copies the newest cadence-eligible surface into
    // a queue-owned texture before closing the WGC frame.
    bool copyLatestFrame(
        ID3D11DeviceContext* context,
        ID3D11Texture2D* destination,
        uint32_t sourceFps,
        int64_t& timestamp100ns
    );

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    uint32_t getDetectedSourceFps() const { return m_detectedSourceFps; }
    bool isCapturing() const { return m_isCapturing; }
    HWND getTargetWindow() const { return m_targetWindow; }
    HANDLE getFrameEvent() const { return m_frameEvent; }
    uint64_t getArrivalCount() const { return m_arrivalCount.load(); }
    uint64_t getDrainedFrameCount() const { return m_drainedFrameCount.load(); }

private:
    void onFrameArrived(
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
        winrt::Windows::Foundation::IInspectable const& args
    );

    Microsoft::WRL::ComPtr<ID3D11Device>        m_d3dDevice;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem         m_captureItem{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool  m_framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession       m_session{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame       m_latestFrame{ nullptr };

    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker m_frameArrivedRevoker;

    HWND m_targetWindow = nullptr;
    std::atomic<bool> m_isCapturing{ false };
    std::atomic<uint64_t> m_arrivalCount{ 0 };
    std::atomic<uint64_t> m_drainedFrameCount{ 0 };
    std::mutex m_frameMutex;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_cropX = 0;
    uint32_t m_cropY = 0;
    HANDLE m_frameEvent = nullptr;
    int64_t m_nextAcceptedTimestamp100ns = 0;
    uint32_t m_cadenceFps = 0;
    int64_t m_lastObservedTimestamp100ns = 0;
    double m_smoothedFrameInterval100ns = 0.0;
    uint32_t m_detectedSourceFps = 30;
    uint32_t m_candidateSourceFps = 30;
    uint32_t m_candidateSampleCount = 0;
    uint32_t m_cadenceRejectionRun = 0;
};
