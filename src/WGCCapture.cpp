#include "WGCCapture.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <dxgi.h>

WGCCapture::WGCCapture() {
    m_frameEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
    } catch (...) {
        // Apartment might already be initialized on this thread
    }
}

WGCCapture::~WGCCapture() {
    stopCapture();
    if (m_frameEvent) {
        CloseHandle(m_frameEvent);
        m_frameEvent = nullptr;
    }
}

bool WGCCapture::initialize(ID3D11Device* d3d11Device) {
    if (!d3d11Device) return false;
    m_d3dDevice = d3d11Device;

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = m_d3dDevice.As(&dxgiDevice);
    if (FAILED(hr)) return false;

    hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), reinterpret_cast<IInspectable**>(winrt::put_abi(m_winrtDevice)));
    if (FAILED(hr)) {
        std::cerr << "Error: CreateDirect3D11DeviceFromDXGIDevice failed (0x" << std::hex << hr << ")\n";
        return false;
    }

    return true;
}

bool WGCCapture::prepareCapture(HWND targetWindow) {
    if (!m_winrtDevice || !targetWindow) return false;

    if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported()) {
        std::cerr << "Error: Windows.Graphics.Capture is not supported on this OS version.\n";
        return false;
    }

    m_targetWindow = targetWindow;

    try {
        auto interop = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        HRESULT hr = interop->CreateForWindow(
            m_targetWindow,
            winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
            winrt::put_abi(m_captureItem)
        );

        if (FAILED(hr) || !m_captureItem) {
            std::cerr << "Error: IGraphicsCaptureItemInterop::CreateForWindow failed (0x" << std::hex << hr << ")\n";
            return false;
        }

        auto itemSize = m_captureItem.Size();
        m_width = static_cast<uint32_t>(itemSize.Width);
        m_height = static_cast<uint32_t>(itemSize.Height);

        RECT clientRect = {};
        RECT windowRect = {};
        POINT clientOrigin = {};
        if (GetClientRect(m_targetWindow, &clientRect) &&
            GetWindowRect(m_targetWindow, &windowRect) &&
            ClientToScreen(m_targetWindow, &clientOrigin)) {
            uint32_t clientWidth = static_cast<uint32_t>(clientRect.right - clientRect.left);
            uint32_t clientHeight = static_cast<uint32_t>(clientRect.bottom - clientRect.top);
            uint32_t cropX = static_cast<uint32_t>(std::max(0L, clientOrigin.x - windowRect.left));
            uint32_t cropY = static_cast<uint32_t>(std::max(0L, clientOrigin.y - windowRect.top));
            if (clientWidth > 0 && clientHeight > 0 &&
                cropX + clientWidth <= static_cast<uint32_t>(itemSize.Width) &&
                cropY + clientHeight <= static_cast<uint32_t>(itemSize.Height)) {
                m_cropX = cropX;
                m_cropY = cropY;
                m_width = clientWidth;
                m_height = clientHeight;
            }
        }

        m_framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            m_winrtDevice,
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            4,
            itemSize
        );

        m_frameArrivedRevoker = m_framePool.FrameArrived(
            winrt::auto_revoke,
            { this, &WGCCapture::onFrameArrived }
        );

        m_session = m_framePool.CreateCaptureSession(m_captureItem);
        try {
            // The system cursor remains visible above the click-through
            // overlay; capturing it would interpolate stale cursor images.
            m_session.IsCursorCaptureEnabled(false);
        } catch (...) {
            // Not supported on older Windows 10 builds, ignore
        }
        try {
            m_session.IsBorderRequired(false);
        } catch (...) {
            // Available on Windows 10 build 2004+ / Windows 11
        }
        try {
            // Windows 11 24H2 otherwise limits some captures to 60 updates/s.
            m_session.MinUpdateInterval(std::chrono::milliseconds(1));
        } catch (...) {
            // Available only on recent Windows builds.
        }

        std::cout << "Windows Graphics Capture prepared successfully (client "
                  << m_width << "x" << m_height << ", crop "
                  << m_cropX << "," << m_cropY << ").\n";
        return true;
    } catch (const winrt::hresult_error& e) {
        m_isCapturing = false;
        std::cerr << "WinRT Exception in startCapture: " << winrt::to_string(e.message()) << "\n";
        return false;
    } catch (...) {
        m_isCapturing = false;
        std::cerr << "Unknown exception in startCapture.\n";
        return false;
    }
}

bool WGCCapture::startCapture() {
    if (!m_session || !m_framePool || !m_frameEvent) return false;
    try {
        m_isCapturing = true;
        ResetEvent(m_frameEvent);
        m_arrivalCount = 0;
        m_drainedFrameCount = 0;
        // StartCapture can synchronously invoke the free-threaded callback.
        // Enable it first so the initial WGC frames always wake the render loop.
        m_session.StartCapture();
        std::cout << "Windows Graphics Capture started successfully.\n";
        return true;
    } catch (const winrt::hresult_error& e) {
        m_isCapturing = false;
        std::cerr << "WinRT Exception starting capture: " << winrt::to_string(e.message()) << "\n";
        return false;
    } catch (...) {
        m_isCapturing = false;
        std::cerr << "Unknown exception starting capture.\n";
        return false;
    }
}

void WGCCapture::stopCapture() {
    m_isCapturing = false;
    m_frameArrivedRevoker.revoke();

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (m_latestFrame) m_latestFrame.Close();
        m_latestFrame = nullptr;
    }

    if (m_session) {
        m_session.Close();
        m_session = nullptr;
    }
    if (m_framePool) {
        m_framePool.Close();
        m_framePool = nullptr;
    }
    m_captureItem = nullptr;
    m_nextAcceptedTimestamp100ns = 0;
    m_cadenceFps = 0;
    m_lastObservedTimestamp100ns = 0;
    m_smoothedFrameInterval100ns = 0.0;
    m_detectedSourceFps = 30;
    m_candidateSourceFps = 30;
    m_candidateSampleCount = 0;
    m_arrivalCount = 0;
    m_drainedFrameCount = 0;
    if (m_frameEvent) ResetEvent(m_frameEvent);
}

void WGCCapture::onFrameArrived(
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
    winrt::Windows::Foundation::IInspectable const&
) {
    if (!m_isCapturing) return;

    ++m_arrivalCount;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame newestFrame{ nullptr };
    while (auto frame = sender.TryGetNextFrame()) {
        ++m_drainedFrameCount;
        if (newestFrame) newestFrame.Close();
        newestFrame = frame;
    }
    if (!newestFrame) return;

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (m_latestFrame) m_latestFrame.Close();
        m_latestFrame = newestFrame;
    }
    if (m_frameEvent) SetEvent(m_frameEvent);
}

bool WGCCapture::copyLatestFrame(
    ID3D11DeviceContext* context,
    ID3D11Texture2D* destination,
    uint32_t sourceFps,
    int64_t& timestamp100ns
) {
    // Do not consume the latest WGC frame or advance the cadence clock until
    // the presenter has a slot to receive it. Cross-adapter operation can keep
    // all timeline slots occupied for several ticks; dropping here previously
    // reduced dGPU capture to roughly two unique frames per second.
    if (!context || !destination || !m_framePool) return false;

    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame newestFrame{ nullptr };
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (m_latestFrame) {
            newestFrame = m_latestFrame;
            m_latestFrame = nullptr;
        }
    }
    if (!newestFrame) return false;

    auto relativeTime = newestFrame.SystemRelativeTime();
    timestamp100ns = relativeTime.count();
    if (timestamp100ns > 0) {
        if (m_lastObservedTimestamp100ns > 0) {
            int64_t observedInterval = timestamp100ns - m_lastObservedTimestamp100ns;
            if (observedInterval >= 80000 && observedInterval <= 1000000) {
                if (m_smoothedFrameInterval100ns == 0.0) {
                    m_smoothedFrameInterval100ns = static_cast<double>(observedInterval);
                } else {
                    m_smoothedFrameInterval100ns =
                        m_smoothedFrameInterval100ns * 0.9 +
                        static_cast<double>(observedInterval) * 0.1;
                }

                double measuredFps = 10000000.0 / m_smoothedFrameInterval100ns;
                uint32_t candidate = measuredFps >= 45.0 ? 60u :
                                     measuredFps >= 27.0 ? 30u : 24u;
                if (candidate == m_candidateSourceFps) {
                    ++m_candidateSampleCount;
                } else {
                    m_candidateSourceFps = candidate;
                    m_candidateSampleCount = 1;
                }
                if (m_candidateSampleCount >= 20 && candidate != m_detectedSourceFps) {
                    m_detectedSourceFps = candidate;
                    m_candidateSampleCount = 0;
                }
            }
        }
        m_lastObservedTimestamp100ns = timestamp100ns;

        uint32_t effectiveSourceFps = sourceFps > 0 ? sourceFps : m_detectedSourceFps;
        if (m_cadenceFps != effectiveSourceFps) {
            m_cadenceFps = effectiveSourceFps;
            m_nextAcceptedTimestamp100ns = 0;
        }
        int64_t interval = 10000000LL / static_cast<int64_t>(effectiveSourceFps);
        int64_t tolerance = interval / 10;
        if (m_nextAcceptedTimestamp100ns > 0 &&
            timestamp100ns + tolerance < m_nextAcceptedTimestamp100ns) {
            // Chronic rejection means the schedule drifted behind the source
            // (jitter, rate re-detection). Without a rebase every later frame
            // would be rejected forever and commits would stall permanently.
            // Genuine downsampling (60 -> 30) never rejects more than one
            // frame in a row, so a short run of rejections is safe to reset.
            if (++m_cadenceRejectionRun >= 5) {
                m_nextAcceptedTimestamp100ns = timestamp100ns + interval;
                m_cadenceRejectionRun = 0;
            }
            newestFrame.Close();
            return false;
        }
        m_cadenceRejectionRun = 0;

        if (m_nextAcceptedTimestamp100ns == 0) {
            m_nextAcceptedTimestamp100ns = timestamp100ns + interval;
        } else {
            do {
                m_nextAcceptedTimestamp100ns += interval;
            } while (m_nextAcceptedTimestamp100ns <= timestamp100ns + tolerance);
        }
    }

    auto surface = newestFrame.Surface();
    auto access = surface.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    Microsoft::WRL::ComPtr<ID3D11Texture2D> newTexture;
    HRESULT hr = access->GetInterface(IID_PPV_ARGS(&newTexture));

    if (FAILED(hr) || !newTexture) {
        newestFrame.Close();
        return false;
    }

    D3D11_TEXTURE2D_DESC desc;
    newTexture->GetDesc(&desc);

    D3D11_TEXTURE2D_DESC destinationDesc = {};
    destination->GetDesc(&destinationDesc);
    if (destinationDesc.Width != m_width || destinationDesc.Height != m_height ||
        destinationDesc.Format != desc.Format) {
        newestFrame.Close();
        return false;
    }

    D3D11_BOX sourceBox = {};
    sourceBox.left = m_cropX;
    sourceBox.top = m_cropY;
    sourceBox.front = 0;
    sourceBox.right = m_cropX + m_width;
    sourceBox.bottom = m_cropY + m_height;
    sourceBox.back = 1;

    context->CopySubresourceRegion(destination, 0, 0, 0, 0, newTexture.Get(), 0, &sourceBox);
    context->Flush();

    newestFrame.Close();

    return true;
}
