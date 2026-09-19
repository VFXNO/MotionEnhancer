#include "RealtimePresenter.h"

#include "D3D12Context.h"

#include <algorithm>
#include <dwmapi.h>
#include <iostream>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

RealtimePresenter::~RealtimePresenter() {
    if (m_pacingTimer) {
        CancelWaitableTimer(m_pacingTimer);
        CloseHandle(m_pacingTimer);
        m_pacingTimer = nullptr;
    }
}

bool RealtimePresenter::initialize(
    std::shared_ptr<D3D11Context> context,
    HWND outputWindow,
    uint32_t width,
    uint32_t height,
    uint32_t sourceFps,
    uint32_t outputMultiplier
) {
    m_context = std::move(context);
    m_outputWindow = outputWindow;
    m_width = width;
    m_height = height;
    m_outputMultiplier = outputMultiplier;
    m_sourceFps = sourceFps > 0 ? sourceFps : 30u;
    m_nominalSourceInterval100ns = 10000000LL / static_cast<int64_t>(m_sourceFps);
    if (!m_context || !m_context->device || !outputWindow || width == 0 || height == 0) {
        return false;
    }
    if (!m_context->createSwapChain(outputWindow, width, height)) return false;

    DWM_TIMING_INFO timing = {};
    timing.cbSize = sizeof(timing);
    if (SUCCEEDED(DwmGetCompositionTimingInfo(nullptr, &timing)) &&
        timing.rateRefresh.uiNumerator > 0 && timing.rateRefresh.uiDenominator > 0) {
        m_refreshRate = static_cast<double>(timing.rateRefresh.uiNumerator) /
                        static_cast<double>(timing.rateRefresh.uiDenominator);
    }
    updateOutputCadence();
    LARGE_INTEGER frequency = {};
    LARGE_INTEGER now = {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    m_qpcFrequency = frequency.QuadPart;
    m_nextPresentationQpc = now.QuadPart;
    m_pacingTimer = CreateWaitableTimerExW(
        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!m_pacingTimer || m_qpcFrequency <= 0) return false;

    for (auto& slot : m_slots) {
        // Capture slots are NT-handle shared with D3D12, which consumes them
        // after the shared capture-ready fence is signaled.
        if (!m_context->createSharedTexture2D(
                width, height, DXGI_FORMAT_B8G8R8A8_UNORM,
                D3D11_BIND_SHADER_RESOURCE, slot.texture, slot.texture12)) {
            return false;
        }
    }

    // Prime the flip-model chain once. Every subsequent Present is preceded by
    // one successful frame-latency wait and no back-buffer reference is retained.
    if (m_context->nativePresentation()) {
        if (!m_context->nativeD3D12->backBuffer(0)) return false;
    } else {
        ComPtr<ID3D11Texture2D> backBuffer;
        if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
        ComPtr<ID3D11RenderTargetView> rtv;
        if (FAILED(m_context->device->CreateRenderTargetView(
                backBuffer.Get(), nullptr, rtv.GetAddressOf()))) return false;
        const float clear[4] = {};
        m_context->context->ClearRenderTargetView(rtv.Get(), clear);
        backBuffer.Reset();
        rtv.Reset();
    }
    if (FAILED(m_context->presentSwapChain())) return false;
    return scheduleNextPresentation();
}

void RealtimePresenter::setSourceFps(uint32_t sourceFps) {
    if (sourceFps == 0 || sourceFps == m_sourceFps) {
        m_pendingSourceFps = 0;
        m_pendingSourceFpsSamples = 0;
        return;
    }

    // Auto-detection can oscillate by one FPS between measurements. Require a
    // short stable run before changing the presentation clock.
    if (m_pendingSourceFps != sourceFps) {
        m_pendingSourceFps = sourceFps;
        m_pendingSourceFpsSamples = 1;
        return;
    }
    if (++m_pendingSourceFpsSamples < 8) return;

    m_sourceFps = sourceFps;
    m_nominalSourceInterval100ns = 10000000LL / static_cast<int64_t>(sourceFps);
    m_pendingSourceFps = 0;
    m_pendingSourceFpsSamples = 0;
    updateOutputCadence();
}

void RealtimePresenter::updateOutputCadence() {
    m_outputRate = m_outputMultiplier == 0
        ? m_refreshRate
        : std::min(m_refreshRate, static_cast<double>(m_sourceFps * m_outputMultiplier));
    m_outputRate = std::max(1.0, m_outputRate);
    m_refreshInterval100ns = static_cast<int64_t>(10000000.0 / m_outputRate + 0.5);
}

bool RealtimePresenter::scheduleNextPresentation() {
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    int64_t qpcStep = static_cast<int64_t>(
        static_cast<double>(m_qpcFrequency) / m_outputRate + 0.5);
    // Advance the cadence grid from the SCHEDULED time, never from "now":
    // phase stays locked to the original grid, so an overrunning present
    // shortens the next interval slightly instead of producing the
    // long-short-long judder of deadline skipping.
    do {
        m_nextPresentationQpc += qpcStep;
    } while (m_nextPresentationQpc <= now.QuadPart);

    int64_t remainingQpc = std::max<int64_t>(1, m_nextPresentationQpc - now.QuadPart);
    LARGE_INTEGER due = {};
    due.QuadPart = -std::max<int64_t>(
        1, remainingQpc * 10000000LL / m_qpcFrequency);
    return SetWaitableTimer(m_pacingTimer, &due, 0, nullptr, nullptr, FALSE) != FALSE;
}

HANDLE RealtimePresenter::frameLatencyHandle() const {
    return m_context ? m_context->frameLatencyHandle() : nullptr;
}

ID3D11Texture2D* RealtimePresenter::acquireCaptureTarget() {
    if (m_pendingWriteSlot >= 0) return nullptr;

    for (size_t i = 0; i < m_slots.size(); ++i) {
        if (m_slots[i].state == SlotState::Free) {
            m_pendingWriteSlot = static_cast<int>(i);
            return m_slots[i].texture.Get();
        }
    }
    // Queue pressure drops the incoming capture. Active timeline frames are
    // never overwritten, keeping latency bounded without corrupting flow input.
    return nullptr;
}

bool RealtimePresenter::commitCapturedFrame(int64_t timestamp100ns) {
    if (m_pendingWriteSlot < 0 || timestamp100ns <= 0) return false;
    auto& slot = m_slots[static_cast<size_t>(m_pendingWriteSlot)];
    slot.timestamp100ns = timestamp100ns;
    slot.index = ++m_renderedFrameIndex;
    // Publish the D3D11 copy to D3D12 through the shared fence before the
    // slot becomes visible to the interpolation timeline. The slot can be
    // CPU-visible immediately: every D3D12 consumer queue-waits on this exact
    // fence value, so a separate D3D11 event query is redundant and, on the
    // NVIDIA cross-adapter path, sometimes never reports completion.
    if (m_context->signalCaptureReady(slot.texture.Get()) == 0) {
        slot.state = SlotState::Free;
        slot.timestamp100ns = 0;
        m_pendingWriteSlot = -1;
        return false;
    }
    slot.state = SlotState::Ready;
    m_queuedFrameIndex = std::max(m_queuedFrameIndex, slot.index);
    m_pendingWriteSlot = -1;
    return true;
}

void RealtimePresenter::cancelCaptureTarget() {
    if (m_pendingWriteSlot < 0) return;
    auto& slot = m_slots[static_cast<size_t>(m_pendingWriteSlot)];
    slot.state = SlotState::Free;
    slot.timestamp100ns = 0;
    m_pendingWriteSlot = -1;
}

void RealtimePresenter::revealOutput() {
    if (m_outputVisible || !m_outputWindow) return;
    SetLayeredWindowAttributes(m_outputWindow, 0, 255, LWA_ALPHA);
    ShowWindow(m_outputWindow, SW_SHOWNOACTIVATE);
    m_outputVisible = true;
}

std::vector<int> RealtimePresenter::sortedReadySlots() const {
    std::vector<int> result;
    for (size_t i = 0; i < m_slots.size(); ++i) {
        if (m_slots[i].state == SlotState::Ready) result.push_back(static_cast<int>(i));
    }
    std::sort(result.begin(), result.end(), [&](int left, int right) {
        return m_slots[static_cast<size_t>(left)].timestamp100ns <
               m_slots[static_cast<size_t>(right)].timestamp100ns;
    });
    return result;
}

size_t RealtimePresenter::queueDepth() const {
    size_t depth = 0;
    for (const auto& slot : m_slots) {
        if (slot.state != SlotState::Free) ++depth;
    }
    return depth;
}

void RealtimePresenter::retireConsumedFrames(
    const std::vector<int>& sorted,
    int currentSlot
) {
    if (sorted.size() < 3) return;
    for (int slotIndex : sorted) {
        if (slotIndex == currentSlot) break;
        if (slotIndex == m_cachedPreviousSlot || slotIndex == m_cachedCurrentSlot ||
            slotIndex == m_pendingPreviousSlot || slotIndex == m_pendingCurrentSlot) {
            continue;
        }
        m_slots[static_cast<size_t>(slotIndex)].state = SlotState::Free;
    }
}

bool RealtimePresenter::presentTexture(
    GPUInterpolator& interpolator,
    FrameSlot& slot
) {
    if (m_context->nativePresentation()) {
        ComPtr<ID3D12Resource> backBuffer = m_context->nativeD3D12->backBuffer(m_context->nativeD3D12->currentBackBufferIndex());
        if (!backBuffer) {
            std::cerr << "Presenter: D3D12 back buffer unavailable.\n";
            return false;
        }
        if (!interpolator.presentSourceFrameNative(slot.texture.Get(), backBuffer.Get())) {
            std::cerr << "Presenter: presentSourceFrameNative failed.\n";
            return false;
        }
        backBuffer.Reset();
        if (FAILED(m_context->presentSwapChain())) return false;
        revealOutput();
        return true;
    }
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
    if (!interpolator.presentSourceFrame(slot.texture.Get(), backBuffer.Get())) return false;
    backBuffer.Reset();
    if (FAILED(m_context->presentSwapChain())) return false;
    revealOutput();
    return true;
}

bool RealtimePresenter::prefetchPair(GPUInterpolator& interpolator) {
    // Run the expensive flow dispatch on the capture cadence instead of on a
    // presentation tick: mirror presentNext's pair selection for the current
    // (not yet advanced) target timestamp and prepare a changed pair without
    // presenting. presentNext finds the cached slots and skips preparation.
    if (m_pendingPreviousSlot >= 0) {
        if (!interpolator.framePairReady()) return false;
        m_cachedPreviousSlot = m_pendingPreviousSlot;
        m_cachedCurrentSlot = m_pendingCurrentSlot;
        m_pendingPreviousSlot = -1;
        m_pendingCurrentSlot = -1;
        return true;
    }

    auto sorted = sortedReadySlots();
    if (sorted.size() < 2 || m_targetPresentationTimestamp == 0) return false;

    int previous = sorted.front();
    int current = -1;
    for (int slotIndex : sorted) {
        const auto& slot = m_slots[static_cast<size_t>(slotIndex)];
        if (slot.timestamp100ns <= m_targetPresentationTimestamp) previous = slotIndex;
        // At an exact source timestamp presentation shows that source frame,
        // but flow can already advance to the following pair. Strict `>`
        // gives preparation the endpoint tick as additional lead time.
        if (slot.timestamp100ns > m_targetPresentationTimestamp) {
            current = slotIndex;
            break;
        }
    }
    // No slot at/after the target: the timeline holds on the newest frame, so
    // there is no complete pair to prepare.
    if (current < 0 || current == previous) return false;
    if (m_cachedPreviousSlot == previous && m_cachedCurrentSlot == current) return false;

    auto& previousFrame = m_slots[static_cast<size_t>(previous)];
    auto& currentFrame = m_slots[static_cast<size_t>(current)];
    if (!interpolator.prepareFramePair(
            previousFrame.texture.Get(), currentFrame.texture.Get(),
            previousFrame.index, currentFrame.index)) {
        return false;
    }
    m_pendingPreviousSlot = previous;
    m_pendingCurrentSlot = current;
    return true;
}

bool RealtimePresenter::presentNext(GPUInterpolator& interpolator) {
    auto sorted = sortedReadySlots();
    const char* mode = sorted.empty() ? "E" : nullptr;
    if (sorted.empty()) {
        // A consumed latency signal must always be paired with Present or the
        // waitable swapchain will not produce another presentation slot.
        // Presenting an unrendered back buffer flashes stale/black content, so
        // re-render the retained last output when one exists.
        if (!m_context->nativePresentation() ||
            !m_context->nativeD3D12->rePresentLastOutput()) {
            if (FAILED(m_context->presentSwapChain())) return false;
            logPacing(mode);
            return scheduleNextPresentation();
        }
        if (FAILED(m_context->presentSwapChain())) return false;
        revealOutput();
        logPacing(mode);
        return scheduleNextPresentation();
    }

    if (m_targetPresentationTimestamp == 0) {
        m_targetPresentationTimestamp =
            m_slots[static_cast<size_t>(sorted.front())].timestamp100ns;
    }

    int previous = sorted.front();
    int current = -1;
    for (int slotIndex : sorted) {
        const auto& slot = m_slots[static_cast<size_t>(slotIndex)];
        if (slot.timestamp100ns <= m_targetPresentationTimestamp) previous = slotIndex;
        if (slot.timestamp100ns >= m_targetPresentationTimestamp) {
            current = slotIndex;
            break;
        }
    }

    bool holdTimeline = false;
    if (current < 0) {
        current = sorted.back();
        previous = current;
        // Keep the target timestamp when the next source frame has not
        // arrived yet. Advancing here skips the next interpolation midpoint.
        holdTimeline = true;
        // The completed pair is no longer needed once the media timeline is
        // beyond its newest endpoint. Leaving it cached pins two old slots;
        // together with this newest frame that exhausts the three-slot queue
        // and prevents WGC from ever delivering the next frame (especially
        // visible through the slower cross-adapter dGPU path).
        if (m_pendingPreviousSlot < 0) {
            m_cachedPreviousSlot = -1;
            m_cachedCurrentSlot = -1;
        }
    }

    auto& previousFrame = m_slots[static_cast<size_t>(previous)];
    auto& currentFrame = m_slots[static_cast<size_t>(current)];
    m_previousFrameTimestamp = previousFrame.timestamp100ns;
    m_currentFrameTimestamp = currentFrame.timestamp100ns;

    bool presented = false;
    int64_t duration = m_currentFrameTimestamp - m_previousFrameTimestamp;
    if (previous != current && duration > 0) {
        int64_t interpolationStart = m_previousFrameTimestamp;
        if (duration > m_nominalSourceInterval100ns * 3 / 2) {
            // A long capture gap is starvation/discontinuity, not slow motion.
            // Rebase near the newest frame and consume motion within one
            // configured source interval instead of replaying the whole gap.
            interpolationStart = m_currentFrameTimestamp - m_nominalSourceInterval100ns;
            m_targetPresentationTimestamp = std::max(
                m_targetPresentationTimestamp, interpolationStart);
            duration = m_nominalSourceInterval100ns;
        }
        m_interpolationFactor = std::clamp(
            static_cast<float>(m_targetPresentationTimestamp - interpolationStart) /
            static_cast<float>(duration), 0.0f, 1.0f);

        if (m_cachedPreviousSlot != previous || m_cachedCurrentSlot != current) {
            prefetchPair(interpolator);
            if (m_cachedPreviousSlot != previous || m_cachedCurrentSlot != current) {
                // Pair preparation is still running on the GPU. Consume the
                // swap-chain token without blocking this 144 Hz deadline and
                // keep the media timeline fixed until the pair is ready.
                if (!m_context->nativePresentation() ||
                    !m_context->nativeD3D12->rePresentLastOutput()) {
                    if (FAILED(m_context->presentSwapChain())) return false;
                } else if (FAILED(m_context->presentSwapChain())) {
                    return false;
                }
                revealOutput();
                if (!scheduleNextPresentation()) return false;
                ++m_presentedFrameIndex;
                logPacing("W");
                return true;
            }
        }

        if (m_context->nativePresentation()) {
            ComPtr<ID3D12Resource> backBuffer = m_context->nativeD3D12->backBuffer(m_context->nativeD3D12->currentBackBufferIndex());
            if (!backBuffer) {
                std::cerr << "Presenter: D3D12 back buffer unavailable.\n";
                return false;
            }
            bool ok = interpolator.synthesizeNative(backBuffer.Get(), m_interpolationFactor);
            backBuffer.Reset();
            if (!ok) {
                std::cerr << "Presenter: synthesizeNative failed.\n";
                return false;
            }
        } else {
            ComPtr<ID3D11Texture2D> backBuffer;
            if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
            bool ok = interpolator.synthesize(backBuffer.Get(), m_interpolationFactor);
            backBuffer.Reset();
            if (!ok) return false;
        }
        presented = SUCCEEDED(m_context->presentSwapChain());
    } else {
        m_interpolationFactor = 1.0f;
        presented = presentTexture(interpolator, currentFrame);
    }

    if (!presented) return false;
    revealOutput();
    if (!scheduleNextPresentation()) return false;
    ++m_presentedFrameIndex;
    if (!holdTimeline) {
        m_targetPresentationTimestamp += m_refreshInterval100ns;
        // If advancing crossed into an already captured pair, submit its flow
        // now. This gives preparation the whole interval before the next
        // presentation deadline instead of discovering the pair at that tick.
        prefetchPair(interpolator);
    }
    retireConsumedFrames(sorted, current);
    logPacing(mode ? mode : (previous != current ? "I" : "P"));
    return true;
}

void RealtimePresenter::logPacing(const char* mode) {
    // MOTION_ENHANCER_PACING=1: per-present diagnostics. Bursty intervals with
    // a healthy fps average are invisible to the 2-second status line.
    static const bool enabled = [] {
        char buffer[8] = {};
        return GetEnvironmentVariableA("MOTION_ENHANCER_PACING", buffer, sizeof(buffer)) > 0;
    }();
    if (!enabled) return;
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    const int64_t nowQpc = now.QuadPart;
    int64_t deltaUs = m_lastPresentQpc.QuadPart
        ? static_cast<int64_t>((nowQpc - m_lastPresentQpc.QuadPart) * 1000000 / m_qpcFrequency)
        : 0;
    m_lastPresentQpc.QuadPart = nowQpc;
    std::cout << "[Pacing] " << mode
              << " dt=" << deltaUs << "us"
              << " alpha=" << m_interpolationFactor
              << " t0=" << m_previousFrameTimestamp
              << " t1=" << m_currentFrameTimestamp
              << " target=" << m_targetPresentationTimestamp
              << " ready=" << queueDepth()
              << " slots=";
    for (const auto& slot : m_slots) {
        std::cout << (slot.state == SlotState::Free ? 'F' : 'R');
    }
    std::cout << " cache=" << m_cachedPreviousSlot << "," << m_cachedCurrentSlot
              << " pending=" << m_pendingPreviousSlot << "," << m_pendingCurrentSlot
              << "\n";
}
