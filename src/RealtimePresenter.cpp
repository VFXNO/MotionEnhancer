#include "RealtimePresenter.h"

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
    uint32_t sourceFps
) {
    m_context = std::move(context);
    m_outputWindow = outputWindow;
    m_width = width;
    m_height = height;
    setSourceFps(sourceFps > 0 ? sourceFps : 30u);
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
    m_refreshInterval100ns = static_cast<int64_t>(10000000.0 / m_refreshRate + 0.5);
    LARGE_INTEGER frequency = {};
    LARGE_INTEGER now = {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    m_qpcFrequency = frequency.QuadPart;
    m_nextPresentationQpc = now.QuadPart;
    m_pacingTimer = CreateWaitableTimerExW(
        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!m_pacingTimer || m_qpcFrequency <= 0) return false;

    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    for (auto& slot : m_slots) {
        if (FAILED(m_context->device->CreateTexture2D(
                &textureDesc, nullptr, slot.texture.GetAddressOf())) ||
            FAILED(m_context->device->CreateQuery(
                &queryDesc, slot.completion.GetAddressOf()))) {
            return false;
        }
    }

    // Prime the flip-model chain once. Every subsequent Present is preceded by
    // one successful frame-latency wait and no back-buffer reference is retained.
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
    ComPtr<ID3D11RenderTargetView> rtv;
    if (FAILED(m_context->device->CreateRenderTargetView(
            backBuffer.Get(), nullptr, rtv.GetAddressOf()))) return false;
    const float clear[4] = {};
    m_context->context->ClearRenderTargetView(rtv.Get(), clear);
    backBuffer.Reset();
    rtv.Reset();
    if (FAILED(m_context->swapChain->Present(0, 0))) return false;
    return scheduleNextPresentation();
}

void RealtimePresenter::setSourceFps(uint32_t sourceFps) {
    if (sourceFps > 0) {
        m_nominalSourceInterval100ns = 10000000LL / static_cast<int64_t>(sourceFps);
    }
}

bool RealtimePresenter::scheduleNextPresentation() {
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    int64_t qpcStep = static_cast<int64_t>(
        static_cast<double>(m_qpcFrequency) / m_refreshRate + 0.5);
    if (m_nextPresentationQpc <= now.QuadPart) {
        int64_t behind = now.QuadPart - m_nextPresentationQpc;
        m_nextPresentationQpc += (behind / qpcStep + 1) * qpcStep;
    } else {
        m_nextPresentationQpc += qpcStep;
    }

    int64_t remainingQpc = std::max<int64_t>(1, m_nextPresentationQpc - now.QuadPart);
    LARGE_INTEGER due = {};
    due.QuadPart = -std::max<int64_t>(
        1, remainingQpc * 10000000LL / m_qpcFrequency);
    return SetWaitableTimer(m_pacingTimer, &due, 0, nullptr, nullptr, FALSE) != FALSE;
}

HANDLE RealtimePresenter::frameLatencyHandle() const {
    return m_context ? m_context->frameLatencyHandle() : nullptr;
}

void RealtimePresenter::updateCompletions() {
    for (auto& slot : m_slots) {
        if (slot.state == SlotState::CopyPending &&
            m_context->context->GetData(
                slot.completion.Get(), nullptr, 0,
                D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK) {
            slot.state = SlotState::Ready;
            m_queuedFrameIndex = std::max(m_queuedFrameIndex, slot.index);
        }
    }
}

ID3D11Texture2D* RealtimePresenter::acquireCaptureTarget() {
    updateCompletions();
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
    slot.state = SlotState::CopyPending;
    m_context->context->End(slot.completion.Get());
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
        if (slotIndex == m_cachedPreviousSlot || slotIndex == m_cachedCurrentSlot) {
            continue;
        }
        m_slots[static_cast<size_t>(slotIndex)].state = SlotState::Free;
    }
}

bool RealtimePresenter::presentTexture(
    GPUInterpolator& interpolator,
    FrameSlot& slot
) {
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
    if (!interpolator.presentSourceFrame(slot.texture.Get(), backBuffer.Get())) return false;
    backBuffer.Reset();
    if (FAILED(m_context->swapChain->Present(0, 0))) return false;
    revealOutput();
    return true;
}

bool RealtimePresenter::presentNext(GPUInterpolator& interpolator) {
    updateCompletions();
    auto sorted = sortedReadySlots();
    if (sorted.empty()) {
        // A consumed latency signal must always be paired with Present or the
        // waitable swapchain will not produce another presentation slot.
        if (FAILED(m_context->swapChain->Present(0, 0))) return false;
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

    if (current < 0) {
        current = sorted.back();
        previous = current;
        m_targetPresentationTimestamp =
            m_slots[static_cast<size_t>(current)].timestamp100ns;
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
            if (!interpolator.prepareFramePair(
                    previousFrame.texture.Get(), currentFrame.texture.Get(),
                    previousFrame.index, currentFrame.index)) {
                return false;
            }
            m_cachedPreviousSlot = previous;
            m_cachedCurrentSlot = current;
        }

        ComPtr<ID3D11Texture2D> backBuffer;
        if (FAILED(m_context->swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) return false;
        if (!interpolator.synthesize(backBuffer.Get(), m_interpolationFactor)) return false;
        backBuffer.Reset();
        presented = SUCCEEDED(m_context->swapChain->Present(0, 0));
    } else {
        m_interpolationFactor = 1.0f;
        presented = presentTexture(interpolator, currentFrame);
    }

    if (!presented) return false;
    revealOutput();
    if (!scheduleNextPresentation()) return false;
    ++m_presentedFrameIndex;
    m_targetPresentationTimestamp += m_refreshInterval100ns;
    retireConsumedFrames(sorted, current);
    return true;
}
