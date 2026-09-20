#include "Presenter.h"

#include <algorithm>
#include <cmath>
#include <dwmapi.h>
#include <iostream>

namespace {
constexpr DWORD kGpuWaitTimeoutMs = 5000;

// Per-pixel vector selection in InterpolateCS (constant FlowScale > 0).
// On by default; MOTION_ENHANCER_PIXEL_SELECT=0 restores the plain
// bilinear blend for A/B comparison.
bool pixelSelectEnabled() {
    char buffer[8] = {};
    return GetEnvironmentVariableA("MOTION_ENHANCER_PIXEL_SELECT", buffer, sizeof(buffer)) == 0 ||
           buffer[0] != '0';
}

double queryRefreshRate(HWND window) {
    // Prefer the exact rational rate DWM composes at; fall back to the
    // monitor's integer mode rate.
    DWM_TIMING_INFO timing = {};
    timing.cbSize = sizeof(timing);
    if (SUCCEEDED(DwmGetCompositionTimingInfo(nullptr, &timing)) &&
        timing.rateRefresh.uiNumerator > 0 && timing.rateRefresh.uiDenominator > 0) {
        return static_cast<double>(timing.rateRefresh.uiNumerator) /
               static_cast<double>(timing.rateRefresh.uiDenominator);
    }
    HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXA info = {};
    info.cbSize = sizeof(info);
    DEVMODEA mode = {};
    mode.dmSize = sizeof(mode);
    if (GetMonitorInfoA(monitor, &info) &&
        EnumDisplaySettingsA(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) &&
        mode.dmDisplayFrequency > 1) {
        return static_cast<double>(mode.dmDisplayFrequency);
    }
    return 60.0;
}
}

Presenter::~Presenter() {
    if (m_fence && m_fenceEvent && m_fence->GetCompletedValue() < m_fenceValue) {
        if (SUCCEEDED(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent)))
            WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs);
    }
    if (m_latencyWaitable) CloseHandle(m_latencyWaitable);
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    if (m_dcomp) FreeLibrary(m_dcomp);
}

bool Presenter::initialize(GraphicsDevice* device, const ComputeKernels* kernels, HWND window,
                           uint32_t width, uint32_t height) {
    m_device = device;
    m_kernels = kernels;
    m_window = window;
    m_width = width;
    m_height = height;
    ID3D12Device* d3d = device->device();

    ComPtr<IDXGIFactory2> factory;
    if (FAILED(device->adapter()->GetParent(IID_PPV_ARGS(&factory)))) return false;

    DXGI_SWAP_CHAIN_DESC1 desc = {};
    desc.Width = width;
    desc.Height = height;
    // RGBA8 typed UAV support is guaranteed; the compute output is copied
    // into the back buffer.
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = kBackBuffers;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    ComPtr<IDXGISwapChain1> swapChain1;
    HRESULT hr = factory->CreateSwapChainForHwnd(device->presentQueue(), window, &desc, nullptr, nullptr,
                                                 swapChain1.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Error: swap chain creation failed (0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }
    if (FAILED(swapChain1.As(&m_swapChain))) return false;
    factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
    // Latency 1: the wait returns as soon as the compositor has taken the
    // previous frame, leaving a full refresh to produce the next one.
    m_swapChain->SetMaximumFrameLatency(1);
    m_latencyWaitable = m_swapChain->GetFrameLatencyWaitableObject();
    if (!m_latencyWaitable) return false;
    for (UINT i = 0; i < kBackBuffers; ++i) {
        if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i])))) return false;
    }

    if (FAILED(device->createTexture2D(width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, m_output, L"PresentOutput"))) return false;

    for (auto& set : m_commands) {
        if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&set.allocator)))) return false;
    }
    if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commands[0].allocator.Get(), nullptr,
                                      IID_PPV_ARGS(&m_list))) ||
        FAILED(m_list->Close()) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
    m_fence->SetName(L"PresentFence");
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;
    if (!m_dispatch.initialize(d3d, kBackBuffers, 16, 4)) return false;

    m_refreshHz = queryRefreshRate(window);
    m_refreshInterval = static_cast<int64_t>(10000000.0 / m_refreshHz + 0.5);
    m_intervalEstimate = static_cast<double>(m_refreshInterval);

    // The compositor clock is the one true vsync for a composed window. The
    // frame-latency waitable object of a layered overlay was observed to
    // signal between refreshes, so it is only used as back-pressure.
    m_dcomp = LoadLibraryW(L"dcomp.dll");
    if (m_dcomp) {
        m_compositorWait = reinterpret_cast<CompositorClockWait>(
            GetProcAddress(m_dcomp, "DCompositionWaitForCompositorClock"));
    }
    if (!m_compositorWait) {
        std::cerr << "Presenter: compositor clock unavailable; pacing on the swap chain only.\n";
    }
    return true;
}

DWORD Presenter::waitForVsync(DWORD count, const HANDLE* handles, DWORD timeoutMs) {
    if (m_compositorWait) {
        const DWORD r = m_compositorWait(count, handles, timeoutMs);
        if (r < WAIT_OBJECT_0 + count + 1 || r == WAIT_TIMEOUT) return r;
        // The compositor clock is unavailable while nothing is being
        // composed (STATUS_GRAPHICS_PRESENT_OCCLUDED on an idle desktop,
        // including before our own first present). Pace ourselves until
        // it comes back: the caller treats WAIT_TIMEOUT as a tick.
        ++m_clockFallbacks;
    }
    const DWORD r = WaitForMultipleObjects(count, handles, FALSE, timeoutMs);
    return r;
}

bool Presenter::waitFence(uint64_t value) {
    if (m_fence->GetCompletedValue() >= value) return true;
    if (FAILED(m_fence->SetEventOnCompletion(value, m_fenceEvent))) return false;
    if (WaitForSingleObject(m_fenceEvent, kGpuWaitTimeoutMs) != WAIT_OBJECT_0) {
        std::cerr << "Presenter: GPU fence timeout (device removed reason 0x" << std::hex
                  << m_device->device()->GetDeviceRemovedReason() << std::dec << ")\n";
        return false;
    }
    return true;
}

int64_t Presenter::beginFrame(int64_t wake) {
    const double nominal = static_cast<double>(m_refreshInterval);
    if (m_lastWake != 0) {
        const double delta = static_cast<double>(wake - m_lastWake);
        m_intervalSum += delta;
        ++m_intervalCount;
        m_stats.maxIntervalMs = std::max(m_stats.maxIntervalMs, delta / 10000.0);
        if (delta > 1.5 * nominal) ++m_stats.lateFrames;
    }
    m_lastWake = wake;

    // The frame rendered after this wake-up is scanned out at the next
    // vsync; the exact lead only shifts latency, but the *spacing* between
    // predictions must be one refresh. Lock a clock to the wake cadence and
    // resync only on a genuine missed refresh.
    const int64_t measured = wake + m_refreshInterval;
    if (m_lastPredicted == 0) {
        m_lastPredicted = measured;
        return measured;
    }
    int64_t predicted = m_lastPredicted + static_cast<int64_t>(m_intervalEstimate + 0.5);
    const int64_t err = measured - predicted;
    if (std::llabs(err) > static_cast<int64_t>(m_intervalEstimate * 0.5)) {
        predicted = measured;
        ++m_stats.resyncs;
    } else {
        predicted += err / 8;
        // Frequency correction: follows the true refresh (e.g. 143.98 Hz
        // vs. the reported 144) without ever drifting a whole frame.
        m_intervalEstimate += static_cast<double>(err) / 64.0;
        m_intervalEstimate = std::clamp(m_intervalEstimate, nominal * 0.97, nominal * 1.03);
    }
    m_lastPredicted = predicted;
    return predicted;
}

void Presenter::revealWindow() {
    if (m_revealed || !m_window) return;
    SetLayeredWindowAttributes(m_window, 0, 255, LWA_ALPHA);
    ShowWindow(m_window, SW_SHOWNOACTIVATE);
    m_revealed = true;
}

bool Presenter::render(const RenderJob& job, uint64_t& presentFenceValue) {
    presentFenceValue = 0;
    if (!job.frame0) return false;
    const UINT idx = static_cast<UINT>(m_frameCount % kBackBuffers);
    CommandSet& set = m_commands[idx];
    // Three lists deep with latency 1: this wait is normally already
    // satisfied. When it is not, the GPU is more than two refreshes behind
    // and there is nothing useful to do but let it catch up.
    if (set.fence && !waitFence(set.fence)) return false;
    if (FAILED(set.allocator->Reset()) || FAILED(m_list->Reset(set.allocator.Get(), nullptr))) return false;
    m_dispatch.beginRegion(idx);

    ShaderConstants c = {};
    c.width = c.levelWidth = m_width;
    c.height = c.levelHeight = m_height;
    bool ok;
    if (job.single || !job.frame1 || !job.forwardFlow || !job.backwardFlow) {
        ID3D12Resource* source = (job.alpha < 0.5f || !job.frame1) ? job.frame0 : job.frame1;
        // Capture textures are shared (simultaneous-access): no barriers.
        std::array<ID3D12Resource*, 4> in = {source, nullptr, nullptr, nullptr};
        ok = m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::PresentFrame, in, m_output.Get(), c,
                                 (m_width + 15) / 16, (m_height + 15) / 16, 0x1);
    } else {
        c.blockSize = c.parentBlockSize = job.flowBlockSize;
        c.timeT = std::clamp(job.alpha, 0.0f, 1.0f);
        c.flowScale = pixelSelectEnabled() ? 1.0f : 0.0f;
        std::array<ID3D12Resource*, 4> in = {job.frame0, job.frame1, job.forwardFlow, job.backwardFlow};
        ok = m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::Interpolate, in, m_output.Get(), c,
                                 (m_width + 15) / 16, (m_height + 15) / 16, 0x3);
    }
    if (!ok) {
        m_list->Close();
        return false;
    }

    ID3D12Resource* back = m_backBuffers[m_swapChain->GetCurrentBackBufferIndex()].Get();
    DispatchContext::transition(m_list.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    DispatchContext::transition(m_list.Get(), back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    m_list->CopyResource(back, m_output.Get());
    DispatchContext::transition(m_list.Get(), back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    DispatchContext::transition(m_list.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (FAILED(m_list->Close())) return false;

    ID3D12CommandQueue* queue = m_device->presentQueue();
    ID3D12Fence* ready = m_device->captureReadyFence12();
    // All ordering is on the GPU: capture copies, then the flow for this
    // pair, then this list. The CPU never blocks on either producer.
    if (job.ready0 && FAILED(queue->Wait(ready, job.ready0))) return false;
    if (job.frame1 && job.ready1 && FAILED(queue->Wait(ready, job.ready1))) return false;
    if (!job.single && job.flowFence && job.flowFenceValue &&
        FAILED(queue->Wait(job.flowFence, job.flowFenceValue))) return false;
    ID3D12CommandList* lists[] = {m_list.Get()};
    queue->ExecuteCommandLists(1, lists);
    presentFenceValue = ++m_fenceValue;
    if (FAILED(queue->Signal(m_fence.Get(), presentFenceValue))) return false;
    set.fence = presentFenceValue;
    ++m_frameCount;

    const HRESULT hr = m_swapChain->Present(1, 0);
    if (FAILED(hr)) {
        std::cerr << "Presenter: Present failed (0x" << std::hex << hr << ", device removed reason 0x"
                  << m_device->device()->GetDeviceRemovedReason() << std::dec << ")\n";
        return false;
    }
    if (hr != m_lastPresentStatus) {
        m_lastPresentStatus = hr;
        if (hr != S_OK) std::cerr << "Presenter: Present status 0x" << std::hex << hr << std::dec << "\n";
    }
    ++m_stats.presented;
    revealWindow();
    return true;
}

PresentStats Presenter::takeStats() {
    PresentStats s = m_stats;
    s.meanIntervalMs = m_intervalCount ? m_intervalSum / static_cast<double>(m_intervalCount) / 10000.0 : 0.0;
    s.clockFallbacks = m_clockFallbacks;
    m_clockFallbacks = 0;
    m_stats = {};
    m_intervalSum = 0.0;
    m_intervalCount = 0;
    return s;
}
