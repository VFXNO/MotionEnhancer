#include "Presenter.h"

#include "CaptureEngine.h"
#include "Image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <dwmapi.h>
#include <iostream>

namespace {
constexpr DWORD kGpuWaitTimeoutMs = 5000;

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
    if (!createProjection(*device, width, height, m_projection)) return false;
    // The replay buffer's presented frames (requestCaptureDump()); without
    // the memory for it the hotkey saves nothing, and presenting goes on.
    m_replayBusy = std::make_shared<std::atomic<bool>>(false);
    for (ReplayFrame& frame : m_replayShown) {
        if (FAILED(device->createTexture2D(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_COMMON, frame.texture, L"ReplayShownFrame"))) {
            for (ReplayFrame& f : m_replayShown) f.texture.Reset();
            break;
        }
    }

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
    // Four dispatches per presented frame (splat clear, splat, resolve,
    // interpolate), seven descriptors and one constant slot each.
    if (!m_dispatch.initialize(d3d, kBackBuffers, 32, 8)) return false;

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

bool createProjection(GraphicsDevice& device, uint32_t width, uint32_t height, ComPtr<ID3D12Resource>& projection) {
    const uint32_t gw = (width + 1) / 2;
    const uint32_t gh = (height + 1) / 2;
    return SUCCEEDED(device.createTexture2D(2 * gw, gh, DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_COMMON, projection, L"MotionProjection"));
}

bool recordDepthView(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const ComputeKernels& kernels,
                     const ShaderConstants& constants, ID3D12Resource* depth0, ID3D12Resource* depth1,
                     ID3D12Resource* output) {
    if (!depth0 || !depth1) return true;
    std::array<ID3D12Resource*, 6> in = {depth0, depth1, nullptr, nullptr, nullptr, nullptr};
    return dispatch.dispatch(list, kernels, Kernel::DepthView, in, output, constants,
                             (constants.width + 15) / 16, (constants.height + 15) / 16);
}

bool recordInterpolate(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const ComputeKernels& kernels,
                       const ShaderConstants& c, ID3D12Resource* frame0, ID3D12Resource* frame1,
                       ID3D12Resource* forward, ID3D12Resource* backward, ID3D12Resource* uiMask,
                       ID3D12Resource* projection, ID3D12Resource* output, uint32_t implicitInputs,
                       ID3D12Resource* depth0, ID3D12Resource* depth1) {
    const bool project = c.flowScale == 0.0f && projection;
    if (project) {
        const uint32_t block = static_cast<uint32_t>(std::max(1, c.blockSize));
        const uint32_t gw = (c.width + block - 1) / block;
        const uint32_t gh = (c.height + block - 1) / block;
        std::array<ID3D12Resource*, 6> none = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const bool depth = depth0 && depth1;
        std::array<ID3D12Resource*, 6> splat = {frame0, frame1, forward, backward, depth ? depth0 : nullptr,
                                                depth ? depth1 : nullptr};
        ShaderConstants s = c;
        s.searchRadius = depth ? 1 : 0;   // ProjectSplatCS: AI depth bound
        if (!dispatch.dispatch(list, kernels, Kernel::ProjectClear, none, projection, c, (2 * gw + 15) / 16, (gh + 15) / 16) ||
            !dispatch.dispatch(list, kernels, Kernel::ProjectSplat, splat, projection, s, (2 * gw + 15) / 16, (gh + 15) / 16,
                               implicitInputs & 0x3)) return false;
    }
    std::array<ID3D12Resource*, 6> in = {frame0, frame1, forward, backward, uiMask, project ? projection : nullptr};
    if (!dispatch.dispatch(list, kernels, Kernel::Interpolate, in, output, c,
                           (c.width + 15) / 16, (c.height + 15) / 16, implicitInputs)) return false;
    if (static_cast<int>(c.neuralControl[1]) == 5) return recordDepthView(list, dispatch, kernels, c, depth0, depth1, output);
    return true;
}

namespace {
std::atomic<bool> g_replayRequested{false};

std::string executableDirectory() {
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string directory(path);
    const size_t slash = directory.find_last_of("\\/");
    return slash == std::string::npos ? std::string(".") : directory.substr(0, slash);
}

struct ReplayItem {
    ComPtr<ID3D12Resource> texture;
    std::string name;
};

// The save thread: reads the frozen replay buffer back on a copy queue of its
// own, one frame at a time, and writes the PNGs one after another at the
// lowest priority. Before, the hotkey read every game frame back on the
// capture thread (a GPU wait per frame), allocated a readback buffer per
// presented frame on the render thread and encoded ~24 PNGs at once: the game
// and the presenter stuttered exactly during the moment being captured.
void saveReplay(ComPtr<ID3D12Device> device, ComPtr<ID3D12Fence> presentFence, uint64_t presentValue,
                std::vector<ReplayItem> items, std::shared_ptr<std::atomic<bool>> busy) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    SYSTEMTIME now = {};
    GetLocalTime(&now);
    char stamp[64] = {};
    std::snprintf(stamp, sizeof(stamp), "%04u%02u%02u-%02u%02u%02u", now.wYear, now.wMonth, now.wDay,
                  now.wHour, now.wMinute, now.wSecond);
    const std::string root = executableDirectory() + "\\captures";
    const std::string folder = root + "\\" + stamp;
    CreateDirectoryA(root.c_str(), nullptr);
    CreateDirectoryA(folder.c_str(), nullptr);

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    int saved = 0;
    if (event && SUCCEEDED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) &&
        SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator))) &&
        SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&list))) &&
        SUCCEEDED(list->Close()) &&
        SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) &&
        SUCCEEDED(queue->Wait(presentFence.Get(), presentValue))) {
        ComPtr<ID3D12Resource> readback;
        UINT64 readbackSize = 0;
        uint64_t value = 0;
        std::vector<uint8_t> rgb;
        for (const ReplayItem& item : items) {
            const D3D12_RESOURCE_DESC desc = item.texture->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
            UINT64 total = 0;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
            if (total > readbackSize) {
                D3D12_HEAP_PROPERTIES heap = {};
                heap.Type = D3D12_HEAP_TYPE_READBACK;
                D3D12_RESOURCE_DESC buffer = {};
                buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                buffer.Width = total;
                buffer.Height = 1;
                buffer.DepthOrArraySize = 1;
                buffer.MipLevels = 1;
                buffer.SampleDesc.Count = 1;
                buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(readback.ReleaseAndGetAddressOf())))) break;
                readbackSize = total;
            }
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.Get(), nullptr))) break;
            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = readback.Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = footprint;
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = item.texture.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            if (FAILED(list->Close())) break;
            ID3D12CommandList* lists[] = {list.Get()};
            queue->ExecuteCommandLists(1, lists);
            if (FAILED(queue->Signal(fence.Get(), ++value)) ||
                FAILED(fence->SetEventOnCompletion(value, event)) ||
                WaitForSingleObject(event, kGpuWaitTimeoutMs) != WAIT_OBJECT_0) break;

            // The game frames are B8G8R8A8 (red is the third byte), the
            // presented ones R8G8B8A8.
            const bool bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                              desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ||
                              desc.Format == DXGI_FORMAT_B8G8R8A8_TYPELESS;
            const UINT width = footprint.Footprint.Width;
            const UINT height = footprint.Footprint.Height;
            rgb.resize(static_cast<size_t>(width) * height * 3);
            uint8_t* data = nullptr;
            if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&data)))) break;
            for (UINT y = 0; y < height; ++y) {
                const uint8_t* row = data + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch;
                uint8_t* out = rgb.data() + static_cast<size_t>(y) * width * 3;
                for (UINT x = 0; x < width; ++x) {
                    out[x * 3 + 0] = row[x * 4 + (bgra ? 2 : 0)];
                    out[x * 3 + 1] = row[x * 4 + 1];
                    out[x * 3 + 2] = row[x * 4 + (bgra ? 0 : 2)];
                }
            }
            const D3D12_RANGE nothing = {0, 0};
            readback->Unmap(0, &nothing);
            const std::string path = folder + "\\" + item.name;
            if (stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 3, rgb.data(),
                               static_cast<int>(width * 3))) ++saved;
        }
    }
    if (event) CloseHandle(event);
    std::cout << "[Capture] saved " << saved << " of " << items.size() << " frames to " << folder << std::endl;
    busy->store(false);
}
}  // namespace

void requestCaptureDump() {
    g_replayRequested.store(true);
}

// Copies this refresh's presented frame (m_output, in COPY_SOURCE) and any
// game frame not seen before into the replay buffer. Skipped while a save
// is reading the buffer back.
void Presenter::recordReplay(const RenderJob& job) {
    if (m_replayBusy->load()) return;
    ID3D12Device* d3d = m_device->device();
    const std::pair<ID3D12Resource*, uint64_t> games[2] = {{job.frame0, job.ready0}, {job.frame1, job.ready1}};
    for (const auto& game : games) {
        if (!game.first || !game.second || game.second <= m_replayLastGame) continue;
        // CopyResource needs identical sizes and formats. The capture slots
        // keep one size for the whole session, but a mismatch here would be
        // a device removal, not a bad picture: start the buffer over instead.
        if (m_replayGame[0].texture) {
            const D3D12_RESOURCE_DESC have = m_replayGame[0].texture->GetDesc();
            const D3D12_RESOURCE_DESC want = game.first->GetDesc();
            if (have.Width != want.Width || have.Height != want.Height || have.Format != want.Format) {
                for (ReplayFrame& f : m_replayGame) f.texture.Reset();
                m_replayGameCount = 0;
            }
        }
        if (!m_replayGame[0].texture) {
            // Allocated on the first game frame, with that frame's layout.
            D3D12_RESOURCE_DESC desc = game.first->GetDesc();
            desc.Alignment = 0;
            desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            D3D12_HEAP_PROPERTIES heap = {};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            for (ReplayFrame& frame : m_replayGame) {
                if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, IID_PPV_ARGS(&frame.texture)))) {
                    for (ReplayFrame& f : m_replayGame) f.texture.Reset();
                    m_replayLastGame = UINT64_MAX;   // no game frames in captures
                    break;
                }
                frame.texture->SetName(L"ReplayGameFrame");
            }
            if (!m_replayGame[0].texture) continue;
        }
        ReplayFrame& slot = m_replayGame[m_replayGameCount % kReplayGame];
        DispatchContext::transition(m_list.Get(), slot.texture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        m_list->CopyResource(slot.texture.Get(), game.first);
        DispatchContext::transition(m_list.Get(), slot.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        slot.ready0 = game.second;
        m_replayLastGame = game.second;
        ++m_replayGameCount;
    }

    ReplayFrame& slot = m_replayShown[m_replayShownCount % kReplayShown];
    if (!slot.texture) return;
    DispatchContext::transition(m_list.Get(), slot.texture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    m_list->CopyResource(slot.texture.Get(), m_output.Get());
    DispatchContext::transition(m_list.Get(), slot.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    // Real frames are shown as-is or at t = 0/1; the rest are generated.
    slot.real = job.single || !job.frame1 || job.alpha <= 0.001f || job.alpha >= 0.999f;
    slot.alpha = job.alpha;
    if (job.single || !job.frame1) {
        slot.ready0 = (job.alpha < 0.5f || !job.frame1) ? job.ready0 : job.ready1;
        slot.ready1 = 0;
    } else {
        slot.ready0 = job.ready0;
        slot.ready1 = job.ready1;
    }
    ++m_replayShownCount;
}

// Freezes the replay buffer and hands it, oldest frame first, to a save
// thread. presentValue completes the last copy into it.
void Presenter::startReplaySave(uint64_t presentValue) {
    if (m_replayBusy->load()) {
        std::cout << "[Capture] still saving the previous capture" << std::endl;
        return;
    }
    std::vector<ReplayItem> items;
    // Game frames, numbered in order; the presented frames name the pair
    // they were made from (shown_005_g2-3_t0.50.png: between game_002 and
    // game_003), so the moment can be replayed offline.
    std::vector<uint64_t> gameReady;
    const uint64_t games = std::min<uint64_t>(m_replayGameCount, kReplayGame);
    for (uint64_t i = m_replayGameCount - games; i < m_replayGameCount; ++i) {
        const ReplayFrame& frame = m_replayGame[i % kReplayGame];
        if (!frame.texture) continue;
        char name[64] = {};
        std::snprintf(name, sizeof(name), "game_%03d.png", static_cast<int>(gameReady.size()));
        gameReady.push_back(frame.ready0);
        items.push_back({frame.texture, name});
    }
    auto gameIndex = [&](uint64_t ready) {
        for (size_t i = 0; i < gameReady.size(); ++i)
            if (gameReady[i] == ready) return static_cast<int>(i);
        return -1;
    };
    const uint64_t shown = std::min<uint64_t>(m_replayShownCount, kReplayShown);
    int index = 0;
    for (uint64_t i = m_replayShownCount - shown; i < m_replayShownCount; ++i) {
        const ReplayFrame& frame = m_replayShown[i % kReplayShown];
        if (!frame.texture) continue;
        char name[96] = {};
        const int g0 = gameIndex(frame.ready0);
        if (frame.real || !frame.ready1)
            std::snprintf(name, sizeof(name), "shown_%03d_g%d_real.png", index, g0);
        else
            std::snprintf(name, sizeof(name), "shown_%03d_g%d-%d_t%.2f.png", index, g0, gameIndex(frame.ready1),
                          frame.alpha);
        ++index;
        items.push_back({frame.texture, name});
    }
    if (items.empty()) return;
    m_replayBusy->store(true);
    std::cout << "[Capture] saving the last " << index << " presented frames and " << gameReady.size()
              << " game frames..." << std::endl;
    std::thread(saveReplay, ComPtr<ID3D12Device>(m_device->device()), m_fence, presentValue, std::move(items),
                m_replayBusy).detach();
}

static std::atomic<int> g_debugView{0};

void cycleDebugView() {
    static const char* const names[kDebugViewCount] = {
        "off", "motion (hue = direction, strength = speed)",
        "static UI (yellow = evidence building, green = confirmed)",
        "occlusion (red = frame0 sample distrusted, blue = frame1)",
        "vector arrows (one per 16 px, quarter length, capped at 14 px, coloured by direction)",
        "AI depth (blue far .. red near; grey where there is none: AI depth off or not ready)"};
    const int view = (g_debugView.load() + 1) % kDebugViewCount;
    g_debugView.store(view);
    std::cout << "[Overlay] debug view " << view << ": " << names[view] << std::endl;
}

int debugView() {
    return g_debugView.load();
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
    c.neuralControl[1] = static_cast<float>(debugView());
    bool ok;
    if (job.single || !job.frame1 || !job.forwardFlow || !job.backwardFlow) {
        ID3D12Resource* source = (job.alpha < 0.5f || !job.frame1) ? job.frame0 : job.frame1;
        // Capture textures are shared (simultaneous-access): no barriers.
        std::array<ID3D12Resource*, 6> in = {source, nullptr, nullptr, nullptr, nullptr, nullptr};
        ok = m_dispatch.dispatch(m_list.Get(), *m_kernels, Kernel::PresentFrame, in, m_output.Get(), c,
                                 (m_width + 15) / 16, (m_height + 15) / 16, 0x1);
        // Debug view 5: the shown frame's own AI depth.
        if (ok && debugView() == 5) {
            ID3D12Resource* depth = (job.alpha < 0.5f || !job.frame1) ? job.depth0 : job.depth1;
            ok = recordDepthView(m_list.Get(), m_dispatch, *m_kernels, c, depth, depth, m_output.Get());
        }
    } else {
        c.blockSize = c.parentBlockSize = job.flowBlockSize;
        c.timeT = std::clamp(job.alpha, 0.0f, 1.0f);
        c.flowScale = job.singleFlowField ? 1.0f : 0.0f;
        // The UI mask is promoted implicitly: the flow queue may be reading
        // it at the same moment as the next pair's prior evidence.
        ok = recordInterpolate(m_list.Get(), m_dispatch, *m_kernels, c, job.frame0, job.frame1, job.forwardFlow,
                               job.backwardFlow, job.uiMask, m_projection.Get(), m_output.Get(), 0x13,
                               job.depth0, job.depth1);
    }
    if (!ok) {
        m_list->Close();
        return false;
    }

    ID3D12Resource* back = m_backBuffers[m_swapChain->GetCurrentBackBufferIndex()].Get();
    DispatchContext::transition(m_list.Get(), m_output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    recordReplay(job);
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
    if (g_replayRequested.exchange(false)) startReplaySave(presentFenceValue);
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
