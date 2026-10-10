#include "RealtimeSession.h"

#include "CaptureEngine.h"
#include "ComputeKernels.h"
#include "FrameTimeline.h"
#include "Image.h"
#include "MotionVector.h"
#include "NeuralFlow.h"
#include "Presenter.h"
#include "WallClock.h"
#include "WindowHelper.h"

#include <avrt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cctype>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

namespace {
bool envFlag(const char* name) {
    char buffer[8] = {};
    return GetEnvironmentVariableA(name, buffer, sizeof(buffer)) > 0 && buffer[0] != '0';
}

// Console writes block for milliseconds when stdout is a pipe or a slow
// terminal; the render thread hands its lines to this thread instead.
class AsyncLog {
public:
    AsyncLog() : m_thread([this] { run(); }) {}
    ~AsyncLog() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_quit = true;
        }
        m_cv.notify_one();
        m_thread.join();
    }
    void write(std::string line) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lines.push_back(std::move(line));
        }
        m_cv.notify_one();
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(m_mutex);
        for (;;) {
            m_cv.wait(lock, [this] { return m_quit || !m_lines.empty(); });
            if (m_quit && m_lines.empty()) return;
            std::vector<std::string> batch;
            batch.swap(m_lines);
            lock.unlock();
            for (const auto& line : batch) std::cout << line;
            std::cout.flush();
            lock.lock();
        }
    }
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<std::string> m_lines;
    bool m_quit = false;
    std::thread m_thread;
};

// Registers the calling thread with the multimedia class scheduler so a
// busy desktop (the captured browser, other apps) cannot delay a refresh.
HANDLE enterMmcss(const wchar_t* task) {
    DWORD index = 0;
    HANDLE handle = AvSetMmThreadCharacteristicsW(task, &index);
    if (handle) AvSetMmThreadPriority(handle, AVRT_PRIORITY_HIGH);
    else SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    return handle;
}

// Bytes per texel of a flow texture read back: R32G32B32A32_FLOAT (xy =
// vector, zw = per-axis confidence; FlowEngine's kFlowFormat) or, for DIS's
// ring entries, R32G32_FLOAT (the vector only). xy come first either way.
size_t flowTexelBytes(ID3D12Resource* texture) {
    return texture->GetDesc().Format == DXGI_FORMAT_R32G32_FLOAT ? 8 : 16;
}

double computePsnr(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) return 0.0;
    double mse = 0.0;
    const size_t total = static_cast<size_t>(a.width) * a.height * a.channels;
    for (size_t i = 0; i < total; ++i) {
        const double d = static_cast<double>(a.data[i]) - static_cast<double>(b.data[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(total);
    return mse < 1e-10 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

// Small direct-queue helper for the offline paths: dispatch + readback with
// CPU waits (never used on the live path).
class OfflineContext {
public:
    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels) {
        m_device = device;
        m_kernels = kernels;
        ID3D12Device* d3d = device->device();
        if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator))) ||
            FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_allocator.Get(), nullptr, IID_PPV_ARGS(&m_list))) ||
            FAILED(m_list->Close()) ||
            FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
        m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return m_event && m_dispatch.initialize(d3d, 1, 64, 16);
    }
    ~OfflineContext() {
        if (m_event) CloseHandle(m_event);
    }
    bool begin() {
        if (FAILED(m_allocator->Reset()) || FAILED(m_list->Reset(m_allocator.Get(), nullptr))) return false;
        m_dispatch.beginRegion(0);
        return true;
    }
    ID3D12GraphicsCommandList* list() { return m_list.Get(); }
    DispatchContext& dispatch() { return m_dispatch; }
    // Executes on the present queue after the given flow fence value and
    // waits for completion.
    bool execute(ID3D12Fence* waitFence, uint64_t waitValue, uint64_t* computeDoneValue = nullptr) {
        if (FAILED(m_list->Close())) return false;
        ID3D12CommandQueue* queue = m_device->presentQueue();
        if (waitFence && waitValue && FAILED(queue->Wait(waitFence, waitValue))) return false;
        ID3D12CommandList* lists[] = {m_list.Get()};
        queue->ExecuteCommandLists(1, lists);
        const uint64_t value = ++m_fenceValue;
        queue->Signal(m_fence.Get(), value);
        if (computeDoneValue) {
            queue->Signal(m_device->computeDoneFence12(), value);
            *computeDoneValue = value;
        }
        if (m_fence->GetCompletedValue() >= value) return true;
        if (FAILED(m_fence->SetEventOnCompletion(value, m_event))) return false;
        return WaitForSingleObject(m_event, 5000) == WAIT_OBJECT_0;
    }
    // Copies a 2D texture into a readback buffer and returns its rows.
    bool readback(ID3D12Resource* texture, std::vector<uint8_t>& bytes, UINT& rowPitch, UINT& width, UINT& height) {
        const D3D12_RESOURCE_DESC desc = texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT64 total = 0;
        m_device->device()->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
        ComPtr<ID3D12Resource> buffer;
        if (FAILED(m_device->createBuffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                          D3D12_RESOURCE_FLAG_NONE, buffer))) return false;
        if (!begin()) return false;
        DispatchContext::transition(m_list.Get(), texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = buffer.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = texture;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        m_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        DispatchContext::transition(m_list.Get(), texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (!execute(nullptr, 0)) return false;
        uint8_t* data = nullptr;
        if (FAILED(buffer->Map(0, nullptr, reinterpret_cast<void**>(&data)))) return false;
        bytes.assign(data, data + total);
        buffer->Unmap(0, nullptr);
        rowPitch = footprint.Footprint.RowPitch;
        width = static_cast<UINT>(desc.Width);
        height = desc.Height;
        return true;
    }

private:
    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    ComPtr<ID3D12CommandAllocator> m_allocator;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE m_event = nullptr;
    DispatchContext m_dispatch;
};

// MOTION_ENHANCER_DEBUG_VIEW=n renders the live overlay's debug view n
// (Presenter.h) in the offline paths instead of the picture.
float offlineDebugView() {
    char buffer[8] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_DEBUG_VIEW", buffer, sizeof(buffer)) == 0) return 0.0f;
    return static_cast<float>(std::clamp(std::atoi(buffer), 0, kDebugViewCount - 1));
}

bool uploadFrame(GraphicsDevice& device, const Image& img, SharedTexture& texture, uint64_t& readyValue) {
    std::vector<uint8_t> pixels(static_cast<size_t>(img.width) * img.height * 4);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            const size_t o = (static_cast<size_t>(y) * img.width + x) * 4;
            for (int c = 0; c < 3; ++c) {
                const float v = img.channels >= 3 ? img.get(x, y, c) : img.get(x, y, 0);
                pixels[o + c] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            }
            pixels[o + 3] = 255;
        }
    }
    if (!device.createSharedTexture(static_cast<uint32_t>(img.width), static_cast<uint32_t>(img.height),
                                    DXGI_FORMAT_R8G8B8A8_UNORM, texture)) return false;
    device.captureContext()->UpdateSubresource(texture.texture11.Get(), 0, nullptr, pixels.data(),
                                               static_cast<UINT>(img.width * 4), 0);
    readyValue = device.signalCaptureReady();
    return readyValue != 0;
}
}  // namespace

// ======================================================================
// Live overlay
// ======================================================================

int runRealtimeSession(const RealtimeOptions& options) {
    int exitCode = 0;
    HANDLE stopEvent = nullptr;
    if (!options.stopEventName.empty()) {
        stopEvent = OpenEventA(SYNCHRONIZE, FALSE, options.stopEventName.c_str());
        if (!stopEvent) {
            std::cerr << "Error: could not open the GUI capture stop event\n";
            return 1;
        }
    }
    std::thread renderThread([&] {
        WindowHelper::attachInteractiveDesktop();
        std::cout << "\n============================================================\n"
                  << " Initializing Real-Time GPU Frame Interpolation\n"
                  << "============================================================\n";

        HWND target = WindowHelper::findWindowByTitle(options.windowTitle);
        if (!target) {
            std::cerr << "Error: Target window matching \"" << options.windowTitle << "\" not found!\n";
            WindowHelper::printWindowList();
            exitCode = 1;
            return;
        }
        char title[256] = {};
        GetWindowTextA(target, title, sizeof(title));
        std::cout << "Target window: \"" << title << "\"\n" << std::flush;

        std::cout << "[1/5] Creating D3D12 device, queues and D3D11 capture device...\n" << std::flush;
        GraphicsDevice device;
        if (!device.initialize(options.adapter)) { exitCode = 1; return; }
        ComputeKernels kernels;
        if (!kernels.load(device.device())) { exitCode = 1; return; }

        std::cout << "[2/5] Creating click-through overlay window...\n" << std::flush;
        OverlayWindow overlay;
        if (!overlay.create(target, "Motion Enhancer GPU Overlay")) { exitCode = 1; return; }

        std::cout << "[3/5] Preparing Windows Graphics Capture...\n" << std::flush;
        FrameTimeline timeline;
        FlowEngine flow;
        CaptureEngine capture;
        Presenter presenter;
        if (!capture.initialize(&device, &flow, &timeline)) { exitCode = 1; return; }
        if (!capture.prepare(target)) { exitCode = 1; return; }
        const uint32_t width = capture.width();
        const uint32_t height = capture.height();

        std::cout << "[4/5] Creating presentation surface (" << width << "x" << height << ")...\n" << std::flush;
        if (!presenter.initialize(&device, &kernels, overlay.hwnd, width, height)) { exitCode = 1; return; }
        // The present fence gates slot and flow-ring recycling.
        capture.setPresentFence(presenter.fence());

        std::cout << "[5/5] Loading compute pipelines and optical flow engine...\n" << std::flush;
        if (!flow.initialize(&device, &kernels, presenter.fence(), options.flow) || !flow.resize(width, height, CaptureEngine::kSlots)) {
            std::cerr << "Error: flow engine initialization failed.\n";
            exitCode = 1;
            return;
        }
        timeline.configure(options.sourceFps, presenter.refreshRate(), options.outputMultiplier);

        const char* engine = options.flow.neuralFlow ? "neural learned-feature flow"
            : options.flow.disFlow ? "DIS (Dense Inverse Search) optical flow" : options.flow.preferFfxOpticalFlow
            ? (flow.ffxAvailable() ? "AMD FidelityFX Optical Flow (SAD fallback on failure)"
                                   : "native SAD graph (FFX unavailable)")
            : "native SAD graph (selected)";
        std::cout << "\n============================================================\n"
                  << " >>> Real-Time GPU Frame Interpolation Active\n"
                  << "  Source Window:  \"" << title << "\" (" << width << "x" << height << ")\n"
                  << "  Adapter:        " << device.adapterName() << "\n"
                  << "  Flow Engine:    " << engine << ", " << flow.searchedLevels() << " pyramid levels\n"
                  << "  Source Cadence: " << (options.sourceFps ? std::to_string(options.sourceFps) + " fps" : "auto") << "\n"
                  << "  Output:         " << std::fixed << std::setprecision(2) << presenter.refreshRate() << " Hz display"
                  << (options.outputMultiplier ? ", quantized to " + std::to_string(options.outputMultiplier) + "x source" : ", every refresh") << "\n"
                  << "  Hotkeys:        [Ctrl+Alt+F1] Toggle | [Ctrl+Alt+Esc] Exit\n"
                  << "============================================================\n\n" << std::flush;

        if (!capture.start()) { exitCode = 1; return; }
        HANDLE mmcss = enterMmcss(L"Games");

        const bool pacingLog = envFlag("MOTION_ENHANCER_PACING");
        AsyncLog log;
        HANDLE handles[1] = {capture.acceptedEvent()};
        bool presenting = false;
        uint64_t skippedTicks = 0;
        uint64_t forcedTicks = 0;
        uint64_t idleTicks = 0;
        uint64_t consecutiveSkips = 0;
        uint64_t forcedPresents = 0;
        // Last presented content; an unchanged static frame is not presented
        // again (it would only keep the compositor awake for nothing).
        uint64_t lastKeyF0 = 0, lastKeyF1 = 0;
        int lastKeyAlpha = -1;
        uint64_t lastPairIndex = 0;
        uint64_t starvedPairIndex = 0;
        uint64_t starvationTicks = 0;
        uint64_t holdTicks = 0;
        auto statusTime = std::chrono::steady_clock::now();
        double renderCpuMs = 0.0;
        uint64_t renderCount = 0;

        auto renderTick = [&]() -> bool {
            const auto cpu0 = std::chrono::steady_clock::now();
            const int64_t wake = wallClock100ns();
            const int64_t display = presenter.beginFrame(wake);
            const FrameTimeline::Selection sel = timeline.select(display);
            if (!sel.valid) return true;

            // Same frames, same blend as what is on screen: nothing to do.
            const int alphaKey = static_cast<int>(sel.alpha * 4096.0f);
            if (sel.f0.index == lastKeyF0 && sel.f1.index == lastKeyF1 && alphaKey == lastKeyAlpha) {
                ++idleTicks;
                return true;
            }
            // The frame-latency object is a semaphore: taking it commits us
            // to a Present, because DXGI only signals it again when a frame
            // we presented has been consumed. It is therefore taken here,
            // after every reason not to present has been ruled out.
            if (!presenter.canPresent()) {
                ++skippedTicks;
                // A visible window whose frames are never consumed for a
                // whole second means the token was lost, not that the
                // compositor is busy: present regardless rather than
                // freeze on the last frame.
                if (++consecutiveSkips < static_cast<uint64_t>(presenter.refreshRate())) return true;
                ++forcedPresents;
            }
            consecutiveSkips = 0;

            RenderJob job;
            const FlowInput in0 = capture.slotInput(sel.f0.slot);
            const FlowInput in1 = capture.slotInput(sel.f1.slot);
            job.frame0 = in0.texture;
            job.ready0 = in0.readyValue;
            job.frame1 = in1.texture;
            job.ready1 = in1.readyValue;
            job.alpha = sel.alpha;
            const bool pair = sel.f0.index != sel.f1.index;
            FlowEngine::Entry entry;
            const char* mode = "S";
            if (!pair) {
                job.single = true;
                mode = sel.hold ? "H" : (sel.startup ? "B" : "S");
                if (sel.hold) ++holdTicks;
            } else if (sel.f1.sceneCut || sel.f1.flowEntry < 0) {
                job.single = true;   // hard switch across a cut
                mode = "C";
            } else {
                entry = flow.entry(sel.f1.flowEntry);
                if (!entry.valid) {
                    job.single = true;
                    job.alpha = 0.0f;
                    mode = "X";
                } else {
                    job.forwardFlow = entry.forward.Get();
                    job.backwardFlow = entry.backward.Get();
                    job.flowBlockSize = entry.blockSize;
                    job.singleFlowField = entry.singleField;
                    job.uiMask = entry.uiMaskInput();
                    job.flowFence = flow.fence();
                    job.flowFenceValue = sel.f1.flowFence;
                    mode = "I";
                    if (!flow.fenceReached(sel.f1.flowFence)) {
                        // The GPU waits for the flow; the CPU never stalls.
                        // Grow the budget so the next pairs are early.
                        mode = "W";
                        ++starvationTicks;
                        if (starvedPairIndex != sel.f1.index) {
                            starvedPairIndex = sel.f1.index;
                            timeline.reportStarvation();
                        }
                    } else if (lastPairIndex != sel.f1.index) {
                        const int64_t flowTime = static_cast<int64_t>(flow.lastGpuTimeMs() * 10000.0);
                        timeline.reportSlack(wake - sel.f1.arrival - flowTime);
                    }
                }
            }
            lastPairIndex = sel.f1.index;

            uint64_t presentFence = 0;
            if (!presenter.render(job, presentFence)) {
                std::cerr << "Error: presentation failed.\n";
                return false;
            }
            lastKeyF0 = sel.f0.index;
            lastKeyF1 = sel.f1.index;
            lastKeyAlpha = alphaKey;
            capture.noteSlotPresented(sel.f0.slot, presentFence);
            capture.noteSlotPresented(sel.f1.slot, presentFence);
            if (entry.valid) flow.noteEntryPresented(sel.f1.flowEntry, presentFence);
            capture.releaseSlots(timeline.retireBefore(sel.f0.index));

            renderCpuMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu0).count();
            ++renderCount;
            if (pacingLog) {
                std::ostringstream line;
                line << "[Pacing] " << mode << " display=" << display / 10 << "us media=" << sel.mediaTime / 10
                     << "us pair=" << sel.f0.index << "/" << sel.f1.index
                     << " alpha=" << std::fixed << std::setprecision(3) << sel.alpha << "\n";
                log.write(line.str());
            }
            return true;
        };

        // The compositor clock is unavailable while the desktop is idle
        // (nothing being composed, including before our first present). A
        // timeout of 1.5 refreshes then stands in for the tick: our own
        // present wakes the compositor and real ticks resume.
        const DWORD tickTimeoutMs = static_cast<DWORD>(std::max<int64_t>(4, presenter.refreshInterval() * 3 / 20000));
        while (true) {
            if (stopEvent && WaitForSingleObject(stopEvent, 0) == WAIT_OBJECT_0) break;
            // One iteration per compositor tick (or per accepted frame while
            // nothing is on screen yet). Messages are only hotkeys and
            // window tracking, pumped once per tick.
            DWORD wait = presenter.waitForVsync(1, handles, presenting ? tickTimeoutMs : 250);
            if (!overlay.processMessages()) break;
            overlay.updateTracking();

            if (!presenting) {
                TimelineFrame first;
                if (!timeline.lastFrame(first)) continue;
                presenting = true;
            } else if (wait == WAIT_TIMEOUT) {
                ++forcedTicks;
                wait = WAIT_OBJECT_0 + 1;
            } else if (wait != WAIT_OBJECT_0 && wait != WAIT_OBJECT_0 + 1) {
                static bool reported = false;
                if (!reported) {
                    reported = true;
                    std::cerr << "Presenter: unexpected vsync wait result " << wait
                              << " (error " << GetLastError() << ")\n";
                }
            }
            if (wait == WAIT_OBJECT_0 + 1) {
                if (!IsWindowVisible(overlay.hwnd)) {
                    // Hidden with Ctrl+Alt+F1: keep the clocks running, skip
                    // the GPU work.
                    presenter.beginFrame(wallClock100ns());
                } else if (!renderTick()) {
                    exitCode = 1;
                    break;
                }
            }

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - statusTime).count();
            if (elapsed >= 2.0) {
                const PresentStats ps = presenter.takeStats();
                const FrameTimeline::Stats ts = timeline.stats();
                const CaptureEngine::Stats cs = capture.stats();
                std::ostringstream status;
                status << "[Overlay] " << std::fixed << std::setprecision(1)
                          << static_cast<double>(ps.presented) / elapsed << " fps"
                          << " | present " << std::setprecision(2) << ps.meanIntervalMs << " ms avg, "
                          << ps.maxIntervalMs << " ms max, late " << ps.lateFrames << ", resync " << ps.resyncs
                          << " | cpu " << (renderCount ? renderCpuMs / static_cast<double>(renderCount) : 0.0) << " ms"
                          << " | source " << std::setprecision(1) << (ts.sourcePeriodMs > 0 ? 1000.0 / ts.sourcePeriodMs : 0.0)
                          << " fps" << (ts.cadenceLocked ? "" : " (unlocked)")
                          << " | budget " << ts.latencyBudgetMs << " ms, offset " << ts.captureOffsetMs << " ms"
                          << " | starve " << ts.starvations << "/" << starvationTicks << " hold " << holdTicks
                          << " skip " << skippedTicks << " forced " << forcedTicks << " idle " << idleTicks << " fp " << forcedPresents
                          << " noclock " << ps.clockFallbacks
                          << " | wgc " << cs.arrivals << " dup " << cs.duplicates << " acc " << cs.accepted
                          << " def " << ts.deferred << " drop " << ts.dropped + cs.queueDrops << " rep " << ts.replaced
                          << " cut " << cs.sceneCuts
                           << " | flow " << (flow.usingFfx() ? "FFX " : flow.usingNeural() ? "neural " : flow.usingDis() ? "DIS " : "SAD ") << std::setprecision(2)
                          << flow.lastGpuTimeMs() << " ms, diff " << cs.lastDiffMs << " ms"
                          << " | q " << ts.frames << "\n";
                log.write(status.str());
                if (device.debugLayerEnabled()) device.dumpInfoQueue("live");
                statusTime = now;
                renderCpuMs = 0.0;
                renderCount = 0;
                starvationTicks = 0;
                holdTicks = 0;
                skippedTicks = 0;
                forcedTicks = 0;
                idleTicks = 0;
                forcedPresents = 0;
            }
        }

        if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
        capture.stop();
        std::cout << "Real-time GPU frame interpolation terminated.\n";
    });
    renderThread.join();
    if (stopEvent) CloseHandle(stopEvent);
    return exitCode;
}

// ======================================================================
// Offline verification
// ======================================================================

int runOfflineGpuInterpolation(const std::string& frame0Path, const std::string& frame1Path,
                               const std::string& outputPath, const std::string& flowOutputPath,
                               const std::string& groundTruthPath, float t,
                               const FlowSettings& settings, GraphicsAdapterPreference adapter,
                               const std::string& previousPath) {
    Image frame0 = Image::load(frame0Path);
    Image frame1 = Image::load(frame1Path);
    if (frame0.empty() || frame1.empty()) return 1;
    if (frame0.width != frame1.width || frame0.height != frame1.height) {
        std::cerr << "Error: frame dimensions mismatch\n";
        return 1;
    }
    Image previous;
    if (!previousPath.empty()) {
        previous = Image::load(previousPath);
        if (previous.empty() || previous.width != frame0.width || previous.height != frame0.height) {
            std::cerr << "Error: previous frame missing or mismatched\n";
            return 1;
        }
    }
    const uint32_t width = static_cast<uint32_t>(frame0.width);
    const uint32_t height = static_cast<uint32_t>(frame0.height);

    GraphicsDevice device;
    if (!device.initialize(adapter)) return 1;
    ComputeKernels kernels;
    if (!kernels.load(device.device())) return 1;
    OfflineContext offline;
    if (!offline.initialize(&device, &kernels)) return 1;
    FlowEngine flow;
    // The priming pair always searches at the full radius; the caller's
    // radii apply to the timed pair only.
    FlowSettings primeSettings = settings;
    primeSettings.coarseSearchRadius = primeSettings.refineSearchRadius = FlowSettings{}.coarseSearchRadius;
    if (!flow.initialize(&device, &kernels, nullptr, previous.empty() ? settings : primeSettings) ||
        !flow.resize(width, height, previous.empty() ? 2 : 3)) return 1;

    std::cout << "GPU pipeline: " << width << "x" << height << ", " << flow.searchedLevels()
              << " pyramid levels, radii " << settings.coarseSearchRadius << "/" << settings.refineSearchRadius << "\n";

    SharedTexture tex0, tex1, output;
    uint64_t ready0 = 0, ready1 = 0;
    if (!uploadFrame(device, frame0, tex0, ready0) || !uploadFrame(device, frame1, tex1, ready1)) {
        std::cerr << "Error: frame upload failed\n";
        return 1;
    }
    if (!device.createSharedTexture(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, output)) return 1;

    FlowInput in0{tex0.texture12.Get(), ready0, 1, 0};
    FlowInput in1{tex1.texture12.Get(), ready1, 2, 1};
    // Warm-up so the timed run reports steady-state GPU time. With a
    // previous frame the warm-up is the pair (previous, frame0) instead,
    // which primes the cross-pair state for the timed pair.
    SharedTexture texPrev;
    FlowEngine::Result r;
    if (!previous.empty()) {
        uint64_t readyPrev = 0;
        if (!uploadFrame(device, previous, texPrev, readyPrev)) return 1;
        FlowInput inPrev{texPrev.texture12.Get(), readyPrev, 0, 2};
        r = flow.submitPair(inPrev, in0);
        flow.setSettings(settings);
        std::cout << "  Cross-pair state primed with " << previousPath << "\n";
    } else {
        r = flow.submitPair(in0, in1);
    }
    if (r.entry < 0 || !flow.waitFence(r.fence)) {
        std::cerr << "Error: flow submission failed\n";
        return 1;
    }
    // MOTION_ENHANCER_BENCH_ITERS=n: repeat the timed pair and report the
    // fastest GPU time (laptop GPUs change clocks between single runs).
    int benchIterations = 1;
    {
        char buffer[16] = {};
        if (GetEnvironmentVariableA("MOTION_ENHANCER_BENCH_ITERS", buffer, sizeof(buffer)) > 0) {
            benchIterations = std::clamp(std::atoi(buffer), 1, 200);
        }
    }
    const auto start = std::chrono::steady_clock::now();
    double bestGpuMs = 1e30;
    // Queue the iterations back to back (the engine keeps up to four lists in
    // flight) so the GPU stays busy and clocked up; a list that runs after
    // an idle gap measures the clock ramp, not the shaders.
    for (int i = 0; i < benchIterations; ++i) {
        r = flow.submitPair(in0, in1);
        if (r.entry < 0) return 1;
        if (i >= 4) bestGpuMs = std::min(bestGpuMs, flow.lastGpuTimeMs());
    }
    if (!flow.waitFence(r.fence)) return 1;
    bestGpuMs = std::min(bestGpuMs, flow.lastGpuTimeMs());
    const FlowEngine::Entry entry = flow.entry(r.entry);

    ShaderConstants c = {};
    c.width = c.levelWidth = width;
    c.height = c.levelHeight = height;
    setInterpolationConstants(c, entry, t);
    c.neuralControl[1] = offlineDebugView();
    if (!offline.begin()) return 1;
    std::array<ID3D12Resource*, 6> in = {tex0.texture12.Get(), tex1.texture12.Get(),
                                         entry.forward.Get(), entry.backward.Get(), entry.uiMaskInput(), nullptr};
    // With MOTION_ENHANCER_BENCH_ITERS the interpolation pass is timed too:
    // the same dispatch repeated back to back between two timestamps. Each
    // repeat rewrites the region's identical descriptors and constants, so
    // one region serves them all.
    const int interpolateRepeats = benchIterations > 1 ? 32 : 1;
    ComPtr<ID3D12QueryHeap> interpolateQueries;
    ComPtr<ID3D12Resource> interpolateTimes;
    if (interpolateRepeats > 1) {
        D3D12_QUERY_HEAP_DESC query = {};
        query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        query.Count = 2;
        if (FAILED(device.device()->CreateQueryHeap(&query, IID_PPV_ARGS(&interpolateQueries))) ||
            FAILED(device.createBuffer(2 * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_FLAG_NONE, interpolateTimes))) {
            interpolateQueries.Reset();
        }
    }
    if (interpolateQueries) offline.list()->EndQuery(interpolateQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    for (int i = 0; i < interpolateRepeats; ++i) {
        offline.dispatch().beginRegion(0);
        if (!offline.dispatch().dispatch(offline.list(), kernels, Kernel::Interpolate, in, output.texture12.Get(), c,
                                         (width + 15) / 16, (height + 15) / 16, 0x13)) return 1;
    }
    if (interpolateQueries) {
        offline.list()->EndQuery(interpolateQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        offline.list()->ResolveQueryData(interpolateQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                         interpolateTimes.Get(), 0);
    }
    uint64_t computeDone = 0;
    if (!offline.execute(flow.fence(), r.fence, &computeDone)) {
        std::cerr << "Error: interpolate dispatch failed\n";
        return 1;
    }
    double interpolateMs = 0.0;
    if (interpolateQueries) {
        uint64_t frequency = 0;
        uint64_t* stamps = nullptr;
        if (SUCCEEDED(device.presentQueue()->GetTimestampFrequency(&frequency)) && frequency &&
            SUCCEEDED(interpolateTimes->Map(0, nullptr, reinterpret_cast<void**>(&stamps)))) {
            interpolateMs = static_cast<double>(stamps[1] - stamps[0]) * 1000.0 /
                            static_cast<double>(frequency) / interpolateRepeats;
            interpolateTimes->Unmap(0, nullptr);
        }
    }
    std::vector<uint8_t> bytes;
    UINT w = 0, h = 0, pitch = 0;
    if (!offline.readback(output.texture12.Get(), bytes, pitch, w, h)) return 1;
    const double elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    Image result(static_cast<int>(w), static_cast<int>(h), 3);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* row = bytes.data() + static_cast<size_t>(y) * pitch;
        for (UINT x = 0; x < w; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                result.set(static_cast<int>(x), static_cast<int>(y), ch, static_cast<float>(row[x * 4 + ch]));
            }
        }
    }
    if (!result.savePNG(outputPath)) {
        std::cerr << "Failed to save " << outputPath << "\n";
        return 1;
    }
    std::cout << "GPU offline interpolation (t = " << t << "): flow " << std::fixed << std::setprecision(2)
              << bestGpuMs << " ms GPU" << (benchIterations > 1 ? " (best of " + std::to_string(benchIterations) + ")" : "")
              << ", " << elapsedMs << " ms wall incl. readback -> " << outputPath << "\n";
    if (interpolateMs > 0.0) std::cout << "  Interpolate pass: " << interpolateMs << " ms GPU (mean of " << interpolateRepeats << ")\n";
    std::cout
              << "  Flow estimator: " << (flow.usingFfx() ? "AMD FidelityFX Optical Flow" :
                                             flow.usingNeural() ? "neural learned-feature flow" : flow.usingDis() ? "DIS optical flow" : "native SAD shader graph")
              << " (" << entry.blockSize << "x" << entry.blockSize << " blocks)\n";

    if (!flowOutputPath.empty()) {
        std::vector<uint8_t> flowBytes;
        UINT fw = 0, fh = 0, fpitch = 0;
        if (offline.readback(entry.forward.Get(), flowBytes, fpitch, fw, fh)) {
            const uint32_t block = entry.blockSize;
            const uint32_t gridW = (width + block - 1) / block;
            const uint32_t gridH = (height + block - 1) / block;
            FlowField field(frame0.width, frame0.height);
            for (int y = 0; y < frame0.height; ++y) {
                for (int x = 0; x < frame0.width; ++x) {
                    const uint32_t bx = std::min<uint32_t>(static_cast<uint32_t>(x) / block, gridW - 1);
                    const uint32_t by = std::min<uint32_t>(static_cast<uint32_t>(y) / block, gridH - 1);
                    const float* v = reinterpret_cast<const float*>(flowBytes.data() + static_cast<size_t>(by) * fpitch + bx * flowTexelBytes(entry.forward.Get()));
                    field.set(x, y, MotionVector(v[0], v[1], 1.0f));
                }
            }
            const MotionVector centre = field.get(frame0.width / 2, frame0.height / 2);
            std::cout << "  Centre block vector: (" << centre.vx << ", " << centre.vy << ")\n";
            field.toColorImage().savePNG(flowOutputPath);
            std::ofstream txt(flowOutputPath + ".txt");
            for (uint32_t by = 0; by < gridH; ++by) {
                for (uint32_t bx = 0; bx < gridW; ++bx) {
                    const float* v = reinterpret_cast<const float*>(flowBytes.data() + static_cast<size_t>(by) * fpitch + bx * flowTexelBytes(entry.forward.Get()));
                    txt << static_cast<int>(std::lround(v[0])) << "," << static_cast<int>(std::lround(v[1]))
                        << (bx + 1 < gridW ? " " : "\n");
                }
            }
            std::cout << "  Saved forward flow visualization to: " << flowOutputPath << " (+ .txt grid)\n";
        }
    }

    if (!groundTruthPath.empty()) {
        Image truth = Image::load(groundTruthPath);
        if (!truth.empty() && truth.width == result.width && truth.height == result.height) {
            Image blend(frame0.width, frame0.height, 3);
            for (int y = 0; y < blend.height; ++y)
                for (int x = 0; x < blend.width; ++x)
                    for (int ch = 0; ch < 3; ++ch)
                        blend.set(x, y, ch, (1.0f - t) * frame0.get(x, y, ch) + t * frame1.get(x, y, ch));
            std::cout << "  PSNR vs ground truth: " << std::fixed << std::setprecision(2)
                      << computePsnr(result, truth) << " dB (naive blend: " << computePsnr(blend, truth) << " dB)\n";
        }
    }
    device.dumpInfoQueue("offline");
    return 0;
}


// ======================================================================
// Sequence replay: the frame chain as the live path actually builds it
// ======================================================================
//
// Every other offline measurement in this project runs a single frame pair:
// `--gpu-offline-prev` primes only one pair back, and
// MOTION_ENHANCER_BENCH_ITERS re-submits the same pair, which breaks the
// chain after the first iteration. The live overlay runs a continuous chain
// - FFX keeps its own history and the static-UI evidence is carried from
// pair to pair - so anything that accumulates, latches or feeds back across
// frames is invisible to a single pair.
//
// This replays N frames as consecutive pairs through one engine instance,
// interpolates each midpoint and reports PSNR per step. A defect that only
// exists in the chain shows up as PSNR drifting down across the sequence
// while any single pair looks fine.
//
// Frames are <prefix>_f000.png, _f001.png, ...; the true midpoint between
// f(i) and f(i+1) is <prefix>_m000.png and so on, when present.
int runOfflineGpuSequence(const std::string& prefix, const FlowSettings& settings,
                          GraphicsAdapterPreference adapter) {
    std::vector<Image> frames;
    for (int i = 0;; ++i) {
        char name[MAX_PATH];
        std::snprintf(name, sizeof(name), "%s_f%03d.png", prefix.c_str(), i);
        Image image = Image::load(name);
        if (image.empty()) break;
        frames.push_back(std::move(image));
    }
    if (frames.size() < 3) {
        std::cerr << "Sequence needs at least three frames named " << prefix << "_f000.png ...\n";
        return 1;
    }
    const uint32_t width = static_cast<uint32_t>(frames[0].width);
    const uint32_t height = static_cast<uint32_t>(frames[0].height);
    for (const Image& f : frames)
        if (f.width != frames[0].width || f.height != frames[0].height) {
            std::cerr << "Sequence frames differ in size\n";
            return 1;
        }

    GraphicsDevice device;
    if (!device.initialize(adapter)) return 1;
    ComputeKernels kernels;
    if (!kernels.load(device.device())) return 1;
    OfflineContext offline;
    if (!offline.initialize(&device, &kernels)) return 1;
    FlowEngine flow;
    // One slot per frame where possible, so no luma pyramid is recycled
    // mid-sequence: the chain is what is being measured, not the cache.
    const int slots = std::min<int>(FlowEngine::kMaxSlots, static_cast<int>(frames.size()));
    if (!flow.initialize(&device, &kernels, nullptr, settings) ||
        !flow.resize(width, height, slots)) return 1;

    std::vector<SharedTexture> textures(frames.size());
    std::vector<FlowInput> inputs(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        uint64_t ready = 0;
        if (!uploadFrame(device, frames[i], textures[i], ready)) return 1;
        inputs[i] = FlowInput{textures[i].texture12.Get(), ready, i + 1,
                              static_cast<int>(i % static_cast<size_t>(slots))};
    }
    SharedTexture output;
    if (!device.createSharedTexture(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, output)) return 1;

    std::cout << "Sequence replay: " << frames.size() << " frames at " << width << "x" << height
              << ", " << flow.searchedLevels() << " levels, engine "
              << (flow.usingFfx() ? "ffx" : flow.usingNeural() ? "neural" : flow.usingDis() ? "dis" : "sad") << "\n"
              << "  pair   PSNR dB   blend dB   UI px %   centre vector\n";

    double first = 0.0, last = 0.0;
    int measured = 0;
    for (size_t i = 0; i + 1 < frames.size(); ++i) {
        const auto result = flow.submitPair(inputs[i], inputs[i + 1]);
        if (result.entry < 0 || !flow.waitFence(result.fence)) {
            std::cerr << "Sequence: flow submission failed at pair " << i << "\n";
            return 1;
        }
        const FlowEngine::Entry entry = flow.entry(result.entry);
        ShaderConstants c = {};
        c.width = c.levelWidth = width;
        c.height = c.levelHeight = height;
        setInterpolationConstants(c, entry, 0.5f);
        c.neuralControl[1] = offlineDebugView();
        if (!offline.begin()) return 1;
        std::array<ID3D12Resource*, 6> in = {textures[i].texture12.Get(),
                                             textures[i + 1].texture12.Get(),
                                             entry.forward.Get(), entry.backward.Get(),
                                             entry.uiMaskInput(), nullptr};
        if (!offline.dispatch().dispatch(offline.list(), kernels, Kernel::Interpolate, in,
                                         output.texture12.Get(), c,
                                         (width + 15) / 16, (height + 15) / 16, 0x13)) return 1;
        uint64_t done = 0;
        if (!offline.execute(flow.fence(), result.fence, &done)) return 1;

        std::vector<uint8_t> bytes;
        UINT w = 0, h = 0, pitch = 0;
        if (!offline.readback(output.texture12.Get(), bytes, pitch, w, h)) return 1;
        Image produced(static_cast<int>(w), static_cast<int>(h), 3);
        for (UINT y = 0; y < h; ++y) {
            const uint8_t* row = bytes.data() + static_cast<size_t>(y) * pitch;
            for (UINT x = 0; x < w; ++x)
                for (int ch = 0; ch < 3; ++ch)
                    produced.set(static_cast<int>(x), static_cast<int>(y), ch,
                                 static_cast<float>(row[x * 4 + ch]));
        }
        char outName[MAX_PATH];
        std::snprintf(outName, sizeof(outName), "%s_out%03d.png", prefix.c_str(), static_cast<int>(i));
        produced.savePNG(outName);

        char truthName[MAX_PATH];
        std::snprintf(truthName, sizeof(truthName), "%s_m%03d.png", prefix.c_str(), static_cast<int>(i));
        const Image truth = Image::load(truthName);
        double psnr = 0.0, blendPsnr = 0.0;
        if (!truth.empty() && truth.width == produced.width && truth.height == produced.height) {
            Image blend(produced.width, produced.height, 3);
            for (int y = 0; y < blend.height; ++y)
                for (int x = 0; x < blend.width; ++x)
                    for (int ch = 0; ch < 3; ++ch)
                        blend.set(x, y, ch, 0.5f * (frames[i].get(x, y, ch) + frames[i + 1].get(x, y, ch)));
            psnr = computePsnr(produced, truth);
            blendPsnr = computePsnr(blend, truth);
            if (measured == 0) first = psnr;
            last = psnr;
            ++measured;
        }

        std::vector<uint8_t> flowBytes;
        UINT fw = 0, fh = 0, fpitch = 0;
        float vx = 0.0f, vy = 0.0f;
        if (offline.readback(entry.forward.Get(), flowBytes, fpitch, fw, fh)) {
            const uint32_t bx = (width / entry.blockSize) / 2;
            const uint32_t by = (height / entry.blockSize) / 2;
            const float* v = reinterpret_cast<const float*>(
                flowBytes.data() + static_cast<size_t>(by) * fpitch + bx * flowTexelBytes(entry.forward.Get()));
            vx = v[0];
            vy = v[1];
        }
        // Static-UI mask after this pair: <prefix>_ui000.png, white where
        // the pixel is confirmed UI (InterpolateCS shows it unwarped).
        double uiPercent = 0.0;
        if (ID3D12Resource* uiMask = entry.uiMaskInput()) {
            std::vector<uint8_t> maskBytes;
            UINT mw = 0, mh = 0, mpitch = 0;
            if (offline.readback(uiMask, maskBytes, mpitch, mw, mh)) {
                // Stored as count/4 in R8_UNORM; confirmed from a count of 3.
                constexpr uint8_t kConfirmed = 191;
                Image mask(static_cast<int>(mw), static_cast<int>(mh), 1);
                size_t confirmed = 0;
                for (UINT y = 0; y < mh; ++y)
                    for (UINT x = 0; x < mw; ++x) {
                        const bool ui = maskBytes[static_cast<size_t>(y) * mpitch + x] >= kConfirmed;
                        confirmed += ui ? 1 : 0;
                        mask.set(static_cast<int>(x), static_cast<int>(y), 0, ui ? 255.0f : 0.0f);
                    }
                uiPercent = 100.0 * static_cast<double>(confirmed) / (static_cast<double>(mw) * mh);
                char maskName[MAX_PATH];
                std::snprintf(maskName, sizeof(maskName), "%s_ui%03d.png", prefix.c_str(), static_cast<int>(i));
                mask.savePNG(maskName);
            }
        }
        std::cout << "  " << std::setw(4) << i << std::fixed << std::setprecision(2)
                  << std::setw(10) << psnr << std::setw(11) << blendPsnr << std::setw(10) << uiPercent
                  << "   (" << vx << ", " << vy << ")\n";
    }
    if (measured >= 2) {
        std::cout << "  first " << std::fixed << std::setprecision(2) << first
                  << " dB, last " << last << " dB, drift " << (last - first) << " dB"
                  << (last < first - 1.0 ? "   <- the chain is losing ground\n" : "\n");
    }
    device.dumpInfoQueue("sequence");
    return 0;
}

// ======================================================================
// Per-level vector accuracy against known motion
// ======================================================================
//
// PSNR of the interpolated frame says whether the output looks right, not
// which pyramid level got the motion wrong, and a coarse level's error is
// usually hidden by the finer levels' search until a motion is large enough
// to leave their windows. This reads the forward field back at every level -
// both the matcher's raw answer and the filled field the next level inherits
// - and scores each block against the true motion of the pixels it covers.
//
// Input per case: <prefix>_a.png, <prefix>_b.png and <prefix>_flow.bin
// (int32 width, int32 height, then float32 dx, dy per pixel: where each
// pixel of a goes in b). Only blocks whose footprint moves uniformly (within
// a quarter pixel) and stays inside the frame are scored, so a block
// straddling a motion boundary - which no single vector can describe - is
// not counted against the level.
int runFlowLevelEval(const std::vector<std::string>& prefixes, const FlowSettings& settings,
                     GraphicsAdapterPreference adapter) {
    GraphicsDevice device;
    if (!device.initialize(adapter)) return 1;
    ComputeKernels kernels;
    if (!kernels.load(device.device())) return 1;
    OfflineContext offline;
    if (!offline.initialize(&device, &kernels)) return 1;

    for (const std::string& prefix : prefixes) {
        const Image frame0 = Image::load(prefix + "_a.png");
        const Image frame1 = Image::load(prefix + "_b.png");
        std::ifstream truthFile(prefix + "_flow.bin", std::ios::binary);
        int32_t tw = 0, th = 0;
        if (frame0.empty() || frame1.empty() || !truthFile ||
            !truthFile.read(reinterpret_cast<char*>(&tw), sizeof(tw)) ||
            !truthFile.read(reinterpret_cast<char*>(&th), sizeof(th)) ||
            tw != frame0.width || th != frame0.height ||
            frame1.width != tw || frame1.height != th) {
            std::cerr << "Flow eval: cannot read " << prefix << "_a/_b.png and _flow.bin of one size\n";
            return 1;
        }
        std::vector<float> truth(static_cast<size_t>(tw) * th * 2);
        if (!truthFile.read(reinterpret_cast<char*>(truth.data()),
                            static_cast<std::streamsize>(truth.size() * sizeof(float)))) {
            std::cerr << "Flow eval: truncated " << prefix << "_flow.bin\n";
            return 1;
        }

        const uint32_t width = static_cast<uint32_t>(tw);
        const uint32_t height = static_cast<uint32_t>(th);
        FlowEngine flow;
        if (!flow.initialize(&device, &kernels, nullptr, settings) || !flow.resize(width, height, 2)) return 1;
        SharedTexture tex0, tex1;
        uint64_t ready0 = 0, ready1 = 0;
        if (!uploadFrame(device, frame0, tex0, ready0) || !uploadFrame(device, frame1, tex1, ready1)) return 1;
        const auto result = flow.submitPair({tex0.texture12.Get(), ready0, 1, 0},
                                            {tex1.texture12.Get(), ready1, 2, 1});
        if (result.entry < 0 || !flow.waitFence(result.fence)) {
            std::cerr << "Flow eval: flow submission failed for " << prefix << "\n";
            return 1;
        }

        // Frame a's luma pyramid (2x box reduce), only to tell textured
        // blocks from flat ones the way the matcher does: the mean squared
        // central difference along each axis over the block. A flat block
        // has no measurable motion and a wrong vector there is invisible in
        // the interpolated frame; the textured population is the one whose
        // error shows.
        std::vector<std::vector<float>> luma(static_cast<size_t>(flow.searchedLevels()));
        std::vector<int> lumaW(luma.size()), lumaH(luma.size());
        lumaW[0] = tw; lumaH[0] = th;
        luma[0].resize(static_cast<size_t>(tw) * th);
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                luma[0][static_cast<size_t>(y) * tw + x] =
                    0.2126f * frame0.get(x, y, 0) + 0.7152f * frame0.get(x, y, 1) + 0.0722f * frame0.get(x, y, 2);
        for (size_t l = 1; l < luma.size(); ++l) {
            lumaW[l] = std::max(1, lumaW[l - 1] / 2);
            lumaH[l] = std::max(1, lumaH[l - 1] / 2);
            luma[l].resize(static_cast<size_t>(lumaW[l]) * lumaH[l]);
            for (int y = 0; y < lumaH[l]; ++y)
                for (int x = 0; x < lumaW[l]; ++x) {
                    float s = 0.0f;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            s += luma[l - 1][static_cast<size_t>(std::min(2 * y + dy, lumaH[l - 1] - 1)) * lumaW[l - 1] +
                                             std::min(2 * x + dx, lumaW[l - 1] - 1)];
                    luma[l][static_cast<size_t>(y) * lumaW[l] + x] = 0.25f * s;
                }
        }

        std::cout << prefix << " (" << width << "x" << height << ", engine "
                  << (flow.usingFfx() ? "ffx" : flow.usingNeural() ? "neural" : flow.usingDis() ? "dis" : "sad") << ")\n"
                  << "  level  scored textured |   textured: search  mean  >0.5px   >2px |  filled  mean  >0.5px   >2px"
                  << " |  all filled mean   (errors in that level's pixels)\n";
        // MOTION_ENHANCER_FLOW_EVAL_DUMP=1: every scored block's raw search
        // vector, truth and texture to <prefix>_blocks.txt.
        std::ofstream dump;
        {
            char buffer[8] = {};
            if (GetEnvironmentVariableA("MOTION_ENHANCER_FLOW_EVAL_DUMP", buffer, sizeof(buffer)) > 0 &&
                buffer[0] != '0')
                dump.open(prefix + "_blocks.txt");
        }
        for (int level = flow.searchedLevels() - 1; level >= 0; --level) {
            const int scale = 1 << level;
            const int footprint = 8 * scale;   // a level block in full-resolution pixels
            // Each field's own grid: SAD writes one texel per 8x8 block of
            // the level, DIS a patch grid (8x8 patches every kDisStride
            // pixels) and a dense per-pixel field. A field is scored on the
            // level's 8x8 blocks, as the mean of its texels centred inside
            // the block.
            struct Field { std::vector<uint8_t> bytes; UINT pitch = 0, w = 0, h = 0, texel = 16; };
            Field fields[2];
            ID3D12Resource* sources[2] = {flow.searchFlow(level), flow.trainingFlow(level)};
            bool ok = true;
            for (int f = 0; f < 2; ++f) {
                ok = ok && sources[f] &&
                     offline.readback(sources[f], fields[f].bytes, fields[f].pitch, fields[f].w, fields[f].h);
                if (ok) fields[f].texel = sources[f]->GetDesc().Format == DXGI_FORMAT_R32G32_FLOAT ? 8 : 16;
            }
            if (!ok) {
                std::cerr << "Flow eval: level " << level << " readback failed\n";
                return 1;
            }
            const std::vector<float>& lv = luma[static_cast<size_t>(level)];
            const int lw = lumaW[static_cast<size_t>(level)], lh = lumaH[static_cast<size_t>(level)];
            auto L = [&](int x, int y) {
                return lv[static_cast<size_t>(std::clamp(y, 0, lh - 1)) * lw + std::clamp(x, 0, lw - 1)];
            };
            // Texel centre, in this level's pixels, of texel i of a field.
            auto texelCentre = [&](const Field& fd, int i, int extent) {
                if (static_cast<int>(fd.w) == lw && static_cast<int>(fd.h) == lh) return static_cast<double>(i);   // dense
                if (flow.usingDis()) return std::min(i * FlowEngine::kDisStride, std::max(0, extent - 8)) + 3.5;
                return i * 8 + 3.5;
            };
            auto texelStride = [&](const Field& fd) {
                if (static_cast<int>(fd.w) == lw && static_cast<int>(fd.h) == lh) return 1;
                return flow.usingDis() ? FlowEngine::kDisStride : 8;
            };
            auto blockEstimate = [&](const Field& fd, int bx, int by, double& ex, double& ey) {
                ex = ey = 0.0;
                int n = 0;
                const int stride = texelStride(fd);
                const int i0 = std::max(0, (bx * 8 - 8) / stride - 1), i1 = std::min(static_cast<int>(fd.w) - 1, (bx * 8 + 8) / stride + 1);
                const int j0 = std::max(0, (by * 8 - 8) / stride - 1), j1 = std::min(static_cast<int>(fd.h) - 1, (by * 8 + 8) / stride + 1);
                // The last DIS patches are pulled inside the level, so their
                // centres can land in an earlier block: include them.
                const int jLast = static_cast<int>(fd.h) - 1, iLast = static_cast<int>(fd.w) - 1;
                for (int j = j0; j <= std::max(j1, jLast); j = (j == j1 && j1 < jLast) ? jLast : j + 1) {
                    const double cy = texelCentre(fd, j, lh);
                    if (cy < by * 8 || cy >= by * 8 + 8) continue;
                    for (int i = i0; i <= std::max(i1, iLast); i = (i == i1 && i1 < iLast) ? iLast : i + 1) {
                        const double cx = texelCentre(fd, i, lw);
                        if (cx < bx * 8 || cx >= bx * 8 + 8) continue;
                        const float* v = reinterpret_cast<const float*>(
                            fd.bytes.data() + static_cast<size_t>(j) * fd.pitch + static_cast<size_t>(i) * fd.texel);
                        ex += v[0]; ey += v[1]; ++n;
                    }
                }
                if (n == 0) return false;
                ex /= n; ey /= n;
                return true;
            };
            std::vector<double> errors[2], textured[2];
            const int blocksW = (lw + 7) / 8, blocksH = (lh + 7) / 8;
            for (int by = 0; by < blocksH; ++by) {
                for (int bx = 0; bx < blocksW; ++bx) {
                    const int x0 = static_cast<int>(bx) * footprint, y0 = static_cast<int>(by) * footprint;
                    const int x1 = std::min(x0 + footprint, tw), y1 = std::min(y0 + footprint, th);
                    if (x1 - x0 < footprint || y1 - y0 < footprint) continue;   // partial edge block
                    double sx = 0.0, sy = 0.0;
                    float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
                    for (int y = y0; y < y1; ++y)
                        for (int x = x0; x < x1; ++x) {
                            const float* v = &truth[(static_cast<size_t>(y) * tw + x) * 2];
                            sx += v[0]; sy += v[1];
                            minX = std::min(minX, v[0]); maxX = std::max(maxX, v[0]);
                            minY = std::min(minY, v[1]); maxY = std::max(maxY, v[1]);
                        }
                    if (maxX - minX > 0.25f || maxY - minY > 0.25f) continue;   // motion boundary
                    const double n = static_cast<double>(footprint) * footprint;
                    const double tx = sx / n, ty = sy / n;
                    if (x0 + tx < 0.0 || y0 + ty < 0.0 || x1 + tx > tw || y1 + ty > th) continue;   // leaves the frame
                    // Textured = the smaller eigenvalue of the block's
                    // structure tensor: both directions measurable. Per-axis
                    // energy is not enough - a diagonal edge has plenty along
                    // x and y and still says nothing about motion along it.
                    double gxx = 0.0, gyy = 0.0, gxy = 0.0;
                    for (int y = 0; y < 8; ++y)
                        for (int x = 0; x < 8; ++x) {
                            const int px = static_cast<int>(bx) * 8 + x, py = static_cast<int>(by) * 8 + y;
                            const double gx = L(px + 1, py) - L(px - 1, py), gy = L(px, py + 1) - L(px, py - 1);
                            gxx += gx * gx; gyy += gy * gy; gxy += gx * gy;
                        }
                    const double half = 0.5 * (gxx + gyy);
                    const double lambdaMin = half - std::sqrt(0.25 * (gxx - gyy) * (gxx - gyy) + gxy * gxy);
                    const bool isTextured = lambdaMin / 64.0 > 25.0;
                    for (int f = 0; f < 2; ++f) {
                        double e[2];
                        if (!blockEstimate(fields[f], bx, by, e[0], e[1])) continue;
                        const double ex = e[0] - tx / scale, ey = e[1] - ty / scale;
                        const double err = std::sqrt(ex * ex + ey * ey);
                        errors[f].push_back(err);
                        if (isTextured) textured[f].push_back(err);
                        if (f == 0 && dump)
                            dump << level << ' ' << bx << ' ' << by << ' ' << e[0] << ' ' << e[1] << ' '
                                 << tx / scale << ' ' << ty / scale << ' ' << (isTextured ? 1 : 0) << ' '
                                 << lambdaMin / 64.0 << '\n';
                    }
                }
            }
            auto stats = [](const std::vector<double>& e, double& mean, double& over05, double& over2) {
                mean = over05 = over2 = 0.0;
                if (e.empty()) return;
                for (double v : e) {
                    mean += v;
                    over05 += v > 0.5 ? 1.0 : 0.0;
                    over2 += v > 2.0 ? 1.0 : 0.0;
                }
                mean /= static_cast<double>(e.size());
                over05 *= 100.0 / static_cast<double>(e.size());
                over2 *= 100.0 / static_cast<double>(e.size());
            };
            std::cout << "  " << std::setw(5) << level << std::setw(7) << errors[0].size()
                      << std::setw(9) << textured[0].size() << " |";
            for (int f = 0; f < 2; ++f) {
                double mean, over05, over2;
                stats(textured[f], mean, over05, over2);
                std::cout << (f == 0 ? "                " : " |        ") << std::fixed << std::setprecision(3)
                          << std::setw(6) << mean << std::setprecision(1) << std::setw(7) << over05 << "%"
                          << std::setw(6) << over2 << "%";
            }
            double allMean, a05, a2;
            stats(errors[1], allMean, a05, a2);
            std::cout << " |  " << std::setprecision(3) << std::setw(14) << allMean << "\n";
        }
    }
    device.dumpInfoQueue("flow eval");
    return 0;
}

// ======================================================================
// CPU/GPU parity for the convolutional estimator
// ======================================================================
//
// The weights are fitted against the CPU forward pass in NeuralFlow.cpp and
// NeuralFlowTrain.cpp, and executed by LearnedFeaturesCS.hlsl and the neural
// branch of MotionSearchCS.hlsl. Four implementations of one function: if any
// of them drifts, the model is simply wrong on the GPU and nothing else in
// the pipeline reports it. A shader edit, a layout change or a constant-
// buffer reshuffle all land here.
//
// Compares the level-0 feature map (which isolates the extractor) and then
// the finished forward flow field (which covers the cost volume, the head,
// the anchor stage and the soft-argmax).
int runNeuralParityCheck(const std::string& frame0Path, const std::string& frame1Path,
                         GraphicsAdapterPreference adapter) {
    NeuralFlowModel model;
    char modelPath[MAX_PATH] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_NEURAL_MODEL", modelPath, sizeof(modelPath)) == 0 ||
        !model.load(modelPath)) {
        std::cerr << "Parity check needs a trained model in MOTION_ENHANCER_NEURAL_MODEL\n";
        return 1;
    }
    Image frame0 = Image::load(frame0Path);
    Image frame1 = Image::load(frame1Path);
    if (frame0.empty() || frame1.empty() ||
        frame0.width != frame1.width || frame0.height != frame1.height) {
        std::cerr << "Parity check needs two frames of equal size\n";
        return 1;
    }
    const uint32_t width = static_cast<uint32_t>(frame0.width);
    const uint32_t height = static_cast<uint32_t>(frame0.height);
    std::cout << "Neural CPU/GPU parity on " << width << "x" << height << ", model "
              << modelPath << "\n";

    GraphicsDevice device;
    if (!device.initialize(adapter)) return 1;
    ComputeKernels kernels;
    if (!kernels.load(device.device())) return 1;
    OfflineContext offline;
    if (!offline.initialize(&device, &kernels)) return 1;

    FlowSettings settings;
    settings.neuralFlow = true;
    settings.allowTrainedModel = false;   // injected below, not read from disk
    FlowEngine flow;
    if (!flow.initialize(&device, &kernels, nullptr, settings)) return 1;
    flow.setNeuralModel(model);
    if (!flow.resize(width, height)) return 1;

    SharedTexture tex0, tex1;
    uint64_t ready0 = 0, ready1 = 0;
    if (!uploadFrame(device, frame0, tex0, ready0) ||
        !uploadFrame(device, frame1, tex1, ready1)) return 1;
    const FlowInput in0{tex0.texture12.Get(), ready0, 1, 0};
    const FlowInput in1{tex1.texture12.Get(), ready1, 2, 1};
    const auto result = flow.submitPair(in0, in1);
    if (result.entry < 0 || !flow.waitFence(result.fence)) {
        std::cerr << "Parity check: flow submission failed\n";
        return 1;
    }

    // R16G16_FLOAT readback. tanh output, so every value is in [-1, 1] and
    // half precision carries ~3 decimal digits there; the tolerances below
    // are set by that, not by the arithmetic.
    auto halfToFloat = [](uint16_t bits) {
        const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
        const int exponent = (bits >> 10) & 31;
        const int mantissa = bits & 1023;
        if (exponent == 0) return sign * std::ldexp(static_cast<float>(mantissa), -24);
        if (exponent == 31) return 1000.0f;      // never valid for a tanh
        return sign * std::ldexp(1.0f + mantissa / 1024.0f, exponent - 15);
    };

    const int levels = flow.searchedLevels();
    const auto luma = neuralLumaPyramid(frame0, levels);
    const auto features = neuralFeaturePyramid(luma, model);

    // The GPU's own feature maps, read back as float images. Running the CPU
    // matcher on *these* separates the two things that can differ: whether
    // the matcher computes the same function, and whether the R16G16_FLOAT
    // feature texture's three decimal digits are enough for the anchor
    // argmax to land on the same candidate. Without the split, one number
    // conflates a shader bug with half-precision rounding.
    auto readGpuFeatures = [&](int slot, std::vector<Image>& out) {
        out.assign(static_cast<size_t>(levels), Image());
        for (int level = 0; level < levels; ++level) {
            ID3D12Resource* texture = flow.trainingFeature(slot, level);
            std::vector<uint8_t> bytes;
            UINT pitch = 0, w = 0, h = 0;
            if (!texture || !offline.readback(texture, bytes, pitch, w, h)) return false;
            Image image(static_cast<int>(w), static_cast<int>(h), 2);
            for (UINT y = 0; y < h; ++y)
                for (UINT x = 0; x < w; ++x)
                    for (int channel = 0; channel < 2; ++channel) {
                        uint16_t bits = 0;
                        std::memcpy(&bits, bytes.data() + static_cast<size_t>(y) * pitch +
                                               x * 4 + channel * 2, sizeof(bits));
                        image.set(static_cast<int>(x), static_cast<int>(y), channel,
                                  halfToFloat(bits));
                    }
            out[static_cast<size_t>(level)] = std::move(image);
        }
        return true;
    };

    bool ok = true;
    {
        ID3D12Resource* gpuFeature = flow.trainingFeature(0, 0);
        std::vector<uint8_t> bytes;
        UINT pitch = 0, fw = 0, fh = 0;
        if (!gpuFeature || !offline.readback(gpuFeature, bytes, pitch, fw, fh)) {
            std::cerr << "Parity check: feature readback failed\n";
            return 1;
        }
        double sum = 0.0, worst = 0.0;
        size_t measured = 0;
        // Skip a 2 px frame: the extractor's 5x5 receptive field clamps at
        // the border, and the CPU and the shader clamp through different
        // code paths there. Everything inside is an honest comparison.
        for (UINT y = 2; y + 2 < fh; ++y)
            for (UINT x = 2; x + 2 < fw; ++x)
                for (int channel = 0; channel < 2; ++channel) {
                    uint16_t bits = 0;
                    std::memcpy(&bits, bytes.data() + static_cast<size_t>(y) * pitch +
                                           x * 4 + channel * 2, sizeof(bits));
                    const double error = std::abs(halfToFloat(bits) -
                        features[0].get(static_cast<int>(x), static_cast<int>(y), channel));
                    sum += error;
                    worst = std::max(worst, error);
                    ++measured;
                }
        const double mean = measured ? sum / measured : 1.0;
        const bool pass = mean < 2e-3 && worst < 2e-2;
        ok = ok && pass;
        std::cout << "  level-0 features: mean |CPU - GPU| " << std::scientific
                  << std::setprecision(2) << mean << ", worst " << worst << " over "
                  << std::defaultfloat << measured << " samples"
                  << (pass ? "  ok\n" : "  MISMATCH\n");
    }
    {
        std::vector<uint8_t> bytes;
        UINT pitch = 0, gw = 0, gh = 0;
        if (!offline.readback(flow.entry(result.entry).forward.Get(), bytes, pitch, gw, gh)) {
            std::cerr << "Parity check: flow readback failed\n";
            return 1;
        }
        std::vector<Image> gpuFeature0, gpuFeature1;
        if (!readGpuFeatures(0, gpuFeature0) || !readGpuFeatures(1, gpuFeature1)) {
            std::cerr << "Parity check: feature pyramid readback failed\n";
            return 1;
        }
        const auto features1 = neuralFeaturePyramid(neuralLumaPyramid(frame1, levels), model);

        // The matcher's own parity is the blocking test; the end-to-end
        // number is reported because half-precision features legitimately
        // tip a near-tied anchor and that is not a defect to fix here.
        // Each level against the GPU's own features *and* the GPU's own
        // predictor, so what is measured is that level's arithmetic rather
        // than inherited divergence. This is the blocking test: it is the
        // strongest statement a CPU reference can make about a recursion
        // whose anchor stage is an argmax.
        auto readLevelFlow = [&](int level, Image& out) {
            ID3D12Resource* texture = flow.trainingFlow(level);
            std::vector<uint8_t> levelBytes;
            UINT levelPitch = 0, lw = 0, lh = 0;
            if (!texture || !offline.readback(texture, levelBytes, levelPitch, lw, lh)) return false;
            out = Image(static_cast<int>(lw), static_cast<int>(lh), 2);
            for (UINT by = 0; by < lh; ++by)
                for (UINT bx = 0; bx < lw; ++bx) {
                    const float* v = reinterpret_cast<const float*>(
                        levelBytes.data() + static_cast<size_t>(by) * levelPitch + bx * flowTexelBytes(texture));
                    out.set(static_cast<int>(bx), static_cast<int>(by), 0, v[0]);
                    out.set(static_cast<int>(bx), static_cast<int>(by), 1, v[1]);
                }
            return true;
        };
        std::vector<Image> gpuLevelFlow(static_cast<size_t>(levels));
        bool haveLevels = true;
        for (int level = 0; level < levels; ++level)
            haveLevels = haveLevels && readLevelFlow(level, gpuLevelFlow[static_cast<size_t>(level)]);
        if (!haveLevels) {
            std::cerr << "Parity check: per-level flow readback failed\n";
            return 1;
        }
        std::cout << "  per level, each given the GPU's own predictor:\n";
        for (int level = levels - 1; level >= 0; --level) {
            const Image cpu = neuralFlowLevel(
                luma[static_cast<size_t>(level)],
                gpuFeature0[static_cast<size_t>(level)], gpuFeature1[static_cast<size_t>(level)],
                model, level, levels,
                level + 1 < levels ? &gpuLevelFlow[static_cast<size_t>(level + 1)] : nullptr);
            const Image& gpu = gpuLevelFlow[static_cast<size_t>(level)];
            // Two populations, and conflating them hides real bugs. A block
            // where both sides searched the same window differs only by
            // float arithmetic, and that must be tiny - this is what catches
            // a wrong clamp order or a mismatched radius, both of which
            // showed up here as a small error on *every* block. A block
            // whose near-tied anchor tipped differs by a whole window, and
            // some of those are unavoidable: /fp:fast on the host and
            // reassociation on the GPU mean the two can never be bit-equal,
            // so the test bounds how often it happens rather than requiring
            // never.
            double agreedSum = 0.0, worst = 0.0;
            size_t measured = 0, agreed = 0, tipped = 0;
            for (int by = 0; by < gpu.height && by < cpu.height; ++by)
                for (int bx = 0; bx < gpu.width && bx < cpu.width; ++bx) {
                    const double error = std::hypot(gpu.get(bx, by, 0) - cpu.get(bx, by, 0),
                                                    gpu.get(bx, by, 1) - cpu.get(bx, by, 1));
                    worst = std::max(worst, error);
                    if (error > 0.5) {
                        ++tipped;
                    } else {
                        agreedSum += error;
                        ++agreed;
                    }
                    ++measured;
                }
            const double agreedMean = agreed ? agreedSum / agreed : 0.0;
            const double tippedFraction =
                measured ? static_cast<double>(tipped) / measured : 0.0;
            // The same-window bound is the one with teeth: the clamp-order
            // bug this test found produced 0.09 px there on *every* block
            // while correct arithmetic measures 1e-5 to 1e-2. The limit sits
            // between them. It was 0.01 before FillFlowCS joined the
            // recursion; the fill sums 48 weighted neighbours per block and
            // the two sides accumulate that in different orders, which moved
            // the finest levels' floor to ~0.011 on its own. The tipped-anchor bound is loose on purpose - a block
            // whose anchors are genuinely near-tied has two windows that
            // explain the data about equally well, and which one wins is
            // decided in the last bits, where the host's /fp:fast and the
            // GPU's reassociation disagree. On the GPU that decision is at
            // least *stable*, because the tie falls back to the coarse
            // predictor, but the host and the GPU can still round the two
            // keys apart, so the choice cannot be bit-reproduced.
            const bool pass = agreedMean < 0.03 && tippedFraction < 0.10;
            ok = ok && pass;
            std::cout << "    level " << level << " (" << gpu.width << "x" << gpu.height
                      << "): same-window mean " << std::fixed << std::setprecision(5)
                      << agreedMean << " px over " << agreed << " blocks, tipped anchors "
                      << std::setprecision(2) << 100.0 * tippedFraction << "%, worst "
                      << std::setprecision(3) << worst << " px"
                      << (pass ? "  ok" : "  MISMATCH") << "\n";
        }

        struct Comparison { const char* name; Image cpuFlow; double tolerance; double allowed; bool blocking; };
        Comparison comparisons[] = {
            {"matcher (GPU features)", neuralFlow(luma, gpuFeature0, gpuFeature1, model), 1.0, 0.10, false},
            {"end to end", neuralFlow(luma, features, features1, model), 1.0, 0.10, false},
        };
        for (const Comparison& comparison : comparisons) {
            const Image& cpuFlow = comparison.cpuFlow;
            double sum = 0.0, worst = 0.0;
            size_t measured = 0, disagreed = 0;
            for (UINT by = 1; by + 1 < gh && by + 1 < (UINT)cpuFlow.height; ++by)
                for (UINT bx = 1; bx + 1 < gw && bx + 1 < (UINT)cpuFlow.width; ++bx) {
                    const float* v = reinterpret_cast<const float*>(
                        bytes.data() + static_cast<size_t>(by) * pitch + bx * flowTexelBytes(flow.entry(result.entry).forward.Get()));
                    const float cx = cpuFlow.get(static_cast<int>(bx), static_cast<int>(by), 0);
                    const float cy = cpuFlow.get(static_cast<int>(bx), static_cast<int>(by), 1);
                    const double error = std::hypot(v[0] - cx, v[1] - cy);
                    sum += error;
                    worst = std::max(worst, error);
                    // A block whose anchor stage chose a different candidate
                    // lands a whole window away, so count those separately: a
                    // handful is a near-tie tipping, a large fraction means
                    // the two sides disagree about the cost itself.
                    if (error > 1.0) ++disagreed;
                    ++measured;
                }
            const double mean = measured ? sum / measured : 1.0;
            const double fraction = measured ? static_cast<double>(disagreed) / measured : 1.0;
            const bool pass = mean < comparison.tolerance && fraction < comparison.allowed;
            if (comparison.blocking) ok = ok && pass;
            std::cout << "  " << std::left << std::setw(24) << comparison.name << std::right
                      << "mean |CPU - GPU| " << std::fixed << std::setprecision(4) << mean
                      << " px, worst " << std::setw(8) << worst
                      << " px, anchor disagreements " << std::setprecision(2)
                      << 100.0 * fraction << "% of " << measured << " blocks"
                      << (pass ? "  ok" : (comparison.blocking ? "  MISMATCH" : "  (reported)"))
                      << "\n";
        }
    }
    device.dumpInfoQueue("parity");
    std::cout << (ok ? "Neural parity check passed.\n"
                     : "Neural parity check FAILED: the shader and the trained model "
                       "do not compute the same function.\n");
    return ok ? 0 : 1;
}
int runD3D12SelfTest(GraphicsAdapterPreference adapter) {
    GraphicsDevice device;
    if (!device.initialize(adapter)) return 1;
    ComputeKernels kernels;
    if (!kernels.load(device.device())) {
        std::cerr << "D3D12 self-test: compute pipelines failed to load.\n";
        return 1;
    }
    OfflineContext offline;
    if (!offline.initialize(&device, &kernels)) return 1;
    FlowSettings settings;
    FlowEngine flow;
    if (!flow.initialize(&device, &kernels, nullptr, settings) || !flow.resize(320, 240)) {
        std::cerr << "D3D12 self-test: flow engine failed.\n";
        return 1;
    }
    Image a(320, 240, 3), b(320, 240, 3);
    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 320; ++x) {
            const float v = 128.0f + 100.0f * std::sin(x * 0.2f) * std::cos(y * 0.15f);
            const float w = 128.0f + 100.0f * std::sin((x - 4) * 0.2f) * std::cos((y - 2) * 0.15f);
            for (int ch = 0; ch < 3; ++ch) { a.set(x, y, ch, v); b.set(x, y, ch, w); }
        }
    }
    SharedTexture tex0, tex1, output;
    uint64_t ready0 = 0, ready1 = 0;
    if (!uploadFrame(device, a, tex0, ready0) || !uploadFrame(device, b, tex1, ready1) ||
        !device.createSharedTexture(320, 240, DXGI_FORMAT_R8G8B8A8_UNORM, output)) return 1;
    std::cout << "D3D12 self-test: shared textures + flow submission...\n";
    const FlowEngine::Result r = flow.submitPair({tex0.texture12.Get(), ready0, 1, 0}, {tex1.texture12.Get(), ready1, 2, 1});
    if (r.entry < 0 || !flow.waitFence(r.fence)) {
        std::cerr << "D3D12 self-test: FAIL (flow)\n";
        return 1;
    }
    const FlowEngine::Entry entry = flow.entry(r.entry);
    ShaderConstants c = {};
    c.width = c.levelWidth = 320;
    c.height = c.levelHeight = 240;
    setInterpolationConstants(c, entry, 0.5f);
    if (!offline.begin()) return 1;
    std::array<ID3D12Resource*, 6> in = {tex0.texture12.Get(), tex1.texture12.Get(),
                                         entry.forward.Get(), entry.backward.Get(), nullptr, nullptr};
    if (!offline.dispatch().dispatch(offline.list(), kernels, Kernel::Interpolate, in, output.texture12.Get(), c, 20, 15, 0x3) ||
        !offline.execute(flow.fence(), r.fence)) {
        std::cerr << "D3D12 self-test: FAIL (interpolate)\n";
        return 1;
    }
    device.dumpInfoQueue("self-test");
    std::cout << "D3D12 self-test: PASS (" << (flow.usingFfx() ? "FFX" :
                                                flow.usingNeural() ? "neural" : flow.usingDis() ? "DIS" : "SAD") << ", "
              << std::fixed << std::setprecision(2) << flow.lastGpuTimeMs() << " ms flow)\n";
    return 0;
}
