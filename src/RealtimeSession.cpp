#include "RealtimeSession.h"

#include "CaptureEngine.h"
#include "ComputeKernels.h"
#include "FrameTimeline.h"
#include "Image.h"
#include "MotionVector.h"
#include "Presenter.h"
#include "WallClock.h"
#include "WindowHelper.h"

#include <avrt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace {
bool envFlag(const char* name) {
    char buffer[8] = {};
    return GetEnvironmentVariableA(name, buffer, sizeof(buffer)) > 0 && buffer[0] != '0';
}

// Per-pixel vector selection in InterpolateCS (constant FlowScale > 0).
// On by default; MOTION_ENHANCER_PIXEL_SELECT=0 restores the plain
// bilinear blend for A/B comparison.
bool pixelSelectEnabled() {
    char buffer[8] = {};
    return GetEnvironmentVariableA("MOTION_ENHANCER_PIXEL_SELECT", buffer, sizeof(buffer)) == 0 ||
           buffer[0] != '0';
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

        const char* engine = options.flow.preferFfxOpticalFlow
            ? (flow.ffxAvailable() ? "AMD FidelityFX Optical Flow (MSAD fallback on failure)"
                                   : "native MSAD graph (FFX unavailable)")
            : "native MSAD graph (selected)";
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
                          << " | flow " << (flow.usingFfx() ? "FFX " : "MSAD ") << std::setprecision(2)
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
    // radii apply to the timed pair only, so a radius of 0 isolates what
    // the temporal predictor alone contributes.
    FlowSettings primeSettings = settings;
    primeSettings.coarseSearchRadius = primeSettings.refineSearchRadius = 8;
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
    // which primes the temporal predictor chain for the timed pair.
    SharedTexture texPrev;
    FlowEngine::Result r;
    if (!previous.empty()) {
        uint64_t readyPrev = 0;
        if (!uploadFrame(device, previous, texPrev, readyPrev)) return 1;
        FlowInput inPrev{texPrev.texture12.Get(), readyPrev, 0, 2};
        r = flow.submitPair(inPrev, in0);
        flow.setSettings(settings);
        std::cout << "  Temporal chain primed with " << previousPath << "\n";
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
    c.blockSize = c.parentBlockSize = entry.blockSize;
    c.timeT = std::clamp(t, 0.0f, 1.0f);
    c.flowScale = pixelSelectEnabled() ? 1.0f : 0.0f;
    if (!offline.begin()) return 1;
    std::array<ID3D12Resource*, 4> in = {tex0.texture12.Get(), tex1.texture12.Get(),
                                         entry.forward.Get(), entry.backward.Get()};
    if (!offline.dispatch().dispatch(offline.list(), kernels, Kernel::Interpolate, in, output.texture12.Get(), c,
                                     (width + 15) / 16, (height + 15) / 16, 0x3)) return 1;
    uint64_t computeDone = 0;
    if (!offline.execute(flow.fence(), r.fence, &computeDone)) {
        std::cerr << "Error: interpolate dispatch failed\n";
        return 1;
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
              << ", " << elapsedMs << " ms wall incl. readback -> " << outputPath << "\n"
              << "  Flow estimator: " << (flow.usingFfx() ? "AMD FidelityFX Optical Flow" : "native MSAD shader graph")
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
                    const float* v = reinterpret_cast<const float*>(flowBytes.data() + static_cast<size_t>(by) * fpitch + bx * 8);
                    field.set(x, y, MotionVector(v[0], v[1], 1.0f));
                }
            }
            const MotionVector centre = field.get(frame0.width / 2, frame0.height / 2);
            std::cout << "  Centre block vector: (" << centre.vx << ", " << centre.vy << ")\n";
            field.toColorImage().savePNG(flowOutputPath);
            std::ofstream txt(flowOutputPath + ".txt");
            for (uint32_t by = 0; by < gridH; ++by) {
                for (uint32_t bx = 0; bx < gridW; ++bx) {
                    const float* v = reinterpret_cast<const float*>(flowBytes.data() + static_cast<size_t>(by) * fpitch + bx * 8);
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
    c.blockSize = c.parentBlockSize = entry.blockSize;
    c.timeT = 0.5f;
    c.flowScale = pixelSelectEnabled() ? 1.0f : 0.0f;
    if (!offline.begin()) return 1;
    std::array<ID3D12Resource*, 4> in = {tex0.texture12.Get(), tex1.texture12.Get(), entry.forward.Get(), entry.backward.Get()};
    if (!offline.dispatch().dispatch(offline.list(), kernels, Kernel::Interpolate, in, output.texture12.Get(), c, 20, 15, 0x3) ||
        !offline.execute(flow.fence(), r.fence)) {
        std::cerr << "D3D12 self-test: FAIL (interpolate)\n";
        return 1;
    }
    device.dumpInfoQueue("self-test");
    std::cout << "D3D12 self-test: PASS (" << (flow.usingFfx() ? "FFX" : "MSAD") << ", "
              << std::fixed << std::setprecision(2) << flow.lastGpuTimeMs() << " ms flow)\n";
    return 0;
}
