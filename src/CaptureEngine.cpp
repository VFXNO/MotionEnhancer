#include "CaptureEngine.h"
#include "WallClock.h"

#include <avrt.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace {
constexpr size_t kCallbackQueueLimit = 4;
constexpr uint32_t kDiffStride = 4;      // sample every 4th pixel in x and y
// Below this fraction of changed samples the frame is an identical repaint
// (a caret or clock digit is far below it).
constexpr float kDuplicateFraction = 0.0005f;
// A scene cut: most samples changed and by a lot. Pans on detailed content
// change most samples but by much less on average.
constexpr float kCutFraction = 0.5f;
constexpr float kCutMeanDiff = 48.0f;

// Luma difference between two frames on a sparse grid, reduced to
// {changed count, |diff| sum, sample count}. Compiled at runtime for the
// D3D11 capture device (the DXC artifacts are DXIL, which D3D11 rejects).
const char* kDiffShader = R"(
Texture2D<float4> A : register(t0);
Texture2D<float4> B : register(t1);
RWByteAddressBuffer Out : register(u0);
cbuffer P : register(b0) { uint W; uint H; uint Stride; uint Pad; };
groupshared uint gChanged;
groupshared uint gSum;
groupshared uint gCount;
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint li : SV_GroupIndex)
{
    if (li == 0) { gChanged = 0; gSum = 0; gCount = 0; }
    GroupMemoryBarrierWithGroupSync();
    uint2 p = id.xy * Stride;
    if (p.x < W && p.y < H) {
        float3 a = A.Load(int3(p, 0)).rgb;
        float3 b = B.Load(int3(p, 0)).rgb;
        float la = dot(a, float3(0.2126, 0.7152, 0.0722));
        float lb = dot(b, float3(0.2126, 0.7152, 0.0722));
        uint d = (uint)(abs(la - lb) * 255.0 + 0.5);
        InterlockedAdd(gCount, 1);
        InterlockedAdd(gSum, d);
        if (d > 8) InterlockedAdd(gChanged, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (li == 0) {
        Out.InterlockedAdd(0, gChanged);
        Out.InterlockedAdd(4, gSum);
        Out.InterlockedAdd(8, gCount);
    }
}
)";
}

CaptureEngine::CaptureEngine() {
    m_frameEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    m_acceptedEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
    } catch (...) {
        // Already initialized on this thread.
    }
}

CaptureEngine::~CaptureEngine() {
    stop();
    if (m_frameEvent) CloseHandle(m_frameEvent);
    if (m_acceptedEvent) CloseHandle(m_acceptedEvent);
}

bool CaptureEngine::initialize(GraphicsDevice* device, FlowEngine* flow, FrameTimeline* timeline) {
    m_device = device;
    m_flow = flow;
    m_timeline = timeline;
    if (!m_frameEvent || !m_acceptedEvent) return false;

    ComPtr<IDXGIDevice> dxgiDevice;
    if (FAILED(ComPtr<ID3D11Device>(device->captureDevice()).As(&dxgiDevice))) return false;
    const HRESULT hr = CreateDirect3D11DeviceFromDXGIDevice(
        dxgiDevice.Get(), reinterpret_cast<IInspectable**>(winrt::put_abi(m_winrtDevice)));
    if (FAILED(hr)) {
        std::cerr << "Error: CreateDirect3D11DeviceFromDXGIDevice failed (0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }
    return createDiffResources();
}

bool CaptureEngine::createDiffResources() {
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(kDiffShader, strlen(kDiffShader), "FrameDiffCS", nullptr, nullptr,
                                  "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                                  code.GetAddressOf(), errors.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Error: frame diff shader failed to compile"
                  << (errors ? ": " + std::string(static_cast<const char*>(errors->GetBufferPointer())) : "")
                  << "\n";
        return false;
    }
    ID3D11Device* d3d = m_device->captureDevice();
    if (FAILED(d3d->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                                        m_diffShader.GetAddressOf()))) return false;

    D3D11_BUFFER_DESC cb = {};
    cb.ByteWidth = 16;
    cb.Usage = D3D11_USAGE_DYNAMIC;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(d3d->CreateBuffer(&cb, nullptr, m_diffConstants.GetAddressOf()))) return false;

    D3D11_BUFFER_DESC out = {};
    out.ByteWidth = 16;
    out.Usage = D3D11_USAGE_DEFAULT;
    out.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    out.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    if (FAILED(d3d->CreateBuffer(&out, nullptr, m_diffOutput.GetAddressOf()))) return false;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = 4;
    uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    if (FAILED(d3d->CreateUnorderedAccessView(m_diffOutput.Get(), &uav, m_diffOutputUav.GetAddressOf()))) return false;

    D3D11_BUFFER_DESC staging = {};
    staging.ByteWidth = 16;
    staging.Usage = D3D11_USAGE_STAGING;
    staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    return SUCCEEDED(d3d->CreateBuffer(&staging, nullptr, m_diffStaging.GetAddressOf()));
}

bool CaptureEngine::prepare(HWND target) {
    using namespace winrt::Windows::Graphics::Capture;
    if (!m_winrtDevice || !target) return false;
    if (!GraphicsCaptureSession::IsSupported()) {
        std::cerr << "Error: Windows.Graphics.Capture is not supported on this OS.\n";
        return false;
    }
    m_target = target;
    try {
        auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        const HRESULT hr = interop->CreateForWindow(target, winrt::guid_of<GraphicsCaptureItem>(),
                                                    winrt::put_abi(m_item));
        if (FAILED(hr) || !m_item) {
            std::cerr << "Error: CreateForWindow failed (0x" << std::hex << hr << std::dec << ")\n";
            return false;
        }
        const auto itemSize = m_item.Size();
        m_width = static_cast<uint32_t>(itemSize.Width);
        m_height = static_cast<uint32_t>(itemSize.Height);
        m_cropX = m_cropY = 0;

        // Capture the client area only: the window frame never moves and
        // would anchor static blocks around the content.
        RECT clientRect = {}, windowRect = {};
        POINT clientOrigin = {};
        if (GetClientRect(target, &clientRect) && GetWindowRect(target, &windowRect) &&
            ClientToScreen(target, &clientOrigin)) {
            const uint32_t cw = static_cast<uint32_t>(clientRect.right - clientRect.left);
            const uint32_t ch = static_cast<uint32_t>(clientRect.bottom - clientRect.top);
            const uint32_t cx = static_cast<uint32_t>(std::max(0L, clientOrigin.x - windowRect.left));
            const uint32_t cy = static_cast<uint32_t>(std::max(0L, clientOrigin.y - windowRect.top));
            if (cw > 0 && ch > 0 && cx + cw <= m_width && cy + ch <= m_height) {
                m_cropX = cx;
                m_cropY = cy;
                m_width = cw;
                m_height = ch;
            }
        }

        m_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            m_winrtDevice, winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            4, itemSize);
        m_revoker = m_pool.FrameArrived(winrt::auto_revoke, {this, &CaptureEngine::onFrameArrived});
        m_session = m_pool.CreateCaptureSession(m_item);
        try { m_session.IsCursorCaptureEnabled(false); } catch (...) {}
        try { m_session.IsBorderRequired(false); } catch (...) {}
        try {
            // Ask for every repaint; identical ones are rejected by content.
            m_session.MinUpdateInterval(std::chrono::milliseconds(1));
        } catch (...) {}
    } catch (const winrt::hresult_error& e) {
        std::cerr << "WinRT error preparing capture: " << winrt::to_string(e.message()) << "\n";
        return false;
    }

    for (int i = 0; i < kSlots; ++i) {
        Slot& slot = m_slots[static_cast<size_t>(i)];
        if (!m_device->createSharedTexture(m_width, m_height, DXGI_FORMAT_B8G8R8A8_UNORM, slot.texture)) return false;
        if (FAILED(m_device->captureDevice()->CreateShaderResourceView(
                slot.texture.texture11.Get(), nullptr, m_slotSrv[static_cast<size_t>(i)].GetAddressOf()))) return false;
    }
    std::cout << "Windows Graphics Capture prepared (client " << m_width << "x" << m_height
              << ", crop " << m_cropX << "," << m_cropY << ", " << kSlots << " slots).\n";
    return true;
}

bool CaptureEngine::start() {
    if (!m_session || m_running) return false;
    m_running = true;
    m_producer = std::thread([this] { producerLoop(); });
    try {
        m_session.StartCapture();
    } catch (const winrt::hresult_error& e) {
        std::cerr << "WinRT error starting capture: " << winrt::to_string(e.message()) << "\n";
        m_running = false;
        SetEvent(m_frameEvent);
        m_producer.join();
        return false;
    }
    return true;
}

void CaptureEngine::stop() {
    if (m_running) {
        m_running = false;
        SetEvent(m_frameEvent);
        if (m_producer.joinable()) m_producer.join();
    }
    m_revoker.revoke();
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        for (auto& q : m_queue) q.frame.Close();
        m_queue.clear();
    }
    if (m_session) { m_session.Close(); m_session = nullptr; }
    if (m_pool) { m_pool.Close(); m_pool = nullptr; }
    m_item = nullptr;
}

// ------------------------------------------------------------- callback

void CaptureEngine::onFrameArrived(
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
    winrt::Windows::Foundation::IInspectable const&) {
    if (!m_running) return;
    const int64_t arrival = wallClock100ns();
    std::lock_guard<std::mutex> lock(m_queueMutex);
    while (auto frame = sender.TryGetNextFrame()) {
        {
            std::lock_guard<std::mutex> stats(m_statsMutex);
            ++m_stats.arrivals;
        }
        if (m_queue.size() >= kCallbackQueueLimit) {
            // The producer is behind by several frames; the oldest is stale.
            m_queue.front().frame.Close();
            m_queue.pop_front();
            std::lock_guard<std::mutex> stats(m_statsMutex);
            ++m_stats.queueDrops;
        }
        m_queue.push_back({frame, arrival});
    }
    SetEvent(m_frameEvent);
}

// ------------------------------------------------------------- producer

void CaptureEngine::producerLoop() {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
    } catch (...) {}
    DWORD mmcssIndex = 0;
    HANDLE mmcss = AvSetMmThreadCharacteristicsW(L"Capture", &mmcssIndex);
    if (!mmcss) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    while (m_running) {
        DWORD timeout = INFINITE;
        if (m_haveDeferred) {
            const int64_t remaining = m_deferDeadline - wallClock100ns();
            timeout = remaining <= 0 ? 0 : static_cast<DWORD>((remaining + 9999) / 10000);
        }
        const DWORD wait = WaitForSingleObject(m_frameEvent, timeout);
        if (!m_running) break;

        if (wait == WAIT_TIMEOUT && m_haveDeferred) {
            // Nothing superseded the held frame: it was a real update.
            const FrameTimeline::IngestPlan plan = m_timeline->onDeferTimeout();
            if (plan.acceptDeferredFirst) acceptFrame(m_deferred);
            m_haveDeferred = false;
        }

        for (;;) {
            QueuedFrame queued;
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                if (m_queue.empty()) break;
                queued = std::move(m_queue.front());
                m_queue.pop_front();
            }
            processFrame(queued);
            if (queued.frame) queued.frame.Close();
        }
    }
    if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
}

int CaptureEngine::acquireSlot() {
    std::lock_guard<std::mutex> lock(m_slotMutex);
    const uint64_t flowDone = m_flow->fence()->GetCompletedValue();
    const uint64_t presentDone = m_presentFence ? m_presentFence->GetCompletedValue() : UINT64_MAX;
    for (int i = 0; i < kSlots; ++i) {
        Slot& slot = m_slots[static_cast<size_t>(i)];
        if (slot.state == SlotState::Retiring && slot.flowFence <= flowDone && slot.presentFence <= presentDone) {
            slot.state = SlotState::Free;
        }
    }
    for (int i = 0; i < kSlots; ++i) {
        Slot& slot = m_slots[static_cast<size_t>(i)];
        if (slot.state == SlotState::Free) {
            slot.state = SlotState::Filling;
            return i;
        }
    }
    return -1;
}

void CaptureEngine::releaseSlot(int slot) {
    if (slot < 0) return;
    std::lock_guard<std::mutex> lock(m_slotMutex);
    Slot& s = m_slots[static_cast<size_t>(slot)];
    // Fence-checked before reuse; a Filling slot that was never published
    // has no GPU readers.
    s.state = s.state == SlotState::Filling ? SlotState::Free : SlotState::Retiring;
}

bool CaptureEngine::runDiff(int slot, int referenceSlot, DiffResult& out) {
    ID3D11DeviceContext* ctx = m_device->captureContext();
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(m_diffConstants.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) return false;
    const uint32_t constants[4] = {m_width, m_height, kDiffStride, 0};
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(m_diffConstants.Get(), 0);

    const UINT zero[4] = {};
    ctx->ClearUnorderedAccessViewUint(m_diffOutputUav.Get(), zero);
    ID3D11ShaderResourceView* srvs[] = {m_slotSrv[static_cast<size_t>(slot)].Get(),
                                        m_slotSrv[static_cast<size_t>(referenceSlot)].Get()};
    ID3D11UnorderedAccessView* uavs[] = {m_diffOutputUav.Get()};
    ID3D11Buffer* cbs[] = {m_diffConstants.Get()};
    ctx->CSSetShader(m_diffShader.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    ctx->CSSetConstantBuffers(0, 1, cbs);
    const uint32_t sx = (m_width + kDiffStride - 1) / kDiffStride;
    const uint32_t sy = (m_height + kDiffStride - 1) / kDiffStride;
    ctx->Dispatch((sx + 15) / 16, (sy + 15) / 16, 1);
    ID3D11ShaderResourceView* nullSrv[2] = {};
    ID3D11UnorderedAccessView* nullUav[1] = {};
    ctx->CSSetShaderResources(0, 2, nullSrv);
    ctx->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
    ctx->CopyResource(m_diffStaging.Get(), m_diffOutput.Get());

    // Blocks until the copy and diff finish: sub-millisecond, and this
    // thread has nothing else to do until the verdict is known.
    if (FAILED(ctx->Map(m_diffStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;
    const uint32_t* result = static_cast<const uint32_t*>(mapped.pData);
    const uint32_t changed = result[0], sum = result[1], count = result[2];
    ctx->Unmap(m_diffStaging.Get(), 0);
    if (count == 0) return false;
    out.changedFraction = static_cast<float>(changed) / static_cast<float>(count);
    out.meanAbsDiff = static_cast<float>(sum) / static_cast<float>(count);
    return true;
}

void CaptureEngine::processFrame(QueuedFrame& queued) {
    auto& frame = queued.frame;
    const int64_t tsRaw = frame.SystemRelativeTime().count();
    const auto contentSize = frame.ContentSize();
    if (static_cast<uint32_t>(contentSize.Width) < m_cropX + m_width ||
        static_cast<uint32_t>(contentSize.Height) < m_cropY + m_height) {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        ++m_stats.sizeMismatch;
        return;
    }
    auto access = frame.Surface().as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    ComPtr<ID3D11Texture2D> source;
    if (FAILED(access->GetInterface(IID_PPV_ARGS(&source)))) return;

    const int slot = acquireSlot();
    if (slot < 0) {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        ++m_stats.queueDrops;
        return;
    }
    Slot& s = m_slots[static_cast<size_t>(slot)];
    ID3D11DeviceContext* ctx = m_device->captureContext();
    D3D11_BOX box = {m_cropX, m_cropY, 0, m_cropX + m_width, m_cropY + m_height, 1};
    ctx->CopySubresourceRegion(s.texture.texture11.Get(), 0, 0, 0, 0, source.Get(), 0, &box);
    // The pool surface can be recycled as soon as the copy is queued.
    frame.Close();
    frame = nullptr;

    DiffResult diff;
    const auto t0 = std::chrono::steady_clock::now();
    const bool haveDiff = m_referenceSlot >= 0 && runDiff(slot, m_referenceSlot, diff);
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.lastDiffMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (haveDiff && diff.changedFraction < kDuplicateFraction) {
        releaseSlot(slot);
        std::lock_guard<std::mutex> lock(m_statsMutex);
        ++m_stats.duplicates;
        return;
    }
    const bool sceneCut = haveDiff && diff.changedFraction > kCutFraction && diff.meanAbsDiff > kCutMeanDiff;

    // Publish the copy to D3D12 (the diff above is ordered before it on the
    // D3D11 queue, so one signal covers both).
    s.readyValue = m_device->signalCaptureReady();
    s.captureId = ++m_captureCounter;
    {
        std::lock_guard<std::mutex> lock(m_slotMutex);
        s.state = SlotState::Live;
    }

    PendingFrame pending;
    pending.slot = slot;
    pending.tsRaw = tsRaw;
    pending.arrival = queued.arrival;
    pending.sceneCut = sceneCut;

    const FrameTimeline::IngestPlan plan = m_timeline->onFrameArrived(tsRaw, queued.arrival);
    if (m_haveDeferred) {
        if (plan.acceptDeferredFirst) acceptFrame(m_deferred);
        else if (plan.dropDeferred) releaseSlot(m_deferred.slot);
        m_haveDeferred = false;
    }
    if (plan.replaceLast) {
        const int replaced = m_timeline->popLast();
        releaseSlot(replaced);
    }
    if (plan.acceptThis) {
        acceptFrame(pending);
    } else if (plan.deferThis) {
        m_deferred = pending;
        m_haveDeferred = true;
        m_deferDeadline = plan.deferDeadline;
    }
    m_referenceSlot = slot;
}

void CaptureEngine::acceptFrame(const PendingFrame& pending) {
    TimelineFrame previous;
    const bool havePrevious = m_timeline->lastFrame(previous);
    TimelineFrame frame;
    frame.slot = pending.slot;
    frame.tsRaw = pending.tsRaw;
    frame.arrival = pending.arrival;
    frame.sceneCut = havePrevious && pending.sceneCut;

    if (havePrevious && !frame.sceneCut) {
        const FlowEngine::Result r = m_flow->submitPair(slotInput(previous.slot), slotInput(pending.slot));
        if (r.entry >= 0) {
            frame.flowEntry = r.entry;
            frame.flowFence = r.fence;
            std::lock_guard<std::mutex> lock(m_slotMutex);
            m_slots[static_cast<size_t>(previous.slot)].flowFence = r.fence;
            m_slots[static_cast<size_t>(pending.slot)].flowFence = r.fence;
        } else {
            // No flow: the pair is shown as a hard switch rather than dropped.
            frame.sceneCut = true;
        }
    }
    m_timeline->push(frame);
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        ++m_stats.accepted;
        if (frame.sceneCut && pending.sceneCut) ++m_stats.sceneCuts;
    }
    SetEvent(m_acceptedEvent);
}

// -------------------------------------------------------- render thread

FlowInput CaptureEngine::slotInput(int slot) const {
    FlowInput input;
    if (slot < 0 || slot >= kSlots) return input;
    std::lock_guard<std::mutex> lock(m_slotMutex);
    const Slot& s = m_slots[static_cast<size_t>(slot)];
    input.texture = s.texture.texture12.Get();
    input.readyValue = s.readyValue;
    input.id = s.captureId;
    input.slot = slot;
    return input;
}

void CaptureEngine::noteSlotPresented(int slot, uint64_t presentFenceValue) {
    if (slot < 0 || slot >= kSlots) return;
    std::lock_guard<std::mutex> lock(m_slotMutex);
    Slot& s = m_slots[static_cast<size_t>(slot)];
    s.presentFence = std::max(s.presentFence, presentFenceValue);
}

void CaptureEngine::releaseSlots(const std::vector<int>& slots) {
    for (int slot : slots) releaseSlot(slot);
}

CaptureEngine::Stats CaptureEngine::stats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}
