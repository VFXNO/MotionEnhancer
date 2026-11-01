#include "UiDetector.h"

#include "FlowEngine.h"
#include "OrtRuntime.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <filesystem>

#ifdef MOTION_ENHANCER_UI_MATRIX
#include "UiMatrixDetector.h"
#endif

#ifdef MOTION_ENHANCER_AI_DEPTH
#include "DirectML.h"
#include "dml_provider_factory.h"
#endif

struct UiDetector::Impl {
    using Clock = std::chrono::steady_clock;
    GraphicsDevice* device = nullptr;
    const ComputeKernels* kernels = nullptr;
    uint32_t width = 0, height = 0;
    ComPtr<ID3D12Fence> inputFence, doneFence;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Resource> planar, input, output, readback;
    HANDLE event = nullptr;
    uint64_t doneValue = 0;
    std::mutex mutex;
    std::condition_variable wake, completed;
    std::thread worker;
    bool stop = false, busy = false, recorded = false, failed = false;
    uint64_t queuedFence = 0, generation = 0, jobGeneration = 0, jobId = 0;
    Clock::time_point jobTime{}, nextRequest{}, resultTime{};
    Snapshot result;
    bool profiling = false;
    std::filesystem::path rawOutput;
    bool checksumRequested = false;
    uint64_t diagnosticChecksum = 0;
#ifdef MOTION_ENHANCER_UI_MATRIX
    std::unique_ptr<UiMatrixDetector> matrix;
#endif
#ifdef MOTION_ENHANCER_AI_DEPTH
    ComPtr<IDMLDevice> dml;
    std::unique_ptr<Ort::Session> session;
    const OrtDmlApi* api = nullptr;
    void* inputHandle = nullptr;
    void* outputHandle = nullptr;
    std::string inputName, outputName;
#endif
    static constexpr size_t inputBytes = 3 * kUiDetectSize * kUiDetectSize * sizeof(float);
    static constexpr size_t outputBytes = 5 * kUiDetectProposals * sizeof(float);

    bool decodeResult(const float* data, std::vector<UiRegion>& regions) {
        if (checksumRequested) {
            diagnosticChecksum = 14695981039346656037ull;
            const auto* bytes = reinterpret_cast<const unsigned char*>(data);
            for (size_t i = 0; i < outputBytes; ++i)
                diagnosticChecksum = (diagnosticChecksum ^ bytes[i]) * 1099511628211ull;
        }
        // Opt-in offline diagnostic only: save ONE raw result without replacing
        // any existing file. In normal capture rawOutput is empty.
        if (!rawOutput.empty()) {
            const HANDLE file = CreateFileW(rawOutput.c_str(), GENERIC_WRITE, 0, nullptr,
                                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file == INVALID_HANDLE_VALUE) return false;
            DWORD written = 0;
            const bool ok = WriteFile(file, data, static_cast<DWORD>(outputBytes), &written, nullptr) != FALSE;
            CloseHandle(file);
            if (!ok || written != outputBytes) return false;
            rawOutput.clear();
        }
        std::vector<UiRegion> previous;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const double age = std::chrono::duration<double, std::milli>(Clock::now() - resultTime).count();
            if (jobGeneration == generation && result.valid &&
                uiRegionsCanContinue(result.frameId, jobId, age)) previous = result.regions;
        }
        // Only CURRENT proposals become boxes. History permits a small score
        // dip / stable capacity selection, not a grace period for missed icons.
        regions = decodeUiRegions(data, kUiDetectProposals, width, height, kUiRegionAcquireConfidence, previous);
        return true;
    }

    ~Impl() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        wake.notify_all();
        if (worker.joinable()) worker.join();
        // Drain queued inference/copies before freeing their resource bindings.
        if (queue && doneFence && event && SUCCEEDED(queue->Signal(doneFence.Get(), ++doneValue)) &&
            SUCCEEDED(doneFence->SetEventOnCompletion(doneValue, event))) WaitForSingleObject(event, 5000);
#ifdef MOTION_ENHANCER_AI_DEPTH
        if (api) {
            for (void* handle : {inputHandle, outputHandle})
                if (handle) if (OrtStatus* status = api->FreeGPUAllocation(handle)) Ort::GetApi().ReleaseStatus(status);
        }
#endif
        if (event) CloseHandle(event);
    }

    bool produce(std::vector<UiRegion>& regions) {
#ifndef MOTION_ENHANCER_AI_DEPTH
        return false;
#else
        if (FAILED(queue->Wait(inputFence.Get(), queuedFence))) return false;
#ifdef MOTION_ENHANCER_UI_MATRIX
        if (matrix) {
            // This bridge runs/waits on the detector WORKER, never capture.
            // Complete the D3D12 producer before Vulkan acquires the imported
            // GPU buffer. No input readback/upload or capture-thread wait.
            if (FAILED(queue->Signal(doneFence.Get(), ++doneValue)) ||
                FAILED(doneFence->SetEventOnCompletion(doneValue, event)) ||
                WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) return false;
            std::vector<float> proposals(5 * kUiDetectProposals);
            const bool ok = matrix->infer(proposals.data());
            if (!ok) return false;
            return decodeResult(proposals.data(), regions);
        }
#endif
        try {
            const Ort::MemoryInfo memory("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
            const int64_t inShape[] = {1, 3, kUiDetectSize, kUiDetectSize};
            const int64_t outShape[] = {1, 5, kUiDetectProposals};
            auto in = Ort::Value::CreateTensor(memory, inputHandle, inputBytes, inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            auto out = Ort::Value::CreateTensor(memory, outputHandle, outputBytes, outShape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            Ort::IoBinding binding(*session);
            binding.BindInput(inputName.c_str(), in);
            binding.BindOutput(outputName.c_str(), out);
            session->Run(Ort::RunOptions{nullptr}, binding);
        } catch (const Ort::Exception& e) {
            std::cerr << "AI UI detector: " << e.what() << "\n";
            return false;
        }
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.Get(), nullptr))) return false;
        // Buffers implicitly promote from COMMON and decay after execution.
        list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, outputBytes);
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(doneFence.Get(), ++doneValue)) ||
            FAILED(doneFence->SetEventOnCompletion(doneValue, event)) ||
            WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) return false;
        float* data = nullptr;
        const D3D12_RANGE range{0, outputBytes};
        if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(&data)))) return false;
        const bool decoded = decodeResult(data, regions);
        const D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
        return decoded;
#endif
    }

    void run() {
        for (;;) {
            uint64_t id, serial;
            Clock::time_point captured;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop || queuedFence != 0; });
                if (stop) return;
                id = jobId; serial = jobGeneration; captured = jobTime;
            }
            const auto start = Clock::now();
            std::vector<UiRegion> regions;
            bool ok = false;
            try { ok = produce(regions); }
            catch (const std::exception& e) { std::cerr << "AI UI detector worker: " << e.what() << "\n"; }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!ok) {
                    result = {};
                    failed = true; // Do not reuse buffers after a timeout/failed GPU submission.
                    std::cerr << "AI UI detector failed: UI-region masking is now empty (no full-screen fallback).\n";
                } else if (serial == generation) {
                    result = {true, id, ms, std::move(regions)};
                    result.diagnosticChecksum = diagnosticChecksum;
#ifdef MOTION_ENHANCER_UI_MATRIX
                    if (matrix) {
                        result.matrixDispatches = matrix->matrixDispatches();
                        result.verifiedWmmaDispatches = matrix->verifiedWmmaDispatches();
                    }
#endif
                    resultTime = captured;
                }
                queuedFence = 0;
                busy = false;
            }
            completed.notify_all();
        }
    }
};

UiDetector::UiDetector() : m_impl(std::make_unique<Impl>()) {}
UiDetector::~UiDetector() = default;

std::wstring UiDetector::defaultModelPath(Choice choice) {
    wchar_t path[32768] = {};
    const bool matrix = choice == Choice::MatrixIcons;
    const DWORD n = GetEnvironmentVariableW(matrix ? L"MOTION_ENHANCER_UI_MATRIX_MODEL" :
        L"MOTION_ENHANCER_UI_MODEL", path, 32768);
    if (n >= 32768) return {}; // Invalid explicit override must not select another file.
    if (n > 0 && n < 32768) return path;
    GetModuleFileNameW(nullptr, path, 32768);
    std::wstring directory(path);
    return directory.substr(0, directory.find_last_of(L"\\/")) + (matrix ? L"\\models\\ui_detect.matrix.param" :
        L"\\models\\ui_detect.onnx");
}

bool UiDetector::initialize(GraphicsDevice* device, const ComputeKernels* kernels, ID3D12Fence* inputFence,
                            uint32_t width, uint32_t height, Choice choice) {
#ifndef MOTION_ENHANCER_AI_DEPTH
    std::cerr << "AI UI detector: this build needs ONNX Runtime / DirectML.\n";
    return false;
#else
    Impl& m = *m_impl;
    if (!device || !kernels || !inputFence || !width || !height || !kernels->pipeline(Kernel::UiDetectInput)) return false;
    m.checksumRequested = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_TEST_CHECKSUM", nullptr, 0) > 0;
    wchar_t raw[32768]{};
    const DWORD rawLength = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_RAW_OUTPUT", raw, 32768);
    if (rawLength >= 32768) return false;
    if (rawLength) {
        m.rawOutput = raw;
        if (std::filesystem::exists(m.rawOutput)) {
            std::cerr << "AI UI detector: refusing to overwrite a raw-output diagnostic.\n";
            return false;
        }
    }
    const auto model = defaultModelPath(choice);
    wchar_t backend[32]{};
    const DWORD backendLength = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_BACKEND", backend, 32);
    if (choice == Choice::Automatic && (backendLength >= 32 ||
        (backendLength && std::wstring(backend) != L"matrix" && std::wstring(backend) != L"directml"))) {
        std::cerr << "AI UI detector: backend must be matrix or directml.\n";
        return false;
    }
    bool matrixRequested = choice == Choice::MatrixIcons ||
        (choice == Choice::Automatic && std::wstring(backend) == L"matrix");
    // An ONNX override still tests exactly that graph. No automatic backend
    // substitution may hide a missing/invalid explicitly requested model.
    // Merely staging an optional graph must NOT change the faster default.
    if (choice == Choice::Automatic && !backendLength && !GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_MODEL", nullptr, 0))
        matrixRequested = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_MATRIX_MODEL", nullptr, 0) > 0;
#ifdef MOTION_ENHANCER_UI_MATRIX
    const std::filesystem::path matrixPath = defaultModelPath(Choice::MatrixIcons);
    if (matrixRequested) {
        m.matrix = std::make_unique<UiMatrixDetector>();
        if (!m.matrix->initialize(device->adapter(), device->device(), matrixPath.wstring())) return false;
        m.input = m.matrix->inputResource();
    }
#else
    if (matrixRequested) {
        std::cerr << "AI UI matrix: backend not compiled; no silent DirectML/CPU fallback.\n";
        return false;
    }
#endif
    if (!matrixRequested && GetFileAttributesW(model.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "AI UI detector: requested model missing. See tools/ui-detect/README.md.\n";
        return false;
    }
    if (!matrixRequested && !ortruntime::load("AI UI detector")) return false;
    m.device = device; m.kernels = kernels; m.inputFence = inputFence;
    m.width = width; m.height = height;
    ID3D12Device* d3d = device->device();
    using CreateDevice = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
    const auto create = matrixRequested ? nullptr : reinterpret_cast<CreateDevice>(GetProcAddress(ortruntime::directml(), "DMLCreateDevice"));
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if ((!matrixRequested && (!create || FAILED(create(d3d, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m.dml))))) ||
        FAILED(d3d->CreateCommandQueue(&desc, IID_PPV_ARGS(&m.queue))) ||
        FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m.doneFence))) ||
        FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m.allocator))) ||
        FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m.allocator.Get(), nullptr, IID_PPV_ARGS(&m.list))) ||
        FAILED(m.list->Close())) return false;
    m.queue->SetName(L"AiUiDetectionQueue");
    m.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m.event) return false;
    if (!matrixRequested) try {
        Ort::SessionOptions options;
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // A requested GPU detector must not quietly execute unsupported INT8
        // operators on the CPU. This affects UI detection only, never flow.
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        wchar_t profile[32768]{};
        const DWORD profileLength = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_PROFILE", profile, 32768);
        if (profileLength > 0 && profileLength < 32768) {
            options.EnableProfiling(profile);
            m.profiling = true;
        }
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m.api)));
        Ort::ThrowOnError(m.api->SessionOptionsAppendExecutionProvider_DML1(options, m.dml.Get(), m.queue.Get()));
        m.session = std::make_unique<Ort::Session>(ortruntime::env(), model.c_str(), options);
        if (m.session->GetInputCount() != 1 || m.session->GetOutputCount() != 1) return false;
        const auto inInfo = m.session->GetInputTypeInfo(0);
        const auto outInfo = m.session->GetOutputTypeInfo(0);
        const auto in = inInfo.GetTensorTypeAndShapeInfo();
        const auto out = outInfo.GetTensorTypeAndShapeInfo();
        if (in.GetShape() != std::vector<int64_t>{1, 3, 640, 640} ||
            out.GetShape() != std::vector<int64_t>{1, 5, 8400} ||
            in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "AI UI detector: incompatible model; expected icon proposals float [1,3,640,640] -> [1,5,8400]. Text/HUD maps are not supported.\n";
            return false;
        }
        Ort::AllocatorWithDefaultOptions a;
        m.inputName = m.session->GetInputNameAllocated(0, a).get();
        m.outputName = m.session->GetOutputNameAllocated(0, a).get();
    } catch (const Ort::Exception& e) {
        std::cerr << "AI UI detector: " << e.what() << "\n";
        return false;
    }
    if (FAILED(device->createTexture2D(640, 3 * 640, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, m.planar, L"UiDetectorInput")) ||
        (!matrixRequested && FAILED(device->createBuffer(Impl::inputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, m.input))) ||
        (!matrixRequested && FAILED(device->createBuffer(m.outputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, m.output))) ||
        (!matrixRequested && FAILED(device->createBuffer(m.outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                     D3D12_RESOURCE_FLAG_NONE, m.readback)))) return false;
    if (!matrixRequested) for (const auto& pair : {std::make_pair(m.input.Get(), &m.inputHandle), std::make_pair(m.output.Get(), &m.outputHandle)}) {
        if (OrtStatus* status = m.api->CreateGPUAllocationFromD3DResource(pair.first, pair.second)) {
            Ort::GetApi().ReleaseStatus(status);
            return false;
        }
    }
    m.worker = std::thread([&m] { m.run(); });
    std::cout << "AI UI regions: " << detectorName() << ", " << backendName()
              << " worker at up to 5Hz (CPU fallback disabled); native per-pixel mask inside boxes only.\n";
    return true;
#endif
}

bool UiDetector::record(ID3D12GraphicsCommandList* list, DispatchContext& dispatch, const FlowInput& frame, bool force) {
    Impl& m = *m_impl;
    {
        std::lock_guard<std::mutex> lock(m.mutex);
        const auto now = Impl::Clock::now();
        if (m.failed || m.busy || (!force && now < m.nextRequest)) return true;
        m.busy = m.recorded = true;
        m.jobId = frame.id; m.jobGeneration = m.generation; m.jobTime = now;
        m.nextRequest = now + std::chrono::milliseconds(kUiDetectIntervalMs);
    }
    ShaderConstants c{};
    c.width = c.levelWidth = m.width; c.height = c.levelHeight = m.height;
    std::array<ID3D12Resource*, 6> in{frame.texture, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (!dispatch.dispatch(list, *m.kernels, Kernel::UiDetectInput, in, m.planar.Get(), c, 40, 40, 0x1)) return false;
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = m.planar.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = m.input.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, 640, 3 * 640, 1, 640 * sizeof(float)};
    DispatchContext::transition(list, m.planar.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    DispatchContext::transition(list, m.planar.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    return true;
}

void UiDetector::submitted(uint64_t fenceValue) {
    Impl& m = *m_impl;
    { std::lock_guard<std::mutex> lock(m.mutex);
      if (!m.recorded) return;
      m.recorded = false; m.queuedFence = fenceValue; }
    m.wake.notify_one();
}

void UiDetector::discarded() {
    Impl& m = *m_impl;
    std::lock_guard<std::mutex> lock(m.mutex);
    if (m.recorded) { m.recorded = false; m.busy = false; m.nextRequest = {}; }
}

void UiDetector::invalidate() {
    Impl& m = *m_impl;
    std::lock_guard<std::mutex> lock(m.mutex);
    ++m.generation; m.result = {}; m.nextRequest = {};
}

UiDetector::Snapshot UiDetector::snapshot(uint64_t currentFrameId) const {
    Impl& m = *m_impl;
    std::lock_guard<std::mutex> lock(m.mutex);
    const double age = std::chrono::duration<double, std::milli>(Impl::Clock::now() - m.resultTime).count();
    return m.result.valid && uiRegionsFresh(m.result.frameId, currentFrameId, age) ? m.result : Snapshot{};
}

bool UiDetector::waitIdle(unsigned timeoutMs) {
    Impl& m = *m_impl;
    std::unique_lock<std::mutex> lock(m.mutex);
    return m.completed.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return !m.busy; }) && !m.failed;
}

std::string UiDetector::finishProfiling() {
#ifdef MOTION_ENHANCER_AI_DEPTH
    Impl& m = *m_impl;
    if (m.profiling && m.session) {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            const auto file = m.session->EndProfilingAllocated(allocator);
            m.profiling = false;
            return file.get() ? file.get() : "";
        } catch (const Ort::Exception& e) {
            std::cerr << "AI UI profiling: " << e.what() << "\n";
        }
    }
#endif
    return {};
}

const char* UiDetector::backendName() const {
#ifdef MOTION_ENHANCER_UI_MATRIX
    if (m_impl->matrix) return "Vulkan cooperative matrix (AMD WMMA verified per inference)";
#endif
    return "DirectML";
}

const char* UiDetector::detectorName() const {
    return "OmniParser V1 640x640";
}
