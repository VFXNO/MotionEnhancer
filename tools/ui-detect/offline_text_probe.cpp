// Standalone, opt-in offline model evaluator. NOT part of motion_enhancer.
// Uses existing ORT/DirectML DLLs read-only. No capture, model activation, native
// mask, training or CPU execution-provider fallback. Input preparation/upload
// is outside the measured GPU-bound Run + output-readback worker wall time.
#define ORT_API_MANUAL_INIT
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include "DirectML.h"
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void check(HRESULT status) {
    if (FAILED(status)) throw std::runtime_error("D3D12/DirectML operation failed: " + std::to_string(status));
}

static std::vector<int64_t> tensorShape(const Ort::TypeInfo& info) {
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Probe requires float32 external I/O");
    auto shape = tensor.GetShape();
    for (int64_t d : shape) if (d <= 0) throw std::runtime_error("Probe requires static positive dimensions");
    return shape;
}

static size_t elements(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t d : shape) {
        if (static_cast<size_t>(d) > (size_t{1} << 28) / count) throw std::runtime_error("Tensor exceeds offline probe limit");
        count *= static_cast<size_t>(d);
    }
    return count;
}

struct OfflineGpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t serial = 0;

    ~OfflineGpu() { if (event) CloseHandle(event); }
    void wait() {
        check(queue->Signal(fence.Get(), ++serial));
        check(fence->SetEventOnCompletion(serial, event));
        if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0) throw std::runtime_error("Offline GPU timeout");
    }
    void begin() { check(allocator->Reset()); check(list->Reset(allocator.Get(), nullptr)); }
    void execute() {
        check(list->Close());
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        wait();
    }
    ComPtr<ID3D12Resource> buffer(size_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = heap;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        ComPtr<ID3D12Resource> result;
        check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state,
                                              nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
};

struct Allocation {
    const OrtDmlApi* api;
    void* handle = nullptr;
    ~Allocation() { if (handle) if (auto* e = api->FreeGPUAllocation(handle)) Ort::GetApi().ReleaseStatus(e); }
};

int wmain(int argc, wchar_t** argv) {
    if (argc != 7) {
        std::cerr << "Usage: offline_text_probe <runtime-dir> <static-model.onnx> <input-manifest.txt> <NEW-out-dir> <runs> <profile:0|1>\n";
        return 1;
    }
    try {
        const fs::path runtime(argv[1]), model(argv[2]), manifest(argv[3]), out(argv[4]);
        const int runs = std::stoi(argv[5]);
        if (runs < 4 || runs > 500 || fs::exists(out)) throw std::runtime_error("Need 4..500 runs and NEW output folder");
        std::vector<fs::path> inputs;
        std::ifstream lines(manifest);
        for (std::string s; std::getline(lines, s); ) {
            if (!s.empty() && s.back() == '\r') s.pop_back();
            if (!s.empty()) inputs.emplace_back(fs::u8path(s));
        }
        if (!lines.eof() || inputs.empty()) throw std::runtime_error("Missing/empty input manifest");
        fs::create_directories(out);
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS);
        const auto ortDll = LoadLibraryExW((runtime / L"onnxruntime.dll").c_str(), nullptr,
                                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        const auto dmlDll = LoadLibraryExW((runtime / L"DirectML.dll").c_str(), nullptr,
                                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!ortDll || !dmlDll) throw std::runtime_error("Cannot load existing runtime DLLs");
        using GetBase = const OrtApiBase*(ORT_API_CALL*)();
        const auto getBase = reinterpret_cast<GetBase>(GetProcAddress(ortDll, "OrtGetApiBase"));
        if (!getBase) throw std::runtime_error("OrtGetApiBase unavailable");
        Ort::InitApi(getBase()->GetApi(ORT_API_VERSION));
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "offline-text-only");
        ComPtr<IDXGIFactory1> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        SIZE_T best = 0;
        for (UINT i = 0; ; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            check(candidate->GetDesc1(&desc));
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (!adapter || desc.DedicatedVideoMemory > best)) {
                adapter = candidate; best = desc.DedicatedVideoMemory;
            }
        }
        if (!adapter) throw std::runtime_error("No hardware GPU");
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc));
        std::wcout << L"Offline DirectML adapter: " << desc.Description << L"\n";
        OfflineGpu gpu;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&gpu.device)));
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(gpu.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&gpu.queue)));
        check(gpu.device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&gpu.allocator)));
        check(gpu.device->CreateCommandList(0, queueDesc.Type, gpu.allocator.Get(), nullptr, IID_PPV_ARGS(&gpu.list)));
        check(gpu.list->Close());
        check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence)));
        gpu.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!gpu.event) throw std::runtime_error("Cannot create fence event");
        using CreateDml = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
        const auto createDml = reinterpret_cast<CreateDml>(GetProcAddress(dmlDll, "DMLCreateDevice"));
        if (!createDml) throw std::runtime_error("DMLCreateDevice unavailable");
        ComPtr<IDMLDevice> dml;
        check(createDml(gpu.device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml)));
        const OrtDmlApi* api = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&api)));
        Ort::SessionOptions options;
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        const bool profile = std::wstring(argv[6]) == L"1";
        if (profile) options.EnableProfiling((out / "placement").c_str());
        Ort::ThrowOnError(api->SessionOptionsAppendExecutionProvider_DML1(options, dml.Get(), gpu.queue.Get()));
        Ort::Session session(env, model.c_str(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::runtime_error("Probe needs exactly one input and one output");
        const auto inputShape = tensorShape(session.GetInputTypeInfo(0));
        const auto outputShape = tensorShape(session.GetOutputTypeInfo(0));
        const size_t inBytes = elements(inputShape) * sizeof(float), outBytes = elements(outputShape) * sizeof(float);
        auto input = gpu.buffer(inBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto output = gpu.buffer(outBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto upload = gpu.buffer(inBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = gpu.buffer(outBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        Allocation inputAllocation{api}, outputAllocation{api};
        Ort::ThrowOnError(api->CreateGPUAllocationFromD3DResource(input.Get(), &inputAllocation.handle));
        Ort::ThrowOnError(api->CreateGPUAllocationFromD3DResource(output.Get(), &outputAllocation.handle));
        Ort::MemoryInfo memory("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        auto in = Ort::Value::CreateTensor(memory, inputAllocation.handle, inBytes, inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto result = Ort::Value::CreateTensor(memory, outputAllocation.handle, outBytes, outputShape.data(), outputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        Ort::AllocatorWithDefaultOptions ortAllocator;
        const auto inName = session.GetInputNameAllocated(0, ortAllocator), outName = session.GetOutputNameAllocated(0, ortAllocator);
        Ort::IoBinding binding(session);
        binding.BindInput(inName.get(), in);
        binding.BindOutput(outName.get(), result);
        std::vector<float> values(elements(outputShape)), firstOutput;
        auto stage = [&](const fs::path& path) {
            if (fs::file_size(path) != inBytes) throw std::runtime_error("Input float file size does not match model");
            void* p = nullptr;
            check(upload->Map(0, nullptr, &p));
            std::ifstream f(path, std::ios::binary);
            f.read(static_cast<char*>(p), inBytes);
            upload->Unmap(0, nullptr);
            if (!f) throw std::runtime_error("Cannot read input float file");
            gpu.begin();
            gpu.list->CopyBufferRegion(input.Get(), 0, upload.Get(), 0, inBytes);
            gpu.execute();
        };
        auto infer = [&] {
            const auto start = Clock::now();
            session.Run(Ort::RunOptions{nullptr}, binding);
            gpu.begin();
            gpu.list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, outBytes);
            gpu.execute();
            const D3D12_RANGE range{0, outBytes}, none{0, 0};
            void* p = nullptr;
            check(readback->Map(0, &range, &p));
            std::memcpy(values.data(), p, outBytes);
            readback->Unmap(0, &none);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            for (float v : values) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite model output");
            return ms;
        };
        std::ofstream report(out / "timings.json");
        report.precision(9);
        report << "{\"cpu_fallback\":false,\"profiled\":" << (profile ? "true" : "false")
               << ",\"timing_scope\":\"GPU-bound Run + output readback/wait/copy; excludes input preparation/upload, setup, component decode, flow and presentation\",\"output_shape\":[";
        for (size_t j = 0; j < outputShape.size(); ++j) report << (j ? "," : "") << outputShape[j];
        report << "],\"frames\":[";
        for (size_t i = 0; i < inputs.size(); ++i) {
            stage(inputs[i]);
            report << (i ? "," : "") << "{\"input\":\"" << inputs[i].stem().u8string() << "\",\"worker_ms\":[";
            std::vector<float> repeated;
            for (int r = 0; r < runs; ++r) {
                const double ms = infer();
                if (!r) repeated = values;
                else if (std::memcmp(repeated.data(), values.data(), outBytes)) throw std::runtime_error("Identical input produced different output");
                report << (r ? "," : "") << ms;
            }
            report << "]}";
            if (!i) firstOutput = values;
            std::ofstream map(out / (inputs[i].stem().wstring() + L".f32"), std::ios::binary);
            map.write(reinterpret_cast<const char*>(values.data()), outBytes);
            if (!map) throw std::runtime_error("Cannot write new diagnostic output");
        }
        // Verify buffer reuse consumes changed inputs and restores the first
        // input's exact output; no retained/stale map may pass this check.
        stage(inputs.front());
        infer();
        if (std::memcmp(firstOutput.data(), values.data(), outBytes)) throw std::runtime_error("First-input restoration mismatch");
        report << "],\"first_input_restored_exactly\":true}";
        if (!report) throw std::runtime_error("Cannot write timing report");
        if (profile) {
            const auto path = session.EndProfilingAllocated(ortAllocator);
            std::cout << "PROFILE " << path.get() << "\n";
        }
        gpu.wait();
        std::cout << "GPU-only offline evaluation complete; no activation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Offline text probe failed: " << e.what() << "\n";
        return 1;
    }
}
