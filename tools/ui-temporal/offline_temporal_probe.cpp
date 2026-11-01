// Research-only multi-I/O DirectML feasibility probe. NOT an application build.
// Existing runtime DLLs are read-only. No capture, training, model activation or
// CPU execution-provider fallback. Fixed synthetic inputs, NOT live state replay.
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void check(HRESULT hr) {
    if (FAILED(hr)) throw std::runtime_error("GPU operation failed: " + std::to_string(hr));
}

static bool identifier(const std::string& name) {
    return !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    });
}

struct Commands {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
};

struct OfflineGpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t serial = 0;
    ~OfflineGpu() { if (event) CloseHandle(event); }
    void wait() {
        check(queue->Signal(fence.Get(), ++serial));
        check(fence->SetEventOnCompletion(serial, event));
        if (WaitForSingleObject(event, 15000) != WAIT_OBJECT_0) throw std::runtime_error("GPU completion timeout");
    }
    Commands commands() {
        Commands c;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&c.allocator)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, c.allocator.Get(), nullptr, IID_PPV_ARGS(&c.list)));
        check(c.list->Close());
        return c;
    }
    void begin(Commands& c) { check(c.allocator->Reset()); check(c.list->Reset(c.allocator.Get(), nullptr)); }
    void submit(Commands& c, bool complete = true) {
        check(c.list->Close());
        ID3D12CommandList* lists[] = {c.list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (complete) wait();
    }
    ComPtr<ID3D12Resource> buffer(size_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = heap;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = d.DepthOrArraySize = d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = flags;
        ComPtr<ID3D12Resource> resource;
        check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                              IID_PPV_ARGS(&resource)));
        return resource;
    }
};

struct Allocation {
    const OrtDmlApi* api = nullptr;
    void* handle = nullptr;
    ~Allocation() { if (handle) if (auto* error = api->FreeGPUAllocation(handle)) Ort::GetApi().ReleaseStatus(error); }
};

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    size_t bytes = sizeof(float);
    ComPtr<ID3D12Resource> resource, transfer;
    Allocation allocation;
    Ort::Value value{nullptr}; // destroyed before its allocation and buffers
    std::vector<float> host;

    Tensor(OfflineGpu& gpu, const OrtDmlApi* api, const std::string& key, const Ort::TypeInfo& info, bool input)
        : name(key) {
        if (!identifier(name)) throw std::runtime_error("Unsafe tensor identifier");
        auto type = info.GetTensorTypeAndShapeInfo();
        if (type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("All external I/O must be float32");
        shape = type.GetShape();
        if (shape.empty() || shape.size() > 4) throw std::runtime_error("Need positive static tensor dimensions");
        for (int64_t dim : shape) {
            if (dim < 1 || static_cast<uint64_t>(dim) > (uint64_t{1} << 28) / bytes)
                throw std::runtime_error("Dynamic/oversize tensor rejected");
            bytes *= static_cast<size_t>(dim);
        }
        resource = gpu.buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        transfer = gpu.buffer(bytes, input ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK,
                              input ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST);
        allocation.api = api;
        Ort::ThrowOnError(api->CreateGPUAllocationFromD3DResource(resource.Get(), &allocation.handle));
        Ort::MemoryInfo memory("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        value = Ort::Value::CreateTensor(memory, allocation.handle, bytes, shape.data(), shape.size(),
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        if (!input) host.resize(bytes / sizeof(float));
    }
};

struct Case { std::string name; std::vector<fs::path> files; };

static std::vector<Case> cases(const fs::path& path, size_t inputCount) {
    std::ifstream f(path);
    std::vector<Case> result;
    for (std::string line; std::getline(f, line); ) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream fields(line);
        Case item;
        std::getline(fields, item.name, '\t');
        if (!identifier(item.name)) throw std::runtime_error("Invalid case identifier");
        for (const auto& prior : result) if (prior.name == item.name) throw std::runtime_error("Duplicate case identifier");
        for (std::string file; std::getline(fields, file, '\t'); ) {
            const fs::path p = fs::u8path(file);
            if (!p.is_absolute()) throw std::runtime_error("Manifest paths must be absolute");
            item.files.push_back(p);
        }
        if (item.files.size() != inputCount) throw std::runtime_error("Manifest input count mismatch");
        result.push_back(std::move(item));
    }
    if (!f.eof() || result.empty()) throw std::runtime_error("Missing/empty manifest");
    return result;
}

static std::vector<std::vector<float>> readOutputs(OfflineGpu& gpu, Commands& commands,
                                                   std::vector<std::unique_ptr<Tensor>>& outputs) {
    gpu.begin(commands);
    for (auto& t : outputs) commands.list->CopyBufferRegion(t->transfer.Get(), 0, t->resource.Get(), 0, t->bytes);
    gpu.submit(commands);
    std::vector<std::vector<float>> result;
    for (auto& t : outputs) {
        const D3D12_RANGE range{0, t->bytes}, none{0, 0};
        void* pointer = nullptr;
        check(t->transfer->Map(0, &range, &pointer));
        std::memcpy(t->host.data(), pointer, t->bytes);
        t->transfer->Unmap(0, &none);
        for (float v : t->host) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite model output");
        result.push_back(t->host);
    }
    return result;
}

static double difference(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j)
            maximum = std::max(maximum, std::abs(static_cast<double>(a[i][j]) - b[i][j]));
    return maximum;
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 7) {
        std::cerr << "Usage: offline_temporal_probe <runtime-dir> <model> <inputs.tsv> <NEW-out-dir> <4..500-runs> <profile:0|1>\n";
        return 1;
    }
    try {
        const fs::path runtime(argv[1]), model(argv[2]), manifest(argv[3]), out(argv[4]);
        const int runs = std::stoi(argv[5]);
        const std::wstring profileArg(argv[6]);
        if (runs < 4 || runs > 500 || fs::exists(out) || (profileArg != L"0" && profileArg != L"1"))
            throw std::runtime_error("Need bounded runs, explicit profile flag and NEW output folder");
        fs::create_directories(out);
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS);
        HMODULE ort = LoadLibraryExW((runtime / L"onnxruntime.dll").c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        HMODULE dmlModule = LoadLibraryExW((runtime / L"DirectML.dll").c_str(), nullptr,
                                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!ort || !dmlModule) throw std::runtime_error("Cannot load existing runtime DLLs");
        using GetBase = const OrtApiBase*(ORT_API_CALL*)();
        auto getBase = reinterpret_cast<GetBase>(GetProcAddress(ort, "OrtGetApiBase"));
        if (!getBase) throw std::runtime_error("OrtGetApiBase missing");
        Ort::InitApi(getBase()->GetApi(ORT_API_VERSION));
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "offline-temporal-ui");
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
                adapter = candidate;
                best = desc.DedicatedVideoMemory;
            }
        }
        if (!adapter) throw std::runtime_error("No hardware GPU");
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc));
        std::wcout << L"Offline temporal probe adapter: " << desc.Description << L"\n";
        OfflineGpu gpu;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&gpu.device)));
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(gpu.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&gpu.queue)));
        check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence)));
        gpu.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!gpu.event) throw std::runtime_error("Cannot create completion event");
        using CreateDml = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
        auto createDml = reinterpret_cast<CreateDml>(GetProcAddress(dmlModule, "DMLCreateDevice"));
        if (!createDml) throw std::runtime_error("DMLCreateDevice missing");
        ComPtr<IDMLDevice> dml;
        check(createDml(gpu.device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml)));
        const OrtDmlApi* api = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&api)));
        Ort::SessionOptions options;
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        const bool profile = profileArg == L"1";
        if (profile) options.EnableProfiling((out / "placement").c_str());
        Ort::ThrowOnError(api->SessionOptionsAppendExecutionProvider_DML1(options, dml.Get(), gpu.queue.Get()));
        Ort::Session session(env, model.c_str(), options);
        if (session.GetInputCount() < 1 || session.GetInputCount() > 3 || session.GetOutputCount() < 1 || session.GetOutputCount() > 4)
            throw std::runtime_error("Unexpected research I/O count");
        Ort::AllocatorWithDefaultOptions ortAllocator;
        std::vector<std::unique_ptr<Tensor>> inputs, outputs;
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.push_back(std::make_unique<Tensor>(gpu, api, session.GetInputNameAllocated(i, ortAllocator).get(),
                                                       session.GetInputTypeInfo(i), true));
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputs.push_back(std::make_unique<Tensor>(gpu, api, session.GetOutputNameAllocated(i, ortAllocator).get(),
                                                        session.GetOutputTypeInfo(i), false));
        Ort::IoBinding binding(session);
        for (auto& t : inputs) binding.BindInput(t->name.c_str(), t->value);
        for (auto& t : outputs) binding.BindOutput(t->name.c_str(), t->value);
        auto samples = cases(manifest, inputs.size());
        Commands transfer = gpu.commands(), before = gpu.commands(), after = gpu.commands();
        D3D12_QUERY_HEAP_DESC queryDesc{};
        queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryDesc.Count = 2;
        ComPtr<ID3D12QueryHeap> queries;
        check(gpu.device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queries)));
        auto ticks = gpu.buffer(2 * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        uint64_t frequency = 0;
        check(gpu.queue->GetTimestampFrequency(&frequency));
        auto stage = [&](const Case& item) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto& t = inputs[i];
                if (fs::file_size(item.files[i]) != t->bytes) throw std::runtime_error("Input byte count mismatch");
                void* p = nullptr;
                check(t->transfer->Map(0, nullptr, &p));
                std::ifstream data(item.files[i], std::ios::binary);
                data.read(static_cast<char*>(p), static_cast<std::streamsize>(t->bytes));
                t->transfer->Unmap(0, nullptr);
                if (!data) throw std::runtime_error("Input file read failed");
            }
            gpu.begin(transfer);
            for (auto& t : inputs) transfer.list->CopyBufferRegion(t->resource.Get(), 0, t->transfer.Get(), 0, t->bytes);
            gpu.submit(transfer);
        };
        auto infer = [&]() {
            const auto start = Clock::now();
            gpu.begin(before);
            before.list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            gpu.submit(before, false);
            session.Run(Ort::RunOptions{nullptr}, binding);
            gpu.begin(after);
            after.list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            after.list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ticks.Get(), 0);
            gpu.submit(after);
            const double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            const D3D12_RANGE range{0, 2 * sizeof(uint64_t)}, none{0, 0};
            void* p = nullptr;
            check(ticks->Map(0, &range, &p));
            uint64_t values[2];
            std::memcpy(values, p, sizeof(values));
            ticks->Unmap(0, &none);
            return std::pair<double, double>{wall, 1000.0 * static_cast<double>(values[1] - values[0]) / frequency};
        };
        std::ofstream report(out / "timings.json");
        report.precision(10);
        report << "{\"cpu_fallback\":false,\"profiled\":" << (profile ? "true" : "false")
               << ",\"gaming_load_tested\":false,\"timestamp_frequency\":" << frequency
               << ",\"warmups_per_case\":3,\"scope\":\"GPU-bound model Run through queue completion; GPU timeline includes submission gaps. Excludes setup, input preparation/upload, output validation readback, live state ownership/copy, icon inference, native masking, DIS, synthesis and presentation.\",\"outputs\":[";
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto& t = outputs[i];
            report << (i ? "," : "") << "{\"name\":\"" << t->name << "\",\"bytes\":" << t->bytes << ",\"shape\":[";
            for (size_t j = 0; j < t->shape.size(); ++j) report << (j ? "," : "") << t->shape[j];
            report << "]}";
        }
        report << "],\"cases\":[";
        std::vector<std::vector<float>> first;
        for (size_t i = 0; i < samples.size(); ++i) {
            stage(samples[i]);
            const auto cold = infer();
            auto initial = readOutputs(gpu, transfer, outputs);
            for (int warm = 0; warm < 3; ++warm) infer();
            report << (i ? "," : "") << "{\"name\":\"" << samples[i].name << "\",\"first_wall_ms\":" << cold.first
                   << ",\"first_gpu_ms\":" << cold.second << ",\"samples\":[";
            for (int r = 0; r < runs; ++r) {
                const auto time = infer();
                report << (r ? "," : "") << "{\"wall_ms\":" << time.first << ",\"gpu_ms\":" << time.second << "}";
            }
            auto latest = readOutputs(gpu, transfer, outputs);
            report << "],\"repeat_max_abs_drift\":" << difference(initial, latest) << "}";
            if (!i) first = latest;
            for (size_t j = 0; j < outputs.size(); ++j) {
                std::ofstream data(out / (samples[i].name + "-" + outputs[j]->name + ".f32"), std::ios::binary);
                data.write(reinterpret_cast<const char*>(latest[j].data()), static_cast<std::streamsize>(outputs[j]->bytes));
                if (!data) throw std::runtime_error("Cannot write diagnostic output");
            }
        }
        stage(samples.front());
        infer();
        auto restored = readOutputs(gpu, transfer, outputs);
        report << "],\"first_input_restore_max_abs_drift\":" << difference(first, restored) << ",\"completed\":true}";
        report.close();
        if (!report) throw std::runtime_error("Cannot write timing report");
        if (profile) std::cout << "PROFILE " << session.EndProfilingAllocated(ortAllocator).get() << "\n";
        gpu.wait();
        std::cout << "Synthetic GPU-only feasibility probe complete; no activation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Offline temporal probe failed: " << e.what() << "\n";
        return 1;
    }
}
