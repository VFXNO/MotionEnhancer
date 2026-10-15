#include "DepthEstimator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifdef MOTION_ENHANCER_AI_DEPTH
// onnxruntime.dll is delay-loaded: the C++ API must not bind to it during
// static initialisation, only after initialize() has loaded it from the
// executable's folder (Windows ships an older onnxruntime.dll of its own).
#define ORT_API_MANUAL_INIT
#include <d3d12.h>
#include <wrl/client.h>
#include "DirectML.h"
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#endif

namespace {
std::wstring executableDirectory() {
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring directory(path);
    const size_t slash = directory.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : directory.substr(0, slash);
}
}  // namespace

struct DepthEstimator::Impl {
#ifdef MOTION_ENHANCER_AI_DEPTH
    Microsoft::WRL::ComPtr<IDMLDevice> dml;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputShape;
    std::vector<float> input;
    std::vector<float> sorted;
#endif
    bool ready = false;
};

DepthEstimator::DepthEstimator() : m_impl(std::make_unique<Impl>()) {}
DepthEstimator::~DepthEstimator() = default;

bool DepthEstimator::ready() const { return m_impl->ready; }

std::wstring DepthEstimator::defaultModelPath() {
    return executableDirectory() + L"\\models\\midas_v21_small_256_fp16.onnx";
}

bool DepthEstimator::initialize(ID3D12Device* device, const std::wstring& modelPath) {
#ifndef MOTION_ENHANCER_AI_DEPTH
    (void)device;
    (void)modelPath;
    std::cerr << "AI depth: not part of this build.\n";
    return false;
#else
    Impl& m = *m_impl;
    m.ready = false;
    const std::wstring directory = executableDirectory();
    HMODULE dmlModule = LoadLibraryExW((directory + L"\\DirectML.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE ortModule = LoadLibraryExW((directory + L"\\onnxruntime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!dmlModule || !ortModule) {
        std::cerr << "AI depth: DirectML.dll or onnxruntime.dll missing next to the executable.\n";
        return false;
    }
    if (GetFileAttributesW(modelPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "AI depth: model not found (models\\midas_v21_small_256_fp16.onnx).\n";
        return false;
    }
    using CreateDevice = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
    const auto createDevice = reinterpret_cast<CreateDevice>(GetProcAddress(dmlModule, "DMLCreateDevice"));
    if (!device || !createDevice ||
        FAILED(createDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m.dml)))) {
        std::cerr << "AI depth: could not create the DirectML device.\n";
        return false;
    }
    // A compute queue of its own: the network never waits behind the
    // presenter's or the flow's lists, nor they behind it.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m.queue)))) return false;
    m.queue->SetName(L"AiDepthQueue");

    try {
        Ort::InitApi();
        m.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MotionEnhancerDepth");
        Ort::SessionOptions options;
        // Required by the DirectML execution provider.
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const OrtDmlApi* dmlApi = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                               reinterpret_cast<const void**>(&dmlApi)));
        Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML1(options, m.dml.Get(), m.queue.Get()));
        m.session = std::make_unique<Ort::Session>(*m.env, modelPath.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        m.inputName = m.session->GetInputNameAllocated(0, allocator).get();
        m.outputName = m.session->GetOutputNameAllocated(0, allocator).get();
        m.inputShape = m.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (m.inputShape.size() != 4) {
            std::cerr << "AI depth: unexpected model input rank " << m.inputShape.size() << ".\n";
            return false;
        }
        // NCHW; free dimensions are fixed to one image of kSize x kSize.
        const int64_t fixed[4] = {1, 3, kSize, kSize};
        for (size_t i = 0; i < 4; ++i)
            if (m.inputShape[i] <= 0) m.inputShape[i] = fixed[i];
        if (m.inputShape[1] != 3 || m.inputShape[2] != kSize || m.inputShape[3] != kSize) {
            std::cerr << "AI depth: the model expects " << m.inputShape[2] << "x" << m.inputShape[3]
                      << " input, not " << kSize << "x" << kSize << ".\n";
            return false;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "AI depth: " << e.what() << "\n";
        return false;
    }
    m.input.assign(static_cast<size_t>(3) * kSize * kSize, 0.0f);
    m.ready = true;
    std::cout << "AI depth: MiDaS v2.1 small on DirectML, " << kSize << "x" << kSize << "\n";
    return true;
#endif
}

bool DepthEstimator::estimate(const uint8_t* rgb, float* nearness) {
#ifndef MOTION_ENHANCER_AI_DEPTH
    (void)rgb;
    (void)nearness;
    return false;
#else
    Impl& m = *m_impl;
    if (!m.ready || !rgb || !nearness) return false;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const Clock::time_point t0 = Clock::now();
    // MiDaS's normalisation: ImageNet mean and deviation, planar RGB.
    static const float kMean[3] = {0.485f, 0.456f, 0.406f};
    static const float kDeviation[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane = static_cast<size_t>(kSize) * kSize;
    for (size_t i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
            m.input[c * plane + i] = (rgb[i * 3 + c] / 255.0f - kMean[c]) / kDeviation[c];
    try {
        const Clock::time_point t1 = Clock::now();
        const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(memory, m.input.data(), m.input.size(),
                                                          m.inputShape.data(), m.inputShape.size());
        const char* inputNames[] = {m.inputName.c_str()};
        const char* outputNames[] = {m.outputName.c_str()};
        std::vector<Ort::Value> outputs = m.session->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
        if (outputs.empty() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != plane) return false;
        const float* depth = outputs[0].GetTensorData<float>();
        // Relative inverse depth (larger = nearer) at an unknown scale and
        // shift: the 2nd and 98th percentiles become far and near.
        const Clock::time_point t2 = Clock::now();
        // Percentiles from every 4th value (a 128 x 128 grid): the same range
        // for a quarter of the selection work (0.5 ms of CPU otherwise).
        m.sorted.clear();
        for (int y = 0; y < kSize; y += 2)
            for (int x = 0; x < kSize; x += 2) m.sorted.push_back(depth[static_cast<size_t>(y) * kSize + x]);
        const size_t samples = m.sorted.size();
        const size_t lowIndex = samples / 50, highIndex = samples - 1 - samples / 50;
        std::nth_element(m.sorted.begin(), m.sorted.begin() + lowIndex, m.sorted.end());
        const float low = m.sorted[lowIndex];
        std::nth_element(m.sorted.begin(), m.sorted.begin() + highIndex, m.sorted.end());
        const float high = m.sorted[highIndex];
        const float scale = high > low ? 1.0f / (high - low) : 0.0f;
        for (size_t i = 0; i < plane; ++i)
            nearness[i] = std::clamp((depth[i] - low) * scale, 0.0f, 1.0f);
        m_timings = {ms(t0, t1), ms(t1, t2), ms(t2, Clock::now())};
    } catch (const Ort::Exception& e) {
        std::cerr << "AI depth: " << e.what() << "\n";
        return false;
    }
    return true;
#endif
}
