#include "AiFlowEstimator.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>
#include <filesystem>

#include "OrtRuntime.h"

#ifdef MOTION_ENHANCER_AI_DEPTH
#include <d3d12.h>
#include <wrl/client.h>
#include "DirectML.h"
#include "dml_provider_factory.h"
#endif

struct AiFlowEstimator::Impl {
#ifdef MOTION_ENHANCER_AI_DEPTH
    Microsoft::WRL::ComPtr<IDMLDevice> dml;
    std::unique_ptr<Ort::Session> session;
    const OrtDmlApi* dmlApi = nullptr;
    std::string inputName, flowName, maskName;
#endif
    int width = 0;
    int height = 0;
    bool ready = false;
};

AiFlowEstimator::AiFlowEstimator() : m_impl(std::make_unique<Impl>()) {}
AiFlowEstimator::~AiFlowEstimator() = default;

bool AiFlowEstimator::ready() const { return m_impl->ready; }
int AiFlowEstimator::width() const { return m_impl->width; }
int AiFlowEstimator::height() const { return m_impl->height; }

std::wstring AiFlowEstimator::defaultModelPath() {
    wchar_t buffer[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"MOTION_ENHANCER_AI_FLOW_MODEL", buffer, MAX_PATH) > 0) return buffer;
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::wstring directory(buffer);
    const size_t slash = directory.find_last_of(L"\\/");
    directory = slash == std::wstring::npos ? std::wstring(L".") : directory.substr(0, slash);
    return directory + L"\\models\\ai_flow.onnx";
}

namespace {
std::wstring metadataPath(const std::wstring& modelPath) {
    auto path = std::filesystem::path(modelPath);
    path.replace_extension(L".json");
    return path.wstring();
}

std::string metadataText(const std::wstring& modelPath) {
    std::ifstream file(metadataPath(modelPath));
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

std::string stringField(const std::string& text, const char* name) {
    std::smatch match;
    return std::regex_search(text, match, std::regex(std::string("\"") + name + "\"\\s*:\\s*\"([^\"]*)\""))
        ? match[1].str() : std::string();
}

bool boolField(const std::string& text, const char* name, bool fallback) {
    std::smatch match;
    return std::regex_search(text, match, std::regex(std::string("\"") + name + "\"\\s*:\\s*(true|false)"))
        ? match[1] == "true" : fallback;
}

// The network resolution from mefi.export's <model>.json.
bool readResolution(const std::wstring& modelPath, int& width, int& height) {
    std::wstring json = modelPath;
    const size_t dot = json.find_last_of(L'.');
    if (dot != std::wstring::npos) json = json.substr(0, dot);
    std::ifstream file(json + L".json");
    if (!file) return false;
    std::stringstream text;
    text << file.rdbuf();
    const std::string s = text.str();
    std::smatch w, h;
    if (!std::regex_search(s, w, std::regex("\"width\"\\s*:\\s*(\\d+)")) ||
        !std::regex_search(s, h, std::regex("\"height\"\\s*:\\s*(\\d+)"))) return false;
    try {
        width = std::stoi(w[1]);
        height = std::stoi(h[1]);
    } catch (const std::exception&) {
        return false;
    }
    return width > 0 && height > 0;
}
}  // namespace

bool AiFlowEstimator::selectRefinementModel(const std::wstring& modelPath, bool integrated,
                                           std::wstring& selected, std::string& reason) {
    selected = modelPath;
    reason.clear();
    const std::string primary = metadataText(modelPath);
    const bool hasRefiner = stringField(primary, "arch") == "pyramid-refine-v1" ||
                            boolField(primary, "integrated_refiner", false);
    if (!hasRefiner) {
        if (!integrated) return true; // Legacy load/errors still handled by initialize().
        reason = "This model has no integrated refiner. Train with refinement and export the pair first.";
        return false;
    }
    auto companion = std::filesystem::path(modelPath);
    companion.replace_extension(L".dis.onnx");
    const std::string bypass = metadataText(companion.wstring());
    int width = 0, height = 0, bypassWidth = 0, bypassHeight = 0;
    std::error_code error;
    const bool primaryExists = std::filesystem::is_regular_file(modelPath, error);
    const bool bypassExists = std::filesystem::is_regular_file(companion, error);
    const std::string bundle = stringField(primary, "bundle_id");
    if (!primaryExists || !bypassExists || bundle.empty() || bundle != stringField(bypass, "bundle_id") ||
        stringField(primary, "arch") != "pyramid-refine-v1" ||
        stringField(bypass, "arch") != "pyramid-v2" || boolField(bypass, "integrated_refiner", true) ||
        stringField(primary, "preset") != stringField(bypass, "preset") ||
        !readResolution(modelPath, width, height) || !readResolution(companion.wstring(), bypassWidth, bypassHeight) ||
        width != bypassWidth || height != bypassHeight) {
        reason = "The AI/DIS export pair is missing or mismatched. Re-export and copy both ONNX files and both JSON files.";
        return false;
    }
    if (!integrated) selected = companion.wstring();
    return true;
}

bool AiFlowEstimator::initialize(ID3D12Device* device, ID3D12CommandQueue* queue, const std::wstring& modelPath) {
#ifndef MOTION_ENHANCER_AI_DEPTH
    (void)device;
    (void)queue;
    (void)modelPath;
    std::cerr << "AI flow: not part of this build (needs ONNX Runtime).\n";
    return false;
#else
    Impl& m = *m_impl;
    m.ready = false;
    if (GetFileAttributesW(modelPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "AI flow: no model at models\\ai_flow.onnx (train and export one with tools\\ai-train).\n";
        return false;
    }
    if (!readResolution(modelPath, m.width, m.height)) {
        std::cerr << "AI flow: the model's .json (its resolution) is missing; export it again.\n";
        return false;
    }
    if (m.width % 64 || m.height % 16) {
        std::cerr << "AI flow: the network width must be a multiple of 64 and its height of 16.\n";
        return false;
    }
    if (!ortruntime::load("AI flow")) return false;
    using CreateDevice = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
    const auto createDevice = reinterpret_cast<CreateDevice>(GetProcAddress(ortruntime::directml(), "DMLCreateDevice"));
    if (!device || !queue || !createDevice ||
        FAILED(createDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m.dml)))) {
        std::cerr << "AI flow: could not create the DirectML device.\n";
        return false;
    }
    try {
        Ort::SessionOptions options;
        options.DisableMemPattern();
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                               reinterpret_cast<const void**>(&m.dmlApi)));
        // The flow queue itself: the network then runs between the list that
        // writes its input and the one that reads its output, in queue order.
        Ort::ThrowOnError(m.dmlApi->SessionOptionsAppendExecutionProvider_DML1(options, m.dml.Get(), queue));
        m.session = std::make_unique<Ort::Session>(ortruntime::env(), modelPath.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        if (m.session->GetInputCount() != 1 || m.session->GetOutputCount() != 2) {
            std::cerr << "AI flow: expected one input and two outputs (flow, mask).\n";
            return false;
        }
        m.inputName = m.session->GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < 2; ++i) {
            const std::string name = m.session->GetOutputNameAllocated(i, allocator).get();
            (name == "mask" ? m.maskName : m.flowName) = name;
        }
        const auto in = m.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in.size() != 4 || in[1] != 6 || in[2] != m.height || in[3] != m.width) {
            std::cerr << "AI flow: the model's input is not [1, 6, " << m.height << ", " << m.width << "].\n";
            return false;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "AI flow: " << e.what() << "\n";
        return false;
    }
    m.ready = true;
    std::cout << "AI flow: pyramid model on DirectML at " << m.width << "x" << m.height << "\n";
    return true;
#endif
}

void* AiFlowEstimator::registerBuffer(ID3D12Resource* buffer) {
#ifdef MOTION_ENHANCER_AI_DEPTH
    Impl& m = *m_impl;
    if (!m.ready || !buffer) return nullptr;
    void* allocation = nullptr;
    if (OrtStatus* status = m.dmlApi->CreateGPUAllocationFromD3DResource(buffer, &allocation)) {
        std::cerr << "AI flow: " << Ort::GetApi().GetErrorMessage(status) << "\n";
        Ort::GetApi().ReleaseStatus(status);
        return nullptr;
    }
    return allocation;
#else
    (void)buffer;
    return nullptr;
#endif
}

void AiFlowEstimator::releaseBuffer(void* handle) {
#ifdef MOTION_ENHANCER_AI_DEPTH
    if (handle && m_impl->dmlApi)
        if (OrtStatus* status = m_impl->dmlApi->FreeGPUAllocation(handle)) Ort::GetApi().ReleaseStatus(status);
#else
    (void)handle;
#endif
}

bool AiFlowEstimator::run(void* input, void* flow, void* mask) {
#ifdef MOTION_ENHANCER_AI_DEPTH
    Impl& m = *m_impl;
    if (!m.ready || !input || !flow || !mask) return false;
    try {
        const Ort::MemoryInfo memory("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
        const int64_t inShape[4] = {1, 6, m.height, m.width};
        const int64_t flowShape[4] = {1, 4, m.height, m.width};
        const int64_t maskShape[4] = {1, 1, m.height, m.width};
        Ort::Value in = Ort::Value::CreateTensor(memory, input, inputBytes(), inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        Ort::Value outFlow = Ort::Value::CreateTensor(memory, flow, flowBytes(), flowShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        Ort::Value outMask = Ort::Value::CreateTensor(memory, mask, maskBytes(), maskShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        Ort::IoBinding binding(*m.session);
        binding.BindInput(m.inputName.c_str(), in);
        binding.BindOutput(m.flowName.c_str(), outFlow);
        binding.BindOutput(m.maskName.c_str(), outMask);
        m.session->Run(Ort::RunOptions{nullptr}, binding);
    } catch (const Ort::Exception& e) {
        std::cerr << "AI flow: " << e.what() << "\n";
        return false;
    }
    return true;
#else
    (void)input; (void)flow; (void)mask;
    return false;
#endif
}
