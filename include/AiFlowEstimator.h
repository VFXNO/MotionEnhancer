#pragma once

#include <cstddef>
#include <memory>
#include <string>

struct ID3D12CommandQueue;
struct ID3D12Device;
struct ID3D12Resource;

// The "AI pyramid" flow engine's network (tools/ai-train): a pyramid flow
// model trained on the user's own videos, exported to ONNX and run by ONNX
// Runtime's DirectML execution provider on the program's D3D12 device.
// Built with MOTION_ENHANCER_AI_DEPTH (the ONNX Runtime build).
//
// The model runs at a fixed network resolution (its <model>.json, written
// by mefi.export; width a multiple of 64). Input "frames" [1, 6, H, W]:
// frame0 RGB then frame1 RGB, 0..1, planar. Outputs "flow" [1, 4, H, W]:
// forward F01 (x, y) then backward F10 (x, y) in network pixels, and
// "mask" [1, 1, H, W]. All three are D3D12 buffers (default heap, UAV)
// registered once; run() queues the network on the queue given to
// initialize() - the flow queue, so it orders itself between the lists
// before and after it with no fences - and returns without waiting.
class AiFlowEstimator {
public:
    AiFlowEstimator();
    ~AiFlowEstimator();
    AiFlowEstimator(const AiFlowEstimator&) = delete;
    AiFlowEstimator& operator=(const AiFlowEstimator&) = delete;

    // False (reason on stderr) when the build has no ONNX Runtime, the
    // model or its .json is missing or does not match.
    bool initialize(ID3D12Device* device, ID3D12CommandQueue* queue, const std::wstring& modelPath);
    bool ready() const;
    int width() const;
    int height() const;
    size_t inputBytes() const { return sizeof(float) * 6 * width() * height(); }
    size_t flowBytes() const { return sizeof(float) * 4 * width() * height(); }
    size_t maskBytes() const { return sizeof(float) * width() * height(); }

    void* registerBuffer(ID3D12Resource* buffer);
    void releaseBuffer(void* handle);
    bool run(void* input, void* flow, void* mask);

    // models\ai_flow.onnx next to the executable, or MOTION_ENHANCER_AI_FLOW_MODEL.
    static std::wstring defaultModelPath();

    // Select a paired export without loading DirectML. Unchecked loads the
    // backbone-only .dis.onnx; checked loads the integrated graph. Legacy
    // unrefined models support only DIS. Invalid/mixed pairs fail explicitly.
    static bool selectRefinementModel(const std::wstring& modelPath, bool integrated,
                                      std::wstring& selected, std::string& reason);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
