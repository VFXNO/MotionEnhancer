#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct ID3D12Device;
struct ID3D12Fence;
struct ID3D12Resource;

// Optional AI depth (built with MOTION_ENHANCER_AI_DEPTH): MiDaS v2.1 small
// (Intel ISL, MIT licence; models\midas_v21_small_256_fp16.onnx next to the
// executable) run by ONNX Runtime's DirectML execution provider on the
// program's own D3D12 device, on a compute queue of its own. The weights are
// FP16 (onnxconverter-common, input and output kept FP32): the same depth
// within 0.3 levels of 255 on average, the network about 27% faster than
// FP32 on an RX 9070 XT. The model is small enough that its many tiny
// layers and the transfers, not arithmetic, dominate, so RDNA 4's matrix
// units gain little more here.
//
// One call estimates one frame: kSize x kSize RGB in, kSize x kSize
// "nearness" out - the network's relative inverse depth mapped so that its
// 2nd percentile is 0 (far) and its 98th is 1 (near). Only the order of
// surfaces is used (which one is in front), so the network's unknown scale
// and shift do not matter. Blocking; call it from a worker thread.
class DepthEstimator {
public:
    static constexpr int kSize = 256;

    DepthEstimator();
    ~DepthEstimator();
    DepthEstimator(const DepthEstimator&) = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;

    // False (with the reason on stderr) when the build has no AI depth, the
    // runtime DLLs or the model are missing, or the session cannot be made.
    bool initialize(ID3D12Device* device, const std::wstring& modelPath);
    bool ready() const;
    // rgb: kSize * kSize * 3 bytes, row-major RGB. nearness: kSize * kSize.
    bool estimate(const uint8_t* rgb, float* nearness);
    // Wall-clock split of the last estimate(), milliseconds: building the
    // input tensor, the network (upload, GPU, download), and the percentile
    // mapping of its output.
    struct Timings {
        double prepare = 0.0;
        double network = 0.0;
        double finish = 0.0;
    };
    Timings lastTimings() const { return m_timings; }

    // GPU path (DepthService): the tensors stay in D3D12 buffers - default
    // heap, UAV, kInputBytes / kOutputBytes - registered once. runGpu makes
    // the estimator's queue wait for waitFence >= waitValue (the input being
    // written), queues the network on the buffers and signals doneFence =
    // doneValue after it. Input: planar NCHW float, already normalised for
    // MiDaS. Output: the raw inverse depth, kSize x kSize float.
    static constexpr size_t kInputBytes = sizeof(float) * 3 * kSize * kSize;
    static constexpr size_t kOutputBytes = sizeof(float) * kSize * kSize;
    void* registerBuffer(ID3D12Resource* buffer);
    void releaseBuffer(void* handle);
    bool runGpu(void* input, void* output, ID3D12Fence* waitFence, uint64_t waitValue,
                ID3D12Fence* doneFence, uint64_t doneValue);

    // models\midas_v21_small_256_fp16.onnx next to the executable.
    static std::wstring defaultModelPath();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    Timings m_timings;
};
