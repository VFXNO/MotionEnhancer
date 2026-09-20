#pragma once

#include "GraphicsDevice.h"

#include <array>
#include <cstdint>

enum class Kernel : int {
    Luminance = 0,
    Pyramid,
    MotionSearch,
    FilterFlow,
    Interpolate,
    PresentFrame,
    FfxFlowConvert,
    PackLuma,
    Count
};

// Root signature + one compute PSO per shader. All shaders share the same
// binding layout: t0..t3 SRVs, u0 UAV, b0 constants, s0 linear / s1 point.
class ComputeKernels {
public:
    bool load(ID3D12Device* device);
    ID3D12RootSignature* rootSignature() const { return m_rootSignature.Get(); }
    ID3D12PipelineState* pipeline(Kernel kernel) const {
        return m_pipelines[static_cast<size_t>(kernel)].Get();
    }

private:
    ComPtr<ID3D12RootSignature> m_rootSignature;
    std::array<ComPtr<ID3D12PipelineState>, static_cast<size_t>(Kernel::Count)> m_pipelines;
};

// Descriptor heap and constant-buffer ring split into fixed regions, one per
// command list that can be in flight. A region is rewound when its command
// list is reset, so CPU writes never race the GPU.
class DispatchContext {
public:
    bool initialize(ID3D12Device* device, UINT regions, UINT descriptorsPerRegion,
                    UINT constantsPerRegion);
    ~DispatchContext();

    void beginRegion(UINT region);
    ID3D12DescriptorHeap* heap() const { return m_heap.Get(); }

    // Records: barriers (inputs -> shader read, output -> UAV), descriptors,
    // constants, the dispatch, and barriers back to COMMON.
    //
    // implicitInputs: bit i set means input i is a simultaneous-access
    // resource (the shared capture textures) that is read through implicit
    // COMMON-state promotion with no barrier. Both queues read those
    // textures concurrently; transition barriers on them from two queues
    // are a race the debug layer flags.
    bool dispatch(ID3D12GraphicsCommandList* list, const ComputeKernels& kernels, Kernel kernel,
                  const std::array<ID3D12Resource*, 4>& inputs, ID3D12Resource* output,
                  const ShaderConstants& constants, UINT groupsX, UINT groupsY,
                  uint32_t implicitInputs = 0);

    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    ComPtr<ID3D12Resource> m_constants;
    uint8_t* m_constantsCpu = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS m_constantsGpu = 0;
    UINT m_descriptorSize = 0;
    UINT m_descriptorsPerRegion = 0;
    UINT m_constantsPerRegion = 0;
    UINT m_regions = 0;
    UINT m_region = 0;
    UINT m_descriptorCursor = 0;
    UINT m_constantCursor = 0;
};
