#include "ComputeKernels.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
std::filesystem::path shaderPath(const char* name) {
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    const auto dir = std::filesystem::path(path).parent_path();
    for (const auto& candidate : {dir / "shaders" / name, dir / name,
                                  std::filesystem::path("shaders") / name}) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) return candidate;
    }
    return {};
}

bool loadBytecode(const char* name, std::vector<uint8_t>& bytes) {
    const auto path = shaderPath(name);
    if (path.empty()) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return !bytes.empty();
}

const char* kShaderFiles[] = {
    "LuminanceCS.cso", "PyramidCS.cso", "MotionSearchCS.cso", "FilterFlowCS.cso",
    "InterpolateCS.cso", "PresentFrameCS.cso", "FfxFlowConvertCS.cso", "PackLumaCS.cso"};
}

bool ComputeKernels::load(ID3D12Device* device) {
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[2].Descriptor.ShaderRegister = 0;

    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    for (int i = 0; i < 2; ++i) {
        samplers[i].Filter = i == 0 ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplers[i].AddressU = samplers[i].AddressV = samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].ShaderRegister = static_cast<UINT>(i);
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC root = {};
    root.NumParameters = 3;
    root.pParameters = params;
    root.NumStaticSamplers = 2;
    root.pStaticSamplers = samplers;

    ComPtr<ID3DBlob> serialized, errors;
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors))) {
        std::cerr << "Error: root signature serialization failed.\n";
        return false;
    }
    if (FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                           IID_PPV_ARGS(&m_rootSignature)))) {
        return false;
    }

    for (size_t i = 0; i < static_cast<size_t>(Kernel::Count); ++i) {
        std::vector<uint8_t> bytes;
        if (!loadBytecode(kShaderFiles[i], bytes)) {
#ifndef MOTION_ENHANCER_FFX_OPTICAL_FLOW
            if (i == static_cast<size_t>(Kernel::FfxFlowConvert)) continue;
#endif
            std::cerr << "Error: missing compiled shader " << kShaderFiles[i]
                      << " (expected next to the executable under shaders/).\n";
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_rootSignature.Get();
        desc.CS = {bytes.data(), bytes.size()};
        if (FAILED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pipelines[i])))) {
            std::cerr << "Error: pipeline creation failed for " << kShaderFiles[i] << "\n";
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------

DispatchContext::~DispatchContext() {
    if (m_constants && m_constantsCpu) m_constants->Unmap(0, nullptr);
}

bool DispatchContext::initialize(ID3D12Device* device, UINT regions, UINT descriptorsPerRegion,
                                 UINT constantsPerRegion) {
    m_device = device;
    m_regions = regions;
    m_descriptorsPerRegion = descriptorsPerRegion;
    m_constantsPerRegion = constantsPerRegion;

    D3D12_DESCRIPTOR_HEAP_DESC heap = {};
    heap.NumDescriptors = regions * descriptorsPerRegion;
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&m_heap)))) return false;
    m_descriptorSize = device->GetDescriptorHandleIncrementSize(heap.Type);

    D3D12_HEAP_PROPERTIES upload = {};
    upload.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256ull * regions * constantsPerRegion;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_constants)))) return false;
    if (FAILED(m_constants->Map(0, nullptr, reinterpret_cast<void**>(&m_constantsCpu)))) return false;
    m_constantsGpu = m_constants->GetGPUVirtualAddress();
    return true;
}

void DispatchContext::beginRegion(UINT region) {
    m_region = region % m_regions;
    m_descriptorCursor = 0;
    m_constantCursor = 0;
}

void DispatchContext::transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                                 D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after || !resource) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

bool DispatchContext::dispatch(ID3D12GraphicsCommandList* list, const ComputeKernels& kernels,
                               Kernel kernel, const std::array<ID3D12Resource*, 4>& inputs,
                               ID3D12Resource* output, const ShaderConstants& constants,
                               UINT groupsX, UINT groupsY, uint32_t implicitInputs) {
    ID3D12PipelineState* pso = kernels.pipeline(kernel);
    if (!pso || !output) return false;
    if (m_descriptorCursor + 5 > m_descriptorsPerRegion || m_constantCursor >= m_constantsPerRegion) {
        std::cerr << "DispatchContext: region exhausted.\n";
        return false;
    }

    const UINT base = m_region * m_descriptorsPerRegion + m_descriptorCursor;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(base) * m_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = m_heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += static_cast<UINT64>(base) * m_descriptorSize;

    D3D12_RESOURCE_BARRIER barriers[5] = {};
    UINT barrierCount = 0;
    for (size_t i = 0; i < 4; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu;
        handle.ptr += static_cast<SIZE_T>(i) * m_descriptorSize;
        if (inputs[i]) {
            m_device->CreateShaderResourceView(inputs[i], nullptr, handle);
            if (implicitInputs & (1u << i)) continue;
            auto& b = barriers[barrierCount++];
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = inputs[i];
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        } else {
            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv = {};
            nullSrv.Format = DXGI_FORMAT_R32G32_FLOAT;
            nullSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            nullSrv.Texture2D.MipLevels = 1;
            m_device->CreateShaderResourceView(nullptr, &nullSrv, handle);
        }
    }
    D3D12_CPU_DESCRIPTOR_HANDLE uav = cpu;
    uav.ptr += static_cast<SIZE_T>(4) * m_descriptorSize;
    m_device->CreateUnorderedAccessView(output, nullptr, nullptr, uav);
    {
        auto& b = barriers[barrierCount++];
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = output;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    m_descriptorCursor += 5;

    const UINT slot = m_region * m_constantsPerRegion + m_constantCursor++;
    std::memcpy(m_constantsCpu + static_cast<size_t>(slot) * 256, &constants, sizeof(constants));
    const D3D12_GPU_VIRTUAL_ADDRESS cb = m_constantsGpu + static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(slot) * 256;

    list->ResourceBarrier(barrierCount, barriers);
    ID3D12DescriptorHeap* heaps[] = {m_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetPipelineState(pso);
    list->SetComputeRootSignature(kernels.rootSignature());
    list->SetComputeRootDescriptorTable(0, gpu);
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = gpu;
    uavGpu.ptr += static_cast<UINT64>(m_descriptorSize) * 4;
    list->SetComputeRootDescriptorTable(1, uavGpu);
    list->SetComputeRootConstantBufferView(2, cb);
    list->Dispatch(groupsX, groupsY, 1);

    // Return everything to COMMON: shared textures must be in COMMON at
    // every cross-API and cross-queue hand-off, and the next pass records
    // its own transitions from there.
    for (UINT i = 0; i < barrierCount; ++i) {
        std::swap(barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
    }
    list->ResourceBarrier(barrierCount, barriers);
    return true;
}
