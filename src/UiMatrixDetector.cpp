#include "UiMatrixDetector.h"

#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <command.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace {
// ncnn's minimal Vulkan headers omit this optional inspection extension.
// ABI from Khronos VK_AMD_shader_info; no external SDK/loader replacement.
enum class ShaderInfoTypeAMD : int32_t { Disassembly = 2 };
using GetShaderInfoAMD = VkResult(VKAPI_PTR*)(VkDevice, VkPipeline, VkShaderStageFlagBits,
                                              ShaderInfoTypeAMD, size_t*, void*);
// Khronos VK_KHR_external_memory_win32 ABI (not in ncnn's minimal headers).
struct ImportMemoryWin32 {
    VkStructureType sType; const void* pNext;
    VkExternalMemoryHandleTypeFlagBits handleType; HANDLE handle; LPCWSTR name;
};
struct MemoryWin32Properties { VkStructureType sType; void* pNext; uint32_t memoryTypeBits; };
using GetMemoryWin32Properties = VkResult(VKAPI_PTR*)(VkDevice, VkExternalMemoryHandleTypeFlagBits,
                                                     HANDLE, MemoryWin32Properties*);
// These hooks observe the actual SPIR-V, compiled pipeline ISA and commands
// used by this statically linked, UI-only runtime. No DirectML/flow hooks.
struct PipelineEvidence { bool cooperative = false, wmma = false; };
struct DispatchEvidence { unsigned cooperative = 0, wmma = 0; };
struct GraphCapture {
    VkCommandBuffer command = VK_NULL_HANDLE;
    unsigned submissions = 0;
    VkBuffer input = VK_NULL_HANDLE;
    uint32_t family = 0;
};
std::mutex auditMutex;
std::unordered_map<VkShaderModule, bool> modules;
std::unordered_map<VkPipeline, PipelineEvidence> pipelines;
thread_local DispatchEvidence* activeEvidence = nullptr;
thread_local GraphCapture* activeCapture = nullptr;
thread_local std::unordered_map<VkCommandBuffer, VkPipeline> bound;
PFN_vkCreateDevice originalCreateDevice = nullptr;
PFN_vkCreateShaderModule originalCreateModule = nullptr;
PFN_vkDestroyShaderModule originalDestroyModule = nullptr;
PFN_vkCreateComputePipelines originalCreatePipelines = nullptr;
PFN_vkDestroyPipeline originalDestroyPipeline = nullptr;
PFN_vkCmdBindPipeline originalBind = nullptr;
PFN_vkCmdDispatch originalDispatch = nullptr;
PFN_vkBeginCommandBuffer originalBegin = nullptr;
PFN_vkEndCommandBuffer originalEnd = nullptr;
PFN_vkQueueSubmit originalSubmit = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL beginCommand(VkCommandBuffer command, const VkCommandBufferBeginInfo* info) {
    auto copy = *info;
    if (activeCapture) copy.flags &= ~VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const auto result = originalBegin(command, &copy);
    if (result == VK_SUCCESS && activeCapture) {
        // D3D12 completed its producer fence before worker submission. Acquire
        // the shared resource from the external API for this graph execution.
        VkBufferMemoryBarrier acquire{};
        acquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        acquire.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
        acquire.dstQueueFamilyIndex = activeCapture->family;
        acquire.buffer = activeCapture->input;
        acquire.size = VK_WHOLE_SIZE;
        ncnn::vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &acquire, 0, nullptr);
        // Replays reuse intermediate buffers. Order previous GPU reads/writes
        // before this execution, not just the
        // first execution's resource-state metadata.
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ncnn::vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL endCommand(VkCommandBuffer command) {
    if (activeCapture) {
        VkBufferMemoryBarrier release{};
        release.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        release.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        release.srcQueueFamilyIndex = activeCapture->family;
        release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR;
        release.buffer = activeCapture->input;
        release.size = VK_WHOLE_SIZE;
        ncnn::vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &release, 0, nullptr);
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        ncnn::vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    return originalEnd(command);
}

VKAPI_ATTR VkResult VKAPI_CALL submitQueue(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    if (activeCapture) {
        activeCapture->submissions += count;
        if (count == 1 && submits[0].commandBufferCount == 1 &&
            !submits[0].waitSemaphoreCount && !submits[0].signalSemaphoreCount)
            activeCapture->command = submits[0].pCommandBuffers[0];
    }
    return originalSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL createDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkDevice* device) {
    uint32_t count = 0;
    ncnn::vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    ncnn::vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, available.data());
    auto copy = *info;
    std::vector<const char*> extensions(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);
    if (std::any_of(available.begin(), available.end(), [](const auto& e) { return std::strcmp(e.extensionName, "VK_AMD_shader_info") == 0; }) &&
        std::none_of(extensions.begin(), extensions.end(), [](const char* e) { return std::strcmp(e, "VK_AMD_shader_info") == 0; }))
        extensions.push_back("VK_AMD_shader_info");
    if (std::any_of(available.begin(), available.end(), [](const auto& e) { return std::strcmp(e.extensionName, "VK_KHR_external_memory_win32") == 0; }) &&
        std::none_of(extensions.begin(), extensions.end(), [](const char* e) { return std::strcmp(e, "VK_KHR_external_memory_win32") == 0; }))
        extensions.push_back("VK_KHR_external_memory_win32");
    copy.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    copy.ppEnabledExtensionNames = extensions.data();
    return originalCreateDevice(physical, &copy, allocator, device);
}

VKAPI_ATTR VkResult VKAPI_CALL createModule(VkDevice device, const VkShaderModuleCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkShaderModule* module) {
    bool cooperative = false;
    const size_t words = info->codeSize / sizeof(uint32_t);
    for (size_t offset = 5; offset < words;) {
        const uint32_t length = info->pCode[offset] >> 16;
        if (!length || length > words - offset) break;
        // SPIR-V OpCooperativeMatrixMulAddKHR, per the Khronos grammar.
        if ((info->pCode[offset] & 0xffffu) == 4459u) cooperative = true;
        offset += length;
    }
    const auto result = originalCreateModule(device, info, allocator, module);
    if (result == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(auditMutex);
        modules[*module] = cooperative;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroyModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator) {
    { std::lock_guard<std::mutex> lock(auditMutex); modules.erase(module); }
    originalDestroyModule(device, module, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL createPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkComputePipelineCreateInfo* info, const VkAllocationCallbacks* allocator, VkPipeline* result) {
    const auto status = originalCreatePipelines(device, cache, count, info, allocator, result);
    if (status != VK_SUCCESS) return status;
    const auto shaderInfo = reinterpret_cast<GetShaderInfoAMD>(ncnn::vkGetDeviceProcAddr(device, "vkGetShaderInfoAMD"));
    for (uint32_t i = 0; i < count; ++i) {
        if (!result[i]) continue;
        PipelineEvidence evidence;
        { std::lock_guard<std::mutex> lock(auditMutex); evidence.cooperative = modules[info[i].stage.module]; }
        if (evidence.cooperative && shaderInfo) {
            size_t bytes = 0;
            if (shaderInfo(device, result[i], VK_SHADER_STAGE_COMPUTE_BIT, ShaderInfoTypeAMD::Disassembly, &bytes, nullptr) == VK_SUCCESS && bytes > 0) {
                std::vector<char> text(bytes + 1, 0);
                if (shaderInfo(device, result[i], VK_SHADER_STAGE_COMPUTE_BIT, ShaderInfoTypeAMD::Disassembly, &bytes, text.data()) == VK_SUCCESS) {
                    const std::string isa(text.data());
                    evidence.wmma = isa.find("v_wmma_") != std::string::npos;
                    wchar_t folder[32768]{};
                    const DWORD length = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_MATRIX_TRACE", folder, 32768);
                    if (length > 0 && length < 32768) {
                        static std::atomic<unsigned> serial{0};
                        std::error_code error;
                        std::filesystem::create_directories(folder, error);
                        const auto path = std::filesystem::path(folder) / ("pipeline-" + std::to_string(++serial) + ".isa.txt");
                        if (!error && !std::filesystem::exists(path)) std::ofstream(path) << isa;
                    }
                }
            }
        }
        { std::lock_guard<std::mutex> lock(auditMutex); pipelines[result[i]] = evidence; }
    }
    return status;
}

VKAPI_ATTR void VKAPI_CALL destroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator) {
    { std::lock_guard<std::mutex> lock(auditMutex); pipelines.erase(pipeline); }
    originalDestroyPipeline(device, pipeline, allocator);
}

VKAPI_ATTR void VKAPI_CALL bindPipeline(VkCommandBuffer command, VkPipelineBindPoint point, VkPipeline pipeline) {
    if (activeEvidence && point == VK_PIPELINE_BIND_POINT_COMPUTE) bound[command] = pipeline;
    originalBind(command, point, pipeline);
}

VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z) {
    if (activeEvidence && x && y && z) {
        std::lock_guard<std::mutex> lock(auditMutex);
        const auto evidence = pipelines.find(bound[command]);
        if (evidence != pipelines.end()) {
            activeEvidence->cooperative += evidence->second.cooperative ? 1u : 0u;
            activeEvidence->wmma += evidence->second.wmma ? 1u : 0u;
        }
    }
    originalDispatch(command, x, y, z);
}

void installHooks() {
    static std::once_flag once;
    std::call_once(once, [] {
        originalCreateDevice = ncnn::vkCreateDevice; ncnn::vkCreateDevice = createDevice;
        originalCreateModule = ncnn::vkCreateShaderModule; ncnn::vkCreateShaderModule = createModule;
        originalDestroyModule = ncnn::vkDestroyShaderModule; ncnn::vkDestroyShaderModule = destroyModule;
        originalCreatePipelines = ncnn::vkCreateComputePipelines; ncnn::vkCreateComputePipelines = createPipelines;
        originalDestroyPipeline = ncnn::vkDestroyPipeline; ncnn::vkDestroyPipeline = destroyPipeline;
        originalBind = ncnn::vkCmdBindPipeline; ncnn::vkCmdBindPipeline = bindPipeline;
        originalDispatch = ncnn::vkCmdDispatch; ncnn::vkCmdDispatch = dispatch;
        originalBegin = ncnn::vkBeginCommandBuffer; ncnn::vkBeginCommandBuffer = beginCommand;
        originalEnd = ncnn::vkEndCommandBuffer; ncnn::vkEndCommandBuffer = endCommand;
        originalSubmit = ncnn::vkQueueSubmit; ncnn::vkQueueSubmit = submitQueue;
        // Tear down cached Vulkan pipelines before these audit maps/mutexes;
        // ncnn's later global-holder teardown is then an idempotent no-op.
        std::atexit(ncnn::destroy_gpu_instance);
    });
}
}

struct UiMatrixDetector::Impl {
    ncnn::Net net;
    DispatchEvidence evidence;
    ncnn::VkAllocator* blob = nullptr;
    ncnn::VkAllocator* staging = nullptr;
    bool timing = false;
    std::unique_ptr<ncnn::Extractor> extractor;
    std::unique_ptr<ncnn::VkCompute> command;
    ncnn::VkMat hostInput, gpuInput, gpuOutput, fp32Output, hostOutput;
    ComPtr<ID3D12Resource> sharedInput;
    ncnn::VkBufferMemory imported{};
    GraphCapture graph;
    DispatchEvidence graphEvidence;
    VkFence replayFence = VK_NULL_HANDLE;
    ~Impl() {
        const auto* device = net.vulkan_device();
        if (device && replayFence) {
            // Only a worker uses this graph. Keep every descriptor/buffer alive
            // through the last execution, including an error/timeout path.
            ncnn::vkDeviceWaitIdle(device->vkdevice());
            ncnn::vkDestroyFence(device->vkdevice(), replayFence, nullptr);
        }
        command.reset();
        extractor.reset();
        hostInput.release(); gpuInput.release(); gpuOutput.release();
        fp32Output.release(); hostOutput.release();
        if (device && imported.buffer) ncnn::vkDestroyBuffer(device->vkdevice(), imported.buffer, nullptr);
        if (device && imported.memory) ncnn::vkFreeMemory(device->vkdevice(), imported.memory, nullptr);
        net.clear();
        if (device && blob) device->reclaim_blob_allocator(blob);
        if (device && staging) device->reclaim_staging_allocator(staging);
    }
};

UiMatrixDetector::UiMatrixDetector() : m_impl(std::make_unique<Impl>()) {}
UiMatrixDetector::~UiMatrixDetector() = default;

bool UiMatrixDetector::initialize(IDXGIAdapter1* adapter, ID3D12Device* d3d, const std::wstring& paramPath) {
    if (!adapter || !d3d || ncnn::create_gpu_instance() != 0) return false;
    installHooks();
    DXGI_ADAPTER_DESC1 dxgi{};
    if (FAILED(adapter->GetDesc1(&dxgi))) return false;
    int index = -1;
    for (int i = 0; i < ncnn::get_gpu_count(); ++i) {
        const auto& info = ncnn::get_gpu_info(i);
        VkPhysicalDeviceIDProperties id{};
        id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &id;
        ncnn::vkGetPhysicalDeviceProperties2KHR(info.physicalDevice(), &properties);
        if (id.deviceLUIDValid && std::memcmp(id.deviceLUID, &dxgi.AdapterLuid, VK_LUID_SIZE) == 0 &&
            info.vendor_id() == 0x1002 && info.support_cooperative_matrix() && info.support_fp16_storage()) index = i;
    }
    if (index < 0) {
        std::cerr << "AI UI matrix: matching AMD adapter with cooperative-matrix support unavailable.\n";
        return false;
    }
    auto& net = m_impl->net;
    m_impl->timing = GetEnvironmentVariableW(L"MOTION_ENHANCER_UI_MATRIX_TIMING", nullptr, 0) > 0;
    net.opt.use_vulkan_compute = true;
    net.opt.use_cooperative_matrix = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_packed = true;
    // FP32 accumulation reduces numerical differences from the ONNX model.
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_winograd_convolution = false;
    net.opt.num_threads = 1;
    net.set_vulkan_device(index);
    const auto binary = std::filesystem::path(paramPath).replace_extension(L".bin");
    // wchar FILE I/O also supports non-ASCII model paths.
    FILE* param = _wfopen(paramPath.c_str(), L"rb");
    if (!param) return false;
    const int loaded = net.load_param(param);
    fclose(param);
    if (loaded != 0 || !net.opt.use_vulkan_compute) return false;
    for (const auto* layer : net.layers()) {
        if (!layer || !layer->support_vulkan || layer->featmask) {
            std::cerr << "AI UI matrix: CPU/feature-masked layer rejected.\n";
            return false;
        }
    }
    FILE* weights = _wfopen(binary.c_str(), L"rb");
    if (!weights) return false;
    const int weighted = net.load_model(weights);
    fclose(weights);
    if (weighted != 0 || !net.opt.use_vulkan_compute || net.input_names().size() != 1 || net.output_names().size() != 1) return false;
    for (const auto* layer : net.layers())
        if (!layer || !layer->support_vulkan || layer->featmask) return false;
    const auto* device = net.vulkan_device();
    auto& m = *m_impl;
    const auto externalProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceExternalBufferProperties>(
        ncnn::vkGetInstanceProcAddr(ncnn::get_gpu_instance(), "vkGetPhysicalDeviceExternalBufferPropertiesKHR"));
    const auto handleProperties = reinterpret_cast<GetMemoryWin32Properties>(
        ncnn::vkGetDeviceProcAddr(device->vkdevice(), "vkGetMemoryWin32HandlePropertiesKHR"));
    if (!externalProperties || !handleProperties) return false;
    constexpr auto handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    constexpr size_t inputBytes = 3 * 640 * 640 * sizeof(float);
    VkPhysicalDeviceExternalBufferInfo query{};
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    query.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    query.handleType = handleType;
    VkExternalBufferProperties support{};
    support.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    externalProperties(device->info.physicalDevice(), &query, &support);
    if (!(support.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC resource{};
    resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource.Width = inputBytes; resource.Height = resource.DepthOrArraySize = resource.MipLevels = 1;
    resource.SampleDesc.Count = 1; resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &resource,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m.sharedInput)))) return false;
    HANDLE handle = nullptr;
    if (FAILED(d3d->CreateSharedHandle(m.sharedInput.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;
    struct CloseSharedHandle { HANDLE value; ~CloseSharedHandle() { CloseHandle(value); } } close{handle};
    VkExternalMemoryBufferCreateInfo external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external.handleTypes = handleType;
    VkBufferCreateInfo buffer{};
    buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; buffer.pNext = &external;
    buffer.size = inputBytes; buffer.usage = query.usage;
    if (ncnn::vkCreateBuffer(device->vkdevice(), &buffer, nullptr, &m.imported.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), m.imported.buffer, &requirements);
    MemoryWin32Properties properties{static_cast<VkStructureType>(1000073002), nullptr, 0};
    if (handleProperties(device->vkdevice(), handleType, handle, &properties) != VK_SUCCESS) return false;
    const auto& memory = device->info.physicalDeviceMemoryProperties();
    uint32_t type = memory.memoryTypeCount;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((requirements.memoryTypeBits & properties.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
    if (type == memory.memoryTypeCount) return false;
    ImportMemoryWin32 import{static_cast<VkStructureType>(1000073000), nullptr, handleType, handle, nullptr};
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = &import; dedicated.buffer = m.imported.buffer;
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; allocation.pNext = &dedicated;
    allocation.allocationSize = d3d->GetResourceAllocationInfo(0, 1, &resource).SizeInBytes;
    allocation.memoryTypeIndex = type;
    if (allocation.allocationSize < requirements.size ||
        ncnn::vkAllocateMemory(device->vkdevice(), &allocation, nullptr, &m.imported.memory) != VK_SUCCESS ||
        ncnn::vkBindBufferMemory(device->vkdevice(), m.imported.buffer, m.imported.memory, 0) != VK_SUCCESS) return false;
    m.imported.capacity = inputBytes; m.imported.memory_type_index = type;
    m.imported.access_flags = VK_ACCESS_SHADER_READ_BIT;
    m.imported.stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    m.hostInput = ncnn::VkMat(640, 640, 3, &m.imported, sizeof(float), 1, nullptr);
    m.graph.input = m.imported.buffer;
    m.graph.family = device->info.compute_queue_family_index();
    m_impl->blob = device->acquire_blob_allocator();
    m_impl->staging = device->acquire_staging_allocator();
    if (!m_impl->blob || !m_impl->staging) return false;
    net.opt.blob_vkallocator = net.opt.workspace_vkallocator = m_impl->blob;
    net.opt.staging_vkallocator = m_impl->staging;
    VkFenceCreateInfo fence{};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (ncnn::vkCreateFence(device->vkdevice(), &fence, nullptr, &m_impl->replayFence) != VK_SUCCESS) return false;
    std::cout << "AI UI matrix: " << ncnn::get_gpu_info(index).device_name()
              << ", shared D3D12/Vulkan GPU input, cooperative matrix FP16 / FP32 accumulate; CPU model layers forbidden.\n";
    return true;
}

ID3D12Resource* UiMatrixDetector::inputResource() const { return m_impl->sharedInput.Get(); }

bool UiMatrixDetector::infer(float* proposals) {
    auto& m = *m_impl;
    const auto started = std::chrono::steady_clock::now();
    m.evidence = {};
    const auto* device = m.net.vulkan_device();
    const bool first = !m.command;
    if (m.hostInput.empty() || !proposals) return false;
    int submitted = -1;
    auto recorded = std::chrono::steady_clock::now();
    if (first) {
        struct AuditScope {
            AuditScope(DispatchEvidence& e, GraphCapture& g) { bound.clear(); activeEvidence = &e; activeCapture = &g; }
            ~AuditScope() { activeEvidence = nullptr; activeCapture = nullptr; bound.clear(); }
        } audit(m.evidence, m.graph);
        m.command = std::make_unique<ncnn::VkCompute>(device);
        m.extractor = std::make_unique<ncnn::Extractor>(m.net.create_extractor());
        // Retain every intermediate and descriptor for the reusable graph.
        m.extractor->set_light_mode(false);
        device->convert_packing(m.hostInput, m.gpuInput, 1, 2, *m.command, m.net.opt);
        if (m.gpuInput.empty() || m.extractor->input(m.net.input_names()[0], m.gpuInput) != 0 ||
            m.extractor->extract(m.net.output_names()[0], m.gpuOutput, *m.command) != 0 || m.gpuOutput.empty()) return false;
        device->convert_packing(m.gpuOutput, m.fp32Output, 1, 1, *m.command, m.net.opt);
        if (m.fp32Output.empty() || m.fp32Output.w != 8400 || m.fp32Output.h != 5 ||
            m.fp32Output.total() != 5 * 8400 || m.fp32Output.elemsize != sizeof(float)) return false;
        m.hostOutput.create_like(m.fp32Output, m.staging);
        if (m.hostOutput.empty() || !m.hostOutput.mapped_ptr()) return false;
        auto downloadOptions = m.net.opt;
        downloadOptions.blob_vkallocator = m.staging;
        m.command->record_clone(m.fp32Output, m.hostOutput, downloadOptions);
        if (m.hostOutput.empty() || !m.hostOutput.mapped_ptr()) return false;
        recorded = std::chrono::steady_clock::now();
        submitted = m.command->submit_and_wait();
        m.graphEvidence = m.evidence;
    } else if (m.graph.command && m.graph.submissions == 1) {
        // Same verified command buffer, freshly produced shared GPU input. No rerecording,
        // CPU inference, or reuse of the preceding output snapshot.
        if (ncnn::vkResetFences(device->vkdevice(), 1, &m.replayFence) != VK_SUCCESS) return false;
        VkQueue queue = device->acquire_queue(device->info.compute_queue_family_index());
        if (!queue) return false;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &m.graph.command;
        const auto status = ncnn::vkQueueSubmit(queue, 1, &submit, m.replayFence);
        device->reclaim_queue(device->info.compute_queue_family_index(), queue);
        if (status == VK_SUCCESS && ncnn::vkWaitForFences(device->vkdevice(), 1, &m.replayFence,
                VK_TRUE, 5000000000ull) == VK_SUCCESS) {
            submitted = 0;
            m.evidence = m.graphEvidence;
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    if (m.timing) std::cout << "UI_MATRIX_TIMING record_ms="
        << std::chrono::duration<double, std::milli>(recorded - started).count() << " submit_wait_ms="
        << std::chrono::duration<double, std::milli>(finished - recorded).count() << "\n";
    if (submitted != 0 || m.graph.submissions != 1 || !m.graph.command ||
        !m.evidence.cooperative || m.evidence.wmma != m.evidence.cooperative) {
        std::cerr << "AI UI matrix: inference rejected (shape or dispatched WMMA ISA verification failed). "
                  << "cooperative=" << m.evidence.cooperative << ", WMMA=" << m.evidence.wmma << "\n";
        return false;
    }
    if (!m.hostOutput.mapped_ptr()) return false;
    m.staging->invalidate(m.hostOutput.data);
    std::memcpy(proposals, m.hostOutput.mapped_ptr(), 5 * 8400 * sizeof(float));
    return true;
}

unsigned UiMatrixDetector::matrixDispatches() const { return m_impl->evidence.cooperative; }
unsigned UiMatrixDetector::verifiedWmmaDispatches() const { return m_impl->evidence.wmma; }
