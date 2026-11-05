#include "FsrFrameGen.h"

#include <windows.h>

#include <string>

#ifdef MOTION_ENHANCER_FSR_FG
#include "ffx_api.h"
#include "ffx_api_loader.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_framegeneration.h"
#endif

struct FsrFrameGen::Impl {
#ifdef MOTION_ENHANCER_FSR_FG
    HMODULE module = nullptr;
    ffxFunctions api = {};
    ffxContext context = nullptr;
#endif
    uint32_t displayWidth = 0, displayHeight = 0;
    uint32_t renderWidth = 0, renderHeight = 0;
    std::string provider;
};

FsrFrameGen::FsrFrameGen() : m_impl(std::make_unique<Impl>()) {}

FsrFrameGen::~FsrFrameGen() {
#ifdef MOTION_ENHANCER_FSR_FG
    Impl& m = *m_impl;
    if (m.context && m.api.DestroyContext) m.api.DestroyContext(&m.context, nullptr);
    if (m.module) FreeLibrary(m.module);
#endif
}

bool FsrFrameGen::compiledIn() {
#ifdef MOTION_ENHANCER_FSR_FG
    return true;
#else
    return false;
#endif
}

const std::string& FsrFrameGen::provider() const { return m_impl->provider; }

#ifdef MOTION_ENHANCER_FSR_FG
namespace {
std::wstring executableFolder() {
    wchar_t path[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring folder(path, n);
    const size_t slash = folder.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : folder.substr(0, slash);
}

std::string returnCodeText(ffxReturnCode_t code) {
    switch (code) {
        case FFX_API_RETURN_OK: return "ok";
        case FFX_API_RETURN_ERROR: return "error";
        case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "unknown description type";
        case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "runtime error (D3D12 or the effect)";
        default: return "code " + std::to_string(code);
    }
}
}  // namespace
#endif

bool FsrFrameGen::initialize(ID3D12Device* device, uint32_t displayWidth, uint32_t displayHeight,
                             uint32_t renderWidth, uint32_t renderHeight, DXGI_FORMAT colorFormat,
                             std::string& error) {
#ifndef MOTION_ENHANCER_FSR_FG
    (void)device; (void)displayWidth; (void)displayHeight; (void)renderWidth; (void)renderHeight; (void)colorFormat;
    error = "this build has no FSR frame generation (FSR SDK 2.4 not found at build time)";
    return false;
#else
    Impl& m = *m_impl;
    m.displayWidth = displayWidth;
    m.displayHeight = displayHeight;
    m.renderWidth = renderWidth;
    m.renderHeight = renderHeight;
    // Only from the executable's own folder, never the search path.
    const std::wstring loader = executableFolder() + L"\\amd_fidelityfx_loader_dx12.dll";
    m.module = LoadLibraryExW(loader.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m.module) {
        error = "amd_fidelityfx_loader_dx12.dll not found next to the executable";
        return false;
    }
    ffxLoadFunctions(&m.api, m.module);
    if (!m.api.CreateContext || !m.api.DestroyContext || !m.api.Configure || !m.api.Query || !m.api.Dispatch) {
        error = "amd_fidelityfx_loader_dx12.dll lacks the FFX API entry points";
        return false;
    }

    // Which frame generation versions this machine offers (the first is the
    // default the runtime picks).
    {
        uint64_t count = 0;
        ffxQueryDescGetVersions query = {};
        query.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        query.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        query.device = device;
        query.outputCount = &count;
        if (m.api.Query(nullptr, &query.header) == FFX_API_RETURN_OK && count > 0) {
            std::string names;
            const char* list[16] = {};
            uint64_t ids[16] = {};
            count = count > 16 ? 16 : count;
            query.versionNames = list;
            query.versionIds = ids;
            if (m.api.Query(nullptr, &query.header) == FFX_API_RETURN_OK) {
                for (uint64_t i = 0; i < count; ++i) {
                    if (!list[i]) continue;
                    if (!names.empty()) names += ", ";
                    names += list[i];
                }
            }
            m.provider = names;
        }
    }

    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = device;
    ffxCreateContextDescFrameGenerationVersion version = {};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    version.version = FFX_FRAMEGENERATION_VERSION;
    ffxCreateContextDescFrameGeneration create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    create.displaySize = {displayWidth, displayHeight};
    create.maxRenderSize = {renderWidth, renderHeight};
    create.backBufferFormat = ffxApiGetSurfaceFormatDX12(colorFormat);
    create.header.pNext = &backend.header;
    backend.header.pNext = &version.header;
    const ffxReturnCode_t code = m.api.CreateContext(&m.context, &create.header, nullptr);
    if (code != FFX_API_RETURN_OK) {
        m.context = nullptr;
        error = "FSR frame generation context: " + returnCodeText(code) +
                " (it needs Windows 11, an RX 9000 GPU for the ML version, and Shader Model 6.6)";
        return false;
    }
    return true;
#endif
}

bool FsrFrameGen::frame(ID3D12GraphicsCommandList* list, uint64_t frameId, ID3D12Resource* color,
                        ID3D12Resource* motionVectors, ID3D12Resource* depth, ID3D12Resource* const* outputs,
                        uint32_t outputCount, bool reset, float frameTimeMs, std::string& error) {
#ifndef MOTION_ENHANCER_FSR_FG
    (void)list; (void)frameId; (void)color; (void)motionVectors; (void)depth; (void)outputs; (void)outputCount;
    (void)reset; (void)frameTimeMs;
    error = "this build has no FSR frame generation";
    return false;
#else
    Impl& m = *m_impl;
    if (!m.context) {
        error = "FSR frame generation is not initialized";
        return false;
    }
    if (!outputs || outputCount < 1 || outputCount > 4) {
        error = "FSR frame generation takes 1 to 4 outputs";
        return false;
    }
    const FfxApiRect2D whole = {0, 0, static_cast<int32_t>(m.displayWidth), static_cast<int32_t>(m.displayHeight)};
    const uint32_t flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;

    // 1. Configure (every frame, AMD's required order).
    ffxConfigureDescFrameGeneration configure = {};
    configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    configure.swapChain = nullptr;
    configure.frameGenerationEnabled = true;
    configure.allowAsyncWorkloads = false;
    configure.flags = flags;
    configure.generationRect = whole;
    configure.frameID = frameId;
    ffxReturnCode_t code = m.api.Configure(&m.context, &configure.header);
    if (code != FFX_API_RETURN_OK) {
        error = "FSR configure: " + returnCodeText(code);
        return false;
    }

    // 2. Prepare: this frame's motion vectors and depth. No engine camera:
    // a fixed camera at the origin looking down +z, a flat scene at the
    // constant depth.
    ffxDispatchDescFrameGenerationPrepareV2 prepare = {};
    prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    prepare.frameID = frameId;
    prepare.flags = flags;
    prepare.commandList = list;
    prepare.renderSize = {m.renderWidth, m.renderHeight};
    prepare.jitterOffset = {0.0f, 0.0f};
    prepare.motionVectorScale = {1.0f, 1.0f};   // already render-size pixels (FsrMotionCS)
    prepare.frameTimeDelta = frameTimeMs;
    prepare.reset = reset;
    prepare.cameraNear = 0.1f;
    prepare.cameraFar = 1000.0f;
    prepare.cameraFovAngleVertical = 1.0f;      // about 57 degrees
    prepare.viewSpaceToMetersFactor = 1.0f;
    prepare.depth = ffxApiGetResourceDX12(depth, FFX_API_RESOURCE_STATE_COMMON);
    prepare.motionVectors = ffxApiGetResourceDX12(motionVectors, FFX_API_RESOURCE_STATE_COMMON);
    const float position[3] = {0.0f, 0.0f, 0.0f}, up[3] = {0.0f, 1.0f, 0.0f};
    const float right[3] = {1.0f, 0.0f, 0.0f}, forward[3] = {0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 3; ++i) {
        prepare.cameraPosition[i] = position[i];
        prepare.cameraUp[i] = up[i];
        prepare.cameraRight[i] = right[i];
        prepare.cameraForward[i] = forward[i];
    }
    code = m.api.Dispatch(&m.context, &prepare.header);
    if (code != FFX_API_RETURN_OK) {
        error = "FSR prepare: " + returnCodeText(code);
        return false;
    }

    // 3. Generate the frame between the previous colour and this one.
    ffxDispatchDescFrameGeneration generate = {};
    generate.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    generate.commandList = list;
    generate.presentColor = ffxApiGetResourceDX12(color, FFX_API_RESOURCE_STATE_COMMON);
    for (uint32_t i = 0; i < outputCount; ++i)
        generate.outputs[i] = ffxApiGetResourceDX12(outputs[i], FFX_API_RESOURCE_STATE_COMMON, FFX_API_RESOURCE_USAGE_UAV);
    generate.numGeneratedFrames = outputCount;
    generate.reset = reset;
    generate.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    generate.minMaxLuminance[0] = 0.0f;
    generate.minMaxLuminance[1] = 1.0f;
    generate.generationRect = whole;
    generate.frameID = frameId;
    code = m.api.Dispatch(&m.context, &generate.header);
    if (code != FFX_API_RETURN_OK) {
        error = "FSR generate: " + returnCodeText(code);
        return false;
    }
    return true;
#endif
}
