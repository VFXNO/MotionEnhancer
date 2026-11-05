#include "FsrUpscaler.h"

#include <windows.h>

#include <cstdio>
#include <string>

#ifdef MOTION_ENHANCER_FSR_FG
#include "ffx_api.h"
#include "ffx_api_loader.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_upscale.h"
#endif

struct FsrUpscaler::Impl {
#ifdef MOTION_ENHANCER_FSR_FG
    HMODULE module = nullptr;
    ffxFunctions api = {};
    ffxContext context = nullptr;
#endif
    uint32_t renderWidth = 0, renderHeight = 0;
    uint32_t displayWidth = 0, displayHeight = 0;
    std::string selected;
};

FsrUpscaler::FsrUpscaler() : m_impl(std::make_unique<Impl>()) {}

FsrUpscaler::~FsrUpscaler() {
#ifdef MOTION_ENHANCER_FSR_FG
    Impl& m = *m_impl;
    if (m.context && m.api.DestroyContext) m.api.DestroyContext(&m.context, nullptr);
    if (m.module) FreeLibrary(m.module);
#endif
}

const std::string& FsrUpscaler::selected() const { return m_impl->selected; }

bool FsrUpscaler::initialize(ID3D12Device* device, uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth,
                             uint32_t displayHeight, std::string& error) {
#ifndef MOTION_ENHANCER_FSR_FG
    (void)device; (void)renderWidth; (void)renderHeight; (void)displayWidth; (void)displayHeight;
    error = "this build has no FSR (FSR SDK 2.4 not found at build time)";
    return false;
#else
    Impl& m = *m_impl;
    m.renderWidth = renderWidth;
    m.renderHeight = renderHeight;
    m.displayWidth = displayWidth;
    m.displayHeight = displayHeight;
    // Only from the executable's own folder, never the search path.
    wchar_t path[MAX_PATH] = {};
    std::wstring folder(path, GetModuleFileNameW(nullptr, path, MAX_PATH));
    folder = folder.substr(0, folder.find_last_of(L"\\/"));
    m.module = LoadLibraryExW((folder + L"\\amd_fidelityfx_loader_dx12.dll").c_str(), nullptr,
                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m.module) {
        error = "amd_fidelityfx_loader_dx12.dll not found next to the executable";
        return false;
    }
    ffxLoadFunctions(&m.api, m.module);
    if (!m.api.CreateContext || !m.api.DestroyContext || !m.api.Query || !m.api.Dispatch) {
        error = "amd_fidelityfx_loader_dx12.dll lacks the FFX API entry points";
        return false;
    }

    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = device;
    ffxCreateContextDescUpscaleVersion version = {};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    version.version = FFX_UPSCALER_VERSION;
    ffxCreateContextDescUpscale create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    // Captured frames: 8-bit sRGB, no exposure known (FSR measures it).
    create.flags = FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    create.maxRenderSize = {renderWidth, renderHeight};
    create.maxUpscaleSize = {displayWidth, displayHeight};
    create.header.pNext = &backend.header;
    backend.header.pNext = &version.header;
    const ffxReturnCode_t code = m.api.CreateContext(&m.context, &create.header, nullptr);
    if (code != FFX_API_RETURN_OK) {
        m.context = nullptr;
        error = "FSR upscaling context: code " + std::to_string(code) +
                " (FSR 4 needs an RX 7000/9000 GPU and Shader Model 6.6)";
        return false;
    }
    ffxQueryGetProviderVersion used = {};
    used.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (m.api.Query(&m.context, &used.header) == FFX_API_RETURN_OK && used.versionName) {
        char id[32] = {};
        snprintf(id, sizeof(id), " (id 0x%llx)", static_cast<unsigned long long>(used.versionId));
        m.selected = std::string(used.versionName) + id;
    } else {
        m.selected = "unknown";
    }
    return true;
#endif
}

bool FsrUpscaler::upscale(ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* motionVectors,
                          ID3D12Resource* depth, ID3D12Resource* output, bool reset, float frameTimeMs,
                          float sharpness, std::string& error) {
#ifndef MOTION_ENHANCER_FSR_FG
    (void)list; (void)color; (void)motionVectors; (void)depth; (void)output; (void)reset; (void)frameTimeMs;
    (void)sharpness;
    error = "this build has no FSR";
    return false;
#else
    Impl& m = *m_impl;
    if (!m.context) {
        error = "FSR upscaling is not initialized";
        return false;
    }
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = list;
    d.color = ffxApiGetResourceDX12(color, FFX_API_RESOURCE_STATE_COMMON);
    d.depth = ffxApiGetResourceDX12(depth, FFX_API_RESOURCE_STATE_COMMON);
    d.motionVectors = ffxApiGetResourceDX12(motionVectors, FFX_API_RESOURCE_STATE_COMMON);
    d.output = ffxApiGetResourceDX12(output, FFX_API_RESOURCE_STATE_COMMON, FFX_API_RESOURCE_USAGE_UAV);
    d.jitterOffset = {0.0f, 0.0f};             // a captured frame: no camera jitter we know of
    d.motionVectorScale = {1.0f, 1.0f};        // render-resolution pixels already
    d.renderSize = {m.renderWidth, m.renderHeight};
    d.upscaleSize = {m.displayWidth, m.displayHeight};
    d.enableSharpening = sharpness > 0.0f;
    d.sharpness = sharpness;
    d.frameTimeDelta = frameTimeMs;
    d.preExposure = 1.0f;
    d.reset = reset;
    d.cameraNear = 0.1f;
    d.cameraFar = 1000.0f;
    d.cameraFovAngleVertical = 1.0f;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    const ffxReturnCode_t code = m.api.Dispatch(&m.context, &d.header);
    if (code != FFX_API_RETURN_OK) {
        error = "FSR upscale dispatch: code " + std::to_string(code);
        return false;
    }
    return true;
#endif
}
