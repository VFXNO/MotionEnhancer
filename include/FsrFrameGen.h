#pragma once

#include <d3d12.h>

#include <cstdint>
#include <memory>
#include <string>

// AMD FSR frame generation (FSR SDK 2.4: ML "FSR Frame Generation 4.0.2" on
// RX 9000 / Windows 11, analytical 3.1 elsewhere), run as a plain GPU pass:
// no FSR swapchain, the generated frame is written to a texture the caller
// owns (FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, as AMD's own
// sample does without a swapchain context).
//
// Experimental and separate from the program's own engines: it is used only
// by --fsr-fg-offline. FSR expects a game engine's motion vectors, depth and
// camera; here the motion vectors come from this program's optical flow,
// the depth is constant and the camera fixed. Built only when the SDK is in
// third_party/FidelityFX-SDK-v2.4.0 (MOTION_ENHANCER_FSR_FG); its signed
// DLLs (amd_fidelityfx_loader_dx12.dll, amd_fidelityfx_framegeneration_dx12.dll)
// are loaded at run time from the executable's folder.
// FSR's camera for one frame (world space, left-handed: +x right, +y up,
// +z forward when unrotated). The default is the fixed camera at the origin
// looking down +z.
struct FsrCamera {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float right[3] = {1.0f, 0.0f, 0.0f};
    float forward[3] = {0.0f, 0.0f, 1.0f};
    float fovVertical = 1.0f;   // radians (about 57 degrees)
};

class FsrFrameGen {
public:
    FsrFrameGen();
    ~FsrFrameGen();
    FsrFrameGen(const FsrFrameGen&) = delete;
    FsrFrameGen& operator=(const FsrFrameGen&) = delete;

    // Whether this build includes FSR frame generation at all.
    static bool compiledIn();

    // displaySize: the colour frames and the output; renderSize: the motion
    // vectors and depth. colorFormat: the colour frames' and output's format.
    // debugChecking: FSR's own API checks (offline tests only; not for the
    // live overlay).
    bool initialize(ID3D12Device* device, uint32_t displayWidth, uint32_t displayHeight,
                    uint32_t renderWidth, uint32_t renderHeight, DXGI_FORMAT colorFormat, std::string& error,
                    bool debugChecking = false);

    // One source frame, in AMD's required order (configure, prepare,
    // generate), recorded into `list`. `color` is this frame, `motionVectors`
    // and `depth` are this frame's (vectors to the previous frame). From the
    // second frame on, `output` receives the frame halfway between the
    // previous colour and this one. frameId must increase by exactly 1; reset
    // starts a new sequence. All resources are expected in the COMMON state.
    bool frame(ID3D12GraphicsCommandList* list, uint64_t frameId, ID3D12Resource* color,
               ID3D12Resource* motionVectors, ID3D12Resource* depth, ID3D12Resource* output,
               bool reset, float frameTimeMs, std::string& error, const FsrCamera* camera = nullptr) {
        return frame(list, frameId, color, motionVectors, depth, &output, 1, reset, frameTimeMs, error, camera);
    }

    // The same with outputCount (1 to 4) generated frames, written to
    // outputs[0..outputCount-1] (FSR's numGeneratedFrames; which counts above
    // 1 the runtime accepts depends on the FSR version).
    // camera: this frame's camera (null = the fixed default camera).
    bool frame(ID3D12GraphicsCommandList* list, uint64_t frameId, ID3D12Resource* color,
               ID3D12Resource* motionVectors, ID3D12Resource* depth, ID3D12Resource* const* outputs,
               uint32_t outputCount, bool reset, float frameTimeMs, std::string& error,
               const FsrCamera* camera = nullptr);

    // The FSR frame generation versions the runtime offers (first = default).
    const std::string& provider() const;
    // The provider the created context actually runs (name and id, from
    // FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION), and the full paths of
    // the FSR DLLs loaded in the process (the SDK's next to the executable,
    // or amdxcffx64.dll when the AMD driver supplies its own).
    const std::string& selected() const;
    const std::string& modules() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
