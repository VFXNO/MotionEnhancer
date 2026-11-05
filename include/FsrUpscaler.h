#pragma once

#include <d3d12.h>

#include <cstdint>
#include <memory>
#include <string>

// AMD FSR upscaling (FSR SDK 2.4: ML "FSR Upscaling 4" on RX 7000/9000,
// analytical 3.1 elsewhere) as a plain GPU pass: render-resolution colour,
// motion vectors and depth in, a display-resolution frame out.
//
// Experimental and separate from the program's own engines: used only by
// --fsr-upscale-test so far. FSR expects a game engine's inputs with camera
// jitter; here the colour is a captured, finished frame (no jitter), the
// motion vectors come from this program's optical flow and the depth from
// the AI depth or a constant. Built with the frame generation (the SDK in
// third_party/FidelityFX-SDK-v2.4.0, MOTION_ENHANCER_FSR_FG); its signed
// amd_fidelityfx_upscaler_dx12.dll is loaded at run time from the
// executable's folder through the FFX loader.
class FsrUpscaler {
public:
    FsrUpscaler();
    ~FsrUpscaler();
    FsrUpscaler(const FsrUpscaler&) = delete;
    FsrUpscaler& operator=(const FsrUpscaler&) = delete;

    // renderSize: the input frames, motion vectors and depth; displaySize:
    // the output. The colour is 8-bit sRGB (perceptual), as captured.
    bool initialize(ID3D12Device* device, uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth,
                    uint32_t displayHeight, std::string& error);

    // One frame, recorded into `list`: `color`, `motionVectors` (this frame
    // -> the previous one, render-resolution pixels) and `depth` at render
    // size, `output` (UAV) at display size. reset starts a new sequence.
    // sharpness: FSR's own sharpening, 0 = off. All resources are expected
    // in the COMMON state.
    bool upscale(ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* motionVectors,
                 ID3D12Resource* depth, ID3D12Resource* output, bool reset, float frameTimeMs, float sharpness,
                 std::string& error);

    // The provider the created context runs (name and id), for reports.
    const std::string& selected() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
