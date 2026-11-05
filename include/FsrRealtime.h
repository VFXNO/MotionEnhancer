#pragma once

#include "ComputeKernels.h"
#include "FlowEngine.h"
#include "FsrFrameGen.h"
#include "GraphicsDevice.h"

#include <array>
#include <cstdint>
#include <string>

// Experimental, optional (--fsr-fg, off by default): AMD FSR frame generation
// beside the program's own interpolation in the live overlay. Nothing of the
// normal path is replaced: when this is on, each new pair of game frames is
// also given to FSR (FsrFrameGen) with that pair's optical flow as motion
// vectors, and the frame FSR makes halfway between them is shown on refreshes
// whose blend position is close to the middle. Every other refresh, and every
// pair FSR cannot take (scene cuts, pairs without flow), is drawn by the
// normal interpolation exactly as without FSR.
//
// FSR 4.0.2 makes only the middle frame (it accepts more outputs but leaves
// them empty), so FSR covers every generated frame only at 2x output
// (--multiplier 2); at the display rate it covers the refreshes near the
// middle of each pair.
//
// All work is recorded into this class's own command list and submitted on
// the present queue before the presenter's list, so the presenter sees the
// finished frame through queue order. The GPU waits for the capture copies
// and the pair's flow; the CPU never does.
class FsrRealtime {
public:
    struct Stats {
        uint64_t generated = 0;   // pairs FSR made a middle frame for
        uint64_t resets = 0;      // pairs that restarted FSR's sequence
        uint64_t failures = 0;
        uint64_t aiDepth = 0;     // frames given to FSR with their AI depth (the rest: flat)
        float panX = 0.0f;        // the last measured overall motion (full-resolution pixels per
        float panY = 0.0f;        // frame, current -> previous), when the camera is on
    };

    // FSR's camera (--fsr-camera). Fixed: at the origin looking down +z, as
    // without it. FromFlow: turned by the pairs' overall motion (FsrCameraCS),
    // horizontal as yaw and vertical as pitch for a vertical field of view
    // fovVertical; live the motion of the previous pair is used (the CPU never
    // waits for the GPU). FromFlowExact: this pair's own motion, waiting for
    // it (offline tests only). yawSign/pitchSign flip the directions.
    // Spin: turned by a constant kSpinPixels per frame, unrelated to the
    // picture. FSR 4.0.2 does not use the camera as a real one here (the
    // direction and field of view made no difference) but does better once
    // its camera moves at all: measured against the fixed camera on a 1440p
    // game capture +0.72 dB (AI flow) / +0.75 dB (DIS), sharper, better in
    // both moving and still areas, and on the synthetic scroll and stop
    // sequences +0.15 / +0.47 dB; the camera from the flow only gave
    // +0.09 / -0.17 / -0.19 / -0.47 dB. Undocumented FSR behaviour, so an
    // option.
    enum class CameraMode { Fixed, FromFlow, FromFlowExact, Spin };
    static constexpr float kSpinPixels = 20.0f;
    void setCamera(CameraMode mode, float fovVertical, float yawSign = 1.0f, float pitchSign = 1.0f) {
        m_cameraMode = mode;
        m_camera.fovVertical = fovVertical;
        m_yawSign = yawSign;
        m_pitchSign = pitchSign;
    }

    FsrRealtime() = default;
    // Waits until the present queue has finished FSR's work and every list
    // that read its frames, before the textures and FSR's context go.
    ~FsrRealtime();
    FsrRealtime(const FsrRealtime&) = delete;
    FsrRealtime& operator=(const FsrRealtime&) = delete;

    bool initialize(GraphicsDevice* device, const ComputeKernels* kernels, uint32_t width, uint32_t height,
                    std::string& error);

    // The middle frame of the pair (in0, in1), made the first time a pair is
    // seen and returned again for the same pair. index0/index1 are the
    // frames' timeline indices; entry is the pair's flow (valid), ready on
    // flowFence at flowFenceValue. depth0/depth1: the frames' AI depth
    // (FlowEngine::depthFor, complete), or null for the flat depth. Null when
    // FSR has no frame for this pair.
    ID3D12Resource* middleFrame(const FlowInput& in0, uint64_t index0, const FlowInput& in1, uint64_t index1,
                                const FlowEngine::Entry& entry, ID3D12Fence* flowFence, uint64_t flowFenceValue,
                                float frameTimeMs, ID3D12Resource* depth0 = nullptr,
                                ID3D12Resource* depth1 = nullptr);

    // How far FSR's depth is spread by the AI depth (FsrDepthCS): 0 = always
    // flat, ignoring the AI depth; 1 = from the near to the far plane.
    void setDepthSpan(float span) { m_depthSpan = span; }

    // A pair FSR did not see (a scene cut, a pair without flow, a dropped
    // frame) needs nothing from the caller: frame0 is then not the last frame
    // FSR was given, and the next pair starts FSR's sequence again.

    Stats takeStats();

    // The newest measured overall motion (full-resolution pixels, current ->
    // previous) the camera turned by, for reports.
    float motionX() const { return m_motionX; }
    float motionY() const { return m_motionY; }

    // The FSR runtime in use (selected provider, loaded DLLs), for reports.
    const FsrFrameGen& frameGen() const { return m_fsr; }

private:
    static constexpr uint32_t kLists = 3;
    bool feed(const FlowInput& input, ID3D12Resource* backward, uint32_t blockSize, ID3D12Resource* nearness,
              bool reset, ID3D12Resource* output, float frameTimeMs);
    // Next command list from the ring (waits only if it is still in flight).
    bool beginList(uint32_t& slot);
    bool waitFence(uint64_t value);
    // The pair's overall motion (FsrCameraCS) into m_cameraReadback[slot].
    bool recordCamera(ID3D12Resource* backward, uint32_t blockSize, uint32_t slot);
    bool readCamera(uint32_t slot);
    // Turns the camera by the measured motion and refreshes its vectors.
    void turnCamera(float motionX, float motionY);

    GraphicsDevice* m_device = nullptr;
    const ComputeKernels* m_kernels = nullptr;
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_renderWidth = 0, m_renderHeight = 0;
    FsrFrameGen m_fsr;
    DispatchContext m_dispatch;
    ComPtr<ID3D12GraphicsCommandList> m_list;
    std::array<ComPtr<ID3D12CommandAllocator>, kLists> m_allocators;
    std::array<uint64_t, kLists> m_allocatorFence = {};
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    uint32_t m_submits = 0;
    // FSR's inputs: the game frame as RGBA (two, alternating, so the one FSR
    // holds as its previous frame is never overwritten), motion, depth.
    std::array<ComPtr<ID3D12Resource>, 2> m_color;
    ComPtr<ID3D12Resource> m_motion, m_depth;
    // The middle frames, alternating per pair.
    std::array<ComPtr<ID3D12Resource>, 2> m_outputs;
    uint32_t m_current = 0;
    uint64_t m_frameId = 0;
    float m_depthSpan = 0.0f;
    // Camera (setCamera).
    CameraMode m_cameraMode = CameraMode::Fixed;
    FsrCamera m_camera;
    float m_yawSign = 1.0f, m_pitchSign = 1.0f;
    float m_yaw = 0.0f, m_pitch = 0.0f;           // radians
    float m_motionX = 0.0f, m_motionY = 0.0f;     // the newest measured overall motion
    ComPtr<ID3D12Resource> m_cameraResult;        // 1x1 R32G32B32A32_FLOAT (FsrCameraCS)
    std::array<ComPtr<ID3D12Resource>, kLists> m_cameraReadback;
    uint64_t m_cameraPendingFence = 0;            // FromFlow: the submitted, not yet read result
    uint32_t m_cameraPendingSlot = 0;
    uint64_t m_fedIndex = 0;     // the last game frame given to FSR (0 = none)
    uint64_t m_pairIndex = 0;    // index1 of the pair m_outputs[m_current] belongs to
    bool m_pairValid = false;
    Stats m_stats;
    bool m_errorReported = false;
};
