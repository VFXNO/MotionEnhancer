#pragma once

#include "FlowEngine.h"
#include "GraphicsDevice.h"

#include <cstdint>
#include <string>
#include <vector>

// FSR's camera (FsrRealtime::setCamera, --fsr-camera): 0 = fixed, 1 = from
// the previous pair's overall motion (live), 2 = from the pair's own motion
// (offline tests only), 3 = a constant turn (spin).
struct FsrCameraSettings {
    int mode = 0;
    float fovDegrees = 60.0f;   // assumed vertical field of view of the game
    float yawSign = 1.0f;       // -1 flips the direction of horizontal turns
    float pitchSign = 1.0f;     // -1 flips the direction of vertical turns
};

struct RealtimeOptions {
    std::string windowTitle;
    std::string stopEventName;          // optional GUI-owned graceful stop event
    FlowSettings flow;
    uint32_t sourceFps = 0;          // 0 = auto-detect cadence
    uint32_t outputMultiplier = 0;   // exact source-rate multiplier; 0 = Max (display-rate FPS)
    GraphicsAdapterPreference adapter = GraphicsAdapterPreference::Auto;
    // Fixed buffering: margin above one source period and its cap
    // (<= 0: three periods + 60 ms). Default budget: two periods + 10 ms.
    // (FrameTimeline::setLatencyLimits); frames the swap chain may hold
    // (Presenter, 1-2); captured frames that may wait for the producer
    // (CaptureEngine::setCaptureQueue, 2-8).
    double latencyMinExtraMs = 4.0;
    double latencyMaxMs = 0.0;
    int frameLatency = 1;
    int captureQueue = 4;
    // Legacy --vsync switch now controls tearing only: false allows it.
    // Output always uses the FPS clock and Present(0), never vsync waits.
    // Actual even scanout still needs a compatible display mode or working
    // VRR / independent flip; composed windows remain subject to DWM.
    bool vsync = true;
    // Experimental, beside the normal interpolation (FsrRealtime): AMD FSR
    // frame generation's middle frame is used within fsrWindow of 0.5.
    // Fixed multipliers require exactly 0.5; real frames are never replaced.
    bool fsrFrameGen = false;
    float fsrWindow = 0.1f;
    // How far the frames' AI depth (FlowSettings::aiDepth) spreads FSR's
    // depth (FsrDepthCS): 0 = always flat. Measured on a 1440p game capture
    // (6 pairs): AI flow 37.40 -> 37.47 dB, DIS 37.30 -> 37.19 dB; the
    // reversed order is worse for both, so FSR does use it. The span itself
    // barely matters (0.1-1 within 0.01 dB).
    float fsrDepthSpan = 0.25f;
    FsrCameraSettings fsrCamera;
};

// runRealtimeSession's exit code when the captured window changed size
// (resized, or switched between windowed and fullscreen): every resource is
// sized for the window at the start, so the GUI asks to start again.
constexpr int kExitWindowResized = 3;

// Runs the live overlay until the window closes or Ctrl+Alt+Esc. Returns a
// process exit code.
int runRealtimeSession(const RealtimeOptions& options);

// Runs the real-time GPU pipeline on two PNGs (verification / tuning).
// previousPath, when given, is the frame before frame0: the pair
// (previous, frame0) is submitted first so the timed pair runs with the
// cross-pair state primed (FFX's history, the static-UI evidence), as in a
// live stream.
int runOfflineGpuInterpolation(const std::string& frame0Path, const std::string& frame1Path,
                               const std::string& outputPath, const std::string& flowOutputPath,
                               const std::string& groundTruthPath, float t,
                               const FlowSettings& flow, GraphicsAdapterPreference adapter,
                               const std::string& previousPath = std::string());

// Bounded self-test of the native pipeline without capture.
int runD3D12SelfTest(GraphicsAdapterPreference adapter);
// GPU regression test for projection packing and lookup from 1440p through 8K.
int runProjectionSelfTest(GraphicsAdapterPreference adapter);
// Exact GPU blend-mask checks; also checks network/ring lifetime if a model is installed.
int runAiBlendMaskSelfTest(GraphicsAdapterPreference adapter);
int runAiRefinementSelfTest(GraphicsAdapterPreference adapter);
int runTemporalUiSelfTest(GraphicsAdapterPreference adapter);
int runUiRegionsSelfTest(GraphicsAdapterPreference adapter);
int runUiDetectTest(const std::string& input, const std::string& output, GraphicsAdapterPreference adapter,
                    UiDetector::Choice choice = UiDetector::Choice::Automatic);
// AI depth on one image (DepthEstimator): average time over 20 runs, and the
// nearness map written to `output` as an 8-bit PNG (white = near).
int runDepthTest(const std::string& input, const std::string& output, GraphicsAdapterPreference adapter);

// Experimental: the frame between frame0 and frame1 from AMD FSR frame
// generation fed with this program's optical flow (see FsrFrameGen.h).
int runFsrFrameGenOffline(const std::string& frame0Path, const std::string& frame1Path,
                          const std::string& outputPath, const std::string& groundTruthPath,
                          const FlowSettings& settings, GraphicsAdapterPreference adapter,
                          const std::string& previousPath, float depthValue, uint32_t generatedFrames = 1);
// The live FSR path (FsrRealtime) on saved frames in order: pairs i, i+2 against
// the true middle i+1, two sequences (from frame 0 and frame 1).
int runFsrRealtimeTest(const std::vector<std::string>& framePaths, const FlowSettings& settings,
                       GraphicsAdapterPreference adapter, float depthSpan = 0.0f,
                       const std::string& saveFolder = std::string(),
                       const FsrCameraSettings& camera = FsrCameraSettings());
// Times the AI pyramid flow network (tools/ai-train's ONNX export) alone on
// the GPU: input and outputs in D3D12 buffers, one fence wait per run.
int runAiFlowTest(const std::string& model, GraphicsAdapterPreference adapter);

// Replays <prefix>_f000.png, _f001.png, ... as consecutive pairs through one
// engine, carrying cross-pair state (FFX's history, the static-UI evidence)
// the way the live overlay does. Reports PSNR per step against
// <prefix>_m000.png and so on. Every other offline path measures a single
// pair, which cannot see anything that accumulates across frames.
int runOfflineGpuSequence(const std::string& prefix, const FlowSettings& settings,
                          GraphicsAdapterPreference adapter);

// Scores the forward field at every pyramid level, raw and filled, against
// the known motion in <prefix>_flow.bin for each <prefix>_a.png/_b.png pair.
int runFlowLevelEval(const std::vector<std::string>& prefixes, const FlowSettings& settings,
                     GraphicsAdapterPreference adapter);

// Checks that the shaders compute the same function the model was trained
// against, on one frame pair. Reads the model from MOTION_ENHANCER_NEURAL_MODEL.
int runNeuralParityCheck(const std::string& frame0Path, const std::string& frame1Path,
                         GraphicsAdapterPreference adapter);
