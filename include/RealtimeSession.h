#pragma once

#include "FlowEngine.h"
#include "GraphicsDevice.h"

#include <cstdint>
#include <string>
#include <vector>

struct RealtimeOptions {
    std::string windowTitle;
    std::string stopEventName;          // optional GUI-owned graceful stop event
    FlowSettings flow;
    uint32_t sourceFps = 0;          // 0 = auto-detect cadence
    uint32_t outputMultiplier = 0;   // 0 = every display refresh
    GraphicsAdapterPreference adapter = GraphicsAdapterPreference::Auto;
};

// Runs the live overlay until the window closes or Ctrl+Alt+Esc. Returns a
// process exit code.
int runRealtimeSession(const RealtimeOptions& options);

// Runs the real-time GPU pipeline on two PNGs (verification / tuning).
// previousPath, when given, is the frame before frame0: the pair
// (previous, frame0) is submitted first so the timed pair runs with the
// temporal predictor chain primed, as in a live stream.
int runOfflineGpuInterpolation(const std::string& frame0Path, const std::string& frame1Path,
                               const std::string& outputPath, const std::string& flowOutputPath,
                               const std::string& groundTruthPath, float t,
                               const FlowSettings& flow, GraphicsAdapterPreference adapter,
                               const std::string& previousPath = std::string());

// Bounded self-test of the native pipeline without capture.
int runD3D12SelfTest(GraphicsAdapterPreference adapter);

// Replays <prefix>_f000.png, _f001.png, ... as consecutive pairs through one
// engine, so each estimate becomes the next pair's temporal predictor the way
// the live overlay builds it. Reports PSNR per step against <prefix>_m000.png
// and so on. Every other offline path measures a single pair with the chain
// unbound, which cannot see anything that accumulates across frames.
int runOfflineGpuSequence(const std::string& prefix, const FlowSettings& settings,
                          GraphicsAdapterPreference adapter);

// Checks that the shaders compute the same function the model was trained
// against, on one frame pair. Reads the model from MOTION_ENHANCER_NEURAL_MODEL.
int runNeuralParityCheck(const std::string& frame0Path, const std::string& frame1Path,
                         GraphicsAdapterPreference adapter);
