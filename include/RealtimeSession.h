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
int runUiRegionsSelfTest(GraphicsAdapterPreference adapter);
int runUiDetectTest(const std::string& input, const std::string& output, GraphicsAdapterPreference adapter,
                    UiDetector::Choice choice = UiDetector::Choice::Automatic);
// AI depth on one image (DepthEstimator): average time over 20 runs, and the
// nearness map written to `output` as an 8-bit PNG (white = near).
int runDepthTest(const std::string& input, const std::string& output, GraphicsAdapterPreference adapter);
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
