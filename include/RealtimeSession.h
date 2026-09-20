#pragma once

#include "FlowEngine.h"
#include "GraphicsDevice.h"

#include <cstdint>
#include <string>

struct RealtimeOptions {
    std::string windowTitle;
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
