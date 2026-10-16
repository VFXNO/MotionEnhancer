#pragma once

// The one ONNX Runtime environment and DirectML library of the process,
// shared by the AI depth (DepthEstimator) and the AI flow (AiFlowEstimator):
// ONNX Runtime allows a single default-logging environment per process.
// Only with MOTION_ENHANCER_AI_DEPTH (the build that ships onnxruntime.dll,
// delay-loaded, and DirectML.dll next to the executable).

#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifdef MOTION_ENHANCER_AI_DEPTH
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

namespace ortruntime {
// Loads DirectML.dll and onnxruntime.dll from the executable's folder and
// initialises the C++ API, once. False (reason on stderr, prefixed with
// `who`) when either is missing.
bool load(const char* who);
// The process's environment (after load()).
Ort::Env& env();
// DirectML.dll (after load()), for DMLCreateDevice.
HMODULE directml();
// The folder holding the executable.
std::wstring executableDirectory();
}  // namespace ortruntime
#endif
