#include "OrtRuntime.h"

#ifdef MOTION_ENHANCER_AI_DEPTH
#include <iostream>
#include <memory>
#include <mutex>

namespace ortruntime {
namespace {
std::mutex g_mutex;
bool g_loaded = false;
HMODULE g_directml = nullptr;
std::unique_ptr<Ort::Env> g_env;
}  // namespace

std::wstring executableDirectory() {
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring directory(path);
    const size_t slash = directory.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : directory.substr(0, slash);
}

bool load(const char* who) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded) return true;
    const std::wstring directory = executableDirectory();
    // Windows ships an older onnxruntime.dll of its own: load ours by path.
    g_directml = LoadLibraryExW((directory + L"\\DirectML.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE ort = LoadLibraryExW((directory + L"\\onnxruntime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_directml || !ort) {
        std::cerr << who << ": DirectML.dll or onnxruntime.dll missing next to the executable.\n";
        return false;
    }
    try {
        Ort::InitApi();
        g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MotionEnhancer");
    } catch (const Ort::Exception& e) {
        std::cerr << who << ": " << e.what() << "\n";
        return false;
    }
    g_loaded = true;
    return true;
}

Ort::Env& env() { return *g_env; }

HMODULE directml() { return g_directml; }
}  // namespace ortruntime
#endif
