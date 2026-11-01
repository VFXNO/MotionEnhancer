#pragma once

#include "GraphicsDevice.h"
#include <memory>
#include <string>

// Optional, UI-only Vulkan backend. Every accepted inference must dispatch
// cooperative-matrix shaders with AMD v_wmma ISA, not just request GPU use.
class UiMatrixDetector {
public:
    UiMatrixDetector();
    ~UiMatrixDetector();
    bool initialize(IDXGIAdapter1* adapter, ID3D12Device* d3d, const std::wstring& paramPath);
    ID3D12Resource* inputResource() const;
    // Caller waits for the D3D12 producer on its worker before submitting.
    bool infer(float* proposals);
    unsigned matrixDispatches() const;
    unsigned verifiedWmmaDispatches() const;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
