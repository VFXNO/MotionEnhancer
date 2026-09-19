#include "Image.h"
#include "Pyramid.h"
#include "MotionVector.h"
#include "ZNCCMatcher.h"
#include "FrameInterpolator.h"
#include "D3D11Context.h"
#include "WGCCapture.h"
#include "GPUInterpolator.h"
#include "WindowHelper.h"
#include "GuiApp.h"
#include "RealtimePresenter.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

static void printUsage(const char* progName) {
    std::cout << "========================================================================\n"
              << " Enhanced 8-Level Pyramid ZNCC Block-Matching Frame Interpolator\n"
              << " GPU Shader-Based Real-Time WGC Window Overlay & Offline Processor\n"
              << "========================================================================\n"
              << "Usage:\n"
              << "  " << progName << "                                      # Open graphical interface\n"
              << "  " << progName << " --gui                                # Open graphical interface\n"
              << "  " << progName << " --capture-window <title> [options]   # Real-time GPU overlay\n"
              << "  " << progName << " --list-windows                      # List desktop windows for capture\n"
              << "  " << progName << " <frame0.png> <frame1.png> <out.png> # Offline 2-frame interpolation\n"
              << "  " << progName << " --gpu-offline <f0.png> <f1.png> <out.png> [--ground-truth gt.png] [--save-flow f.png]\n"
              << "                                        # Run the real-time GPU shader pipeline on two PNGs\n"
              << "  " << progName << " --test [options]                    # Synthetic verification benchmark\n\n"
              << "Real-Time GPU Overlay Options:\n"
              << "  --capture-window <title> Target window title substring to capture via WGC\n"
              << "  --source-fps <auto|24|30|60>  Source content rate (default: auto)\n"
              << "  --multiplier <2|3|4|max>      Output FPS multiplier (default: 2)\n"
              << "  --gpu-levels <1-8>       Real-time pyramid levels (default: 7)\n"
              << "  --gpu-min-refine <n>     Finest searched level, 0 to levels-1 (default: 0)\n"
              << "  --gpu-coarse-radius <0-8>   Coarse search radius (default: 8)\n"
              << "  --gpu-refine-radius <0-8>   Refinement radius (default: 2)\n"
              << "  --gpu-smoothness <0-0.1>    Predictor smoothness weight (default: 0.0005)\n"
              << "  --list-windows           List all active top-level desktop windows\n\n"
              << "Offline / Algorithm Options:\n"
              << "  --time <float>          Intermediate timestamp in (0.0, 1.0) (default: 0.5)\n"
              << "  --levels <int>          Number of pyramid levels (default: 8)\n"
              << "  --block-size <int>      Nominal block matching window size (default: 8)\n"
              << "  --search-radius <int>   Refinement search radius per level (default: 4)\n"
              << "  --coarse-radius <int>   Coarsest level search radius (default: 8)\n"
              << "  --grid-step <int>       Grid step: 1 = dense, 2 = 2x2 grid (default: 2)\n"
              << "  --subpel <mode>         Subpixel mode: none, parabolic, half, quarter (default: quarter)\n"
              << "  --smoothness <float>    Predictor smoothness weight in [0, 0.1]\n"
              << "  --occlusion-threshold <float> Forward/backward consistency threshold\n"
              << "  --photo-threshold <float>     Photometric gate threshold in [0, 255]\n"
              << "  --photo-sigma <float>         Photometric gate softness in [0.01, 255]\n"
              << "  --no-spatial-pred       Disable multi-candidate spatial predictor evaluation (EPZS)\n"
              << "  --no-gaussian-weight    Disable 2D Gaussian block window weighting\n"
              << "  --no-adaptive-block     Disable pyramid level-adaptive block sizing\n"
              << "  --min-block-size <int>  Min block size at fine levels (default: 6)\n"
              << "  --max-block-size <int>  Max block size at coarse levels (default: 14)\n"
              << "  --no-bidirectional      Disable backward flow & occlusion consistency\n"
              << "  --no-photo-gate         Disable photometric-aware occlusion gating\n"
              << "  --no-hole-inpaint       Disable bilateral disocclusion hole inpainting\n"
              << "  --no-detail-restore     Disable high-frequency detail restoration\n"
              << "  --no-color-clamp        Disable local color bounding box clamping\n"
              << "  --detail-strength <flt> Detail restoration strength in [0.0, 1.0] (default: 0.65)\n"
              << "  --save-flow <path>      Save color-coded forward flow field to PNG\n"
              << "  --save-occ <path>       Save occlusion mask image to PNG\n"
              << "  --test                  Run synthetic verification benchmark with ground truth\n"
              << "========================================================================\n";
}

static double computePSNR(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        return 0.0;
    }
    double mse = 0.0;
    size_t total = static_cast<size_t>(a.width) * a.height * a.channels;
    for (size_t i = 0; i < total; ++i) {
        double diff = static_cast<double>(a.data[i]) - static_cast<double>(b.data[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(total);
    if (mse < 1e-10) return 99.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// Generate a synthetic test frame with textured background and a translated textured circle
static Image generateSyntheticFrame(int w, int h, float shiftX, float shiftY) {
    Image img(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float bgR = 128.0f + 60.0f * std::sin(x * 0.05f) * std::cos(y * 0.05f);
            float bgG = 128.0f + 60.0f * std::cos(x * 0.04f + y * 0.04f);
            float bgB = 160.0f + 50.0f * std::sin((x + y) * 0.03f);

            float cx = static_cast<float>(w) / 2.0f + shiftX;
            float cy = static_cast<float>(h) / 2.0f + shiftY;
            float radius = 45.0f;

            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < radius) {
                float localX = dx;
                float localY = dy;
                float fgR = 230.0f + 25.0f * std::sin(localX * 0.3f);
                float fgG = 60.0f + 50.0f * std::cos(localY * 0.3f);
                float fgB = 40.0f + 30.0f * std::sin((localX + localY) * 0.2f);

                float alpha = std::clamp(radius - dist, 0.0f, 1.0f);
                img.set(x, y, 0, fgR * alpha + bgR * (1.0f - alpha));
                img.set(x, y, 1, fgG * alpha + bgG * (1.0f - alpha));
                img.set(x, y, 2, fgB * alpha + bgB * (1.0f - alpha));
            } else {
                img.set(x, y, 0, bgR);
                img.set(x, y, 1, bgG);
                img.set(x, y, 2, bgB);
            }
        }
    }
    return img;
}

static int runSyntheticTest(const InterpolatorParams& userParams) {
    std::cout << "\n>>> Starting Synthetic Benchmark (Enhanced 8-Level Pyramid ZNCC)...\n";
    const int W = 320;
    const int H = 240;
    const float motionX = 16.0f;
    const float motionY = 8.0f;

    std::cout << "Creating synthetic frames (" << W << "x" << H << ")...\n";
    std::cout << "Object motion from Frame 0 to Frame 1: dx = +" << motionX << " px, dy = +" << motionY << " px\n";

    Image frame0 = generateSyntheticFrame(W, H, -motionX / 2.0f, -motionY / 2.0f);
    Image frame1 = generateSyntheticFrame(W, H, +motionX / 2.0f, +motionY / 2.0f);
    Image groundTruth = generateSyntheticFrame(W, H, 0.0f, 0.0f);

    frame0.savePNG("test_frame0.png");
    frame1.savePNG("test_frame1.png");
    groundTruth.savePNG("test_groundtruth.png");
    std::cout << "Saved test_frame0.png, test_frame1.png, test_groundtruth.png\n";

    InterpolatorParams params = userParams;
    FrameInterpolator interpolator(params);
    FlowField forwardFlow, backwardFlow;
    Image occMask;

    std::cout << "Interpolating intermediate frame at t = 0.5 using " << params.matcherParams.pyramidLevels << " pyramid levels...\n";
    auto start = std::chrono::high_resolution_clock::now();

    Image interpolated = interpolator.interpolate(frame0, frame1, 0.5f, &forwardFlow, &backwardFlow, &occMask);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    interpolated.savePNG("test_interpolated.png");
    Image flowVis = forwardFlow.toColorImage();
    flowVis.savePNG("test_flow.png");

    Image occVis(W, H, 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v = occMask.get(x, y, 0) * 255.0f;
            occVis.set(x, y, 0, v);
            occVis.set(x, y, 1, v);
            occVis.set(x, y, 2, v);
        }
    }
    occVis.savePNG("test_occlusion.png");

    Image linearBlend(W, H, 3);
    for (size_t i = 0; i < linearBlend.data.size(); ++i) {
        linearBlend.data[i] = 0.5f * frame0.data[i] + 0.5f * frame1.data[i];
    }
    double naivePsnr = computePSNR(linearBlend, groundTruth);
    double psnr = computePSNR(interpolated, groundTruth);
    MotionVector centerMv = forwardFlow.get(W / 2, H / 2);
    float errX = std::abs(centerMv.vx - motionX);
    float errY = std::abs(centerMv.vy - motionY);

    std::cout << "\n>>> Benchmark Results:\n"
              << "  Execution Time:          " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n"
              << "  Object Motion Truth:     vx = +" << motionX << " px, vy = +" << motionY << " px\n"
              << "  Estimated Motion Center: vx = " << centerMv.vx << " px, vy = " << centerMv.vy
              << " px (ZNCC: " << centerMv.score << ")\n"
              << "  Subpixel Motion Error:   dx = " << errX << " px, dy = " << errY << " px\n"
              << "  Uncompensated PSNR:      " << std::fixed << std::setprecision(2) << naivePsnr << " dB (naive blend)\n"
              << "  Motion Compensated PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB (+ "
              << std::fixed << std::setprecision(2) << (psnr - naivePsnr) << " dB improvement!)\n"
              << "  Saved Outputs:           test_interpolated.png, test_flow.png, test_occlusion.png\n";

    if (errX < 0.5f && errY < 0.5f) {
        std::cout << ">>> TEST PASSED (Motion vector accurately captured within < 0.5 px tolerance!)\n\n";
        return 0;
    } else {
        std::cout << ">>> Warning: Motion estimation error exceeds 0.5 px tolerance\n\n";
        return 1;
    }
}

// ---------------------------------------------------------------------------
// Offline GPU path: runs the exact real-time shader pipeline on two PNGs so
// the HLSL can be checked against a known image pair / ground truth.
// ---------------------------------------------------------------------------
static bool uploadFrameTexture(D3D11Context& ctx, const Image& img, ComPtr<ID3D11Texture2D>& texture) {
    std::vector<uint8_t> pixels(static_cast<size_t>(img.width) * img.height * 4);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            size_t o = (static_cast<size_t>(y) * img.width + x) * 4;
            for (int c = 0; c < 3; ++c) {
                float v = img.channels >= 3 ? img.get(x, y, c) : img.get(x, y, 0);
                pixels[o + c] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            }
            pixels[o + 3] = 255;
        }
    }
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = static_cast<UINT>(img.width);
    desc.Height = static_cast<UINT>(img.height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA init = { pixels.data(), static_cast<UINT>(img.width * 4), 0 };
    return SUCCEEDED(ctx.device->CreateTexture2D(&desc, &init, texture.GetAddressOf()));
}

static bool readbackTexture(D3D11Context& ctx, ID3D11Texture2D* source, std::vector<uint8_t>& bytes,
                            UINT& width, UINT& height, UINT& rowPitch) {
    D3D11_TEXTURE2D_DESC desc = {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(ctx.device->CreateTexture2D(&desc, nullptr, staging.GetAddressOf()))) return false;
    ctx.context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx.context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;
    width = desc.Width;
    height = desc.Height;
    rowPitch = mapped.RowPitch;
    bytes.assign(static_cast<const uint8_t*>(mapped.pData),
                 static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(mapped.RowPitch) * desc.Height);
    ctx.context->Unmap(staging.Get(), 0);
    return true;
}

static FlowField readbackFlowGrid(D3D11Context& ctx, ID3D11Texture2D* flowTexture, int imageW, int imageH) {
    std::vector<uint8_t> bytes;
    UINT w = 0, h = 0, pitch = 0;
    if (!flowTexture || !readbackTexture(ctx, flowTexture, bytes, w, h, pitch)) return FlowField();
    // Expand the per-block grid to a per-pixel field so FlowField::toColorImage
    // and the centre-vector check work as they do for the CPU path.
    FlowField field(imageW, imageH);
    for (int y = 0; y < imageH; ++y) {
        for (int x = 0; x < imageW; ++x) {
            UINT bx = std::min<UINT>(static_cast<UINT>(x / 16), w - 1);
            UINT by = std::min<UINT>(static_cast<UINT>(y / 16), h - 1);
            const float* v = reinterpret_cast<const float*>(bytes.data() + static_cast<size_t>(by) * pitch + bx * 8);
            field.set(x, y, MotionVector(v[0], v[1], 1.0f));
        }
    }
    return field;
}

static int runOfflineGPUInterpolation(
    const std::string& input0Path,
    const std::string& input1Path,
    const std::string& outputPath,
    const std::string& flowOutputPath,
    const std::string& groundTruthPath,
    float t,
    const GPUInterpolationSettings& gpuSettings
) {
    Image frame0 = Image::load(input0Path);
    Image frame1 = Image::load(input1Path);
    if (frame0.empty() || frame1.empty()) return 1;
    if (frame0.width != frame1.width || frame0.height != frame1.height) {
        std::cerr << "Error: frame dimensions mismatch\n";
        return 1;
    }

    auto d3dContext = std::make_shared<D3D11Context>();
    if (!d3dContext->initialize()) return 1;

    GPUInterpolator gpu;
    gpu.setSettings(gpuSettings);
    if (!gpu.initialize(d3dContext)) return 1;
    if (!gpu.resizeBuffers(static_cast<uint32_t>(frame0.width), static_cast<uint32_t>(frame0.height))) return 1;

    const uint32_t flowGridWidth = (static_cast<uint32_t>(frame0.width) + 15) / 16;
    const uint32_t flowGridHeight = (static_cast<uint32_t>(frame0.height) + 15) / 16;
    int debugLevels = std::clamp(gpuSettings.pyramidLevels, 1, GPUInterpolator::MAX_PYRAMID_LEVELS);
    while (debugLevels > 1 &&
           std::min(frame0.width, frame0.height) >> (debugLevels - 1) < GPUInterpolator::kMinCoarsestExtent) {
        --debugLevels;
    }
    std::cout << "GPU block matching debug:\n"
              << "  Block size:             16x16 pixels\n"
              << "  Matching support:       16x16 pixels\n"
              << "  Flow grid (full):       " << flowGridWidth << "x" << flowGridHeight
              << " vectors (" << static_cast<uint64_t>(flowGridWidth) * flowGridHeight << ")\n"
              << "  Directions:             forward + backward\n"
              << "  Pyramid levels searched: " << debugLevels << "\n"
              << "  Coarse search radius:   " << gpuSettings.coarseSearchRadius << "\n"
              << "  Refine search radius:   " << gpuSettings.refineSearchRadius << "\n";
    for (int level = 0; level < debugLevels; ++level) {
        uint32_t levelWidth = std::max(1u, static_cast<uint32_t>(frame0.width) >> level);
        uint32_t levelHeight = std::max(1u, static_cast<uint32_t>(frame0.height) >> level);
        uint32_t levelFlowWidth = (levelWidth + 15) / 16;
        uint32_t levelFlowHeight = (levelHeight + 15) / 16;
        std::cout << "  L" << level << ": " << levelWidth << "x" << levelHeight
                  << " image, flow " << levelFlowWidth << "x" << levelFlowHeight << "\n";
    }

    ComPtr<ID3D11Texture2D> tex0, tex1, outTex;
    if (!uploadFrameTexture(*d3dContext, frame0, tex0) || !uploadFrameTexture(*d3dContext, frame1, tex1)) {
        std::cerr << "Error: failed to upload frames\n";
        return 1;
    }
    {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = static_cast<UINT>(frame0.width);
        desc.Height = static_cast<UINT>(frame0.height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        if (FAILED(d3dContext->device->CreateTexture2D(&desc, nullptr, outTex.GetAddressOf()))) return 1;
    }

    // Warm-up pass so the GPU timestamp query of a steady-state run can be
    // read back by the second call (the query resolves one call later).
    if (!gpu.prepareFramePair(tex0.Get(), tex1.Get(), 0, 1)) {
        std::cerr << "Error: prepareFramePair failed\n";
        return 1;
    }
    d3dContext->context->Flush();
    auto start = std::chrono::high_resolution_clock::now();
    if (!gpu.prepareFramePair(tex0.Get(), tex1.Get(), 1, 2)) {
        std::cerr << "Error: prepareFramePair failed\n";
        return 1;
    }
    if (!gpu.synthesize(outTex.Get(), t)) {
        std::cerr << "Error: synthesize failed\n";
        return 1;
    }
    d3dContext->context->Flush();

    std::vector<uint8_t> bytes;
    UINT w = 0, h = 0, pitch = 0;
    if (!readbackTexture(*d3dContext, outTex.Get(), bytes, w, h, pitch)) {
        std::cerr << "Error: readback failed\n";
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    Image output(static_cast<int>(w), static_cast<int>(h), 3);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* row = bytes.data() + static_cast<size_t>(y) * pitch;
        for (UINT x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                output.set(static_cast<int>(x), static_cast<int>(y), c, static_cast<float>(row[x * 4 + c]));
            }
        }
    }
    if (!output.savePNG(outputPath)) {
        std::cerr << "Failed to save " << outputPath << "\n";
        return 1;
    }
    // Third call only to resolve the timestamp query of the timed run.
    gpu.prepareFramePair(tex0.Get(), tex1.Get(), 2, 3);
    std::cout << "GPU offline interpolation (t = " << t << "): flow pipeline "
              << std::fixed << std::setprecision(2) << gpu.getLastGpuTimeMs() << " ms GPU, "
              << elapsedMs << " ms wall incl. readback -> " << outputPath << "\n";

    FlowField forward = readbackFlowGrid(*d3dContext, gpu.forwardFlowTexture(), frame0.width, frame0.height);
    if (!forward.empty()) {
        MotionVector centre = forward.get(frame0.width / 2, frame0.height / 2);
        std::cout << "  Centre block vector: (" << centre.vx << ", " << centre.vy << ")\n";
        if (!flowOutputPath.empty()) {
            forward.toColorImage().savePNG(flowOutputPath);
            std::cout << "  Saved forward flow visualization to: " << flowOutputPath << "\n";
            // Also dump the raw block grid as text (one "vx,vy" per block) so
            // vectors can be inspected exactly rather than through the hue map.
            std::ofstream txt(flowOutputPath + ".txt");
            for (int by = 0; by * 16 < frame0.height; ++by) {
                for (int bx = 0; bx * 16 < frame0.width; ++bx) {
                    MotionVector v = forward.get(bx * 16, by * 16);
                    txt << static_cast<int>(v.vx) << "," << static_cast<int>(v.vy) << (bx * 16 + 16 < frame0.width ? " " : "\n");
                }
            }
        }
    }

    if (!groundTruthPath.empty()) {
        Image truth = Image::load(groundTruthPath);
        if (!truth.empty() && truth.width == output.width && truth.height == output.height) {
            Image blend(frame0.width, frame0.height, 3);
            for (int y = 0; y < blend.height; ++y)
                for (int x = 0; x < blend.width; ++x)
                    for (int c = 0; c < 3; ++c)
                        blend.set(x, y, c, (1.0f - t) * frame0.get(x, y, c) + t * frame1.get(x, y, c));
            std::cout << "  PSNR vs ground truth: " << std::fixed << std::setprecision(2)
                      << computePSNR(output, truth) << " dB (naive blend: "
                      << computePSNR(blend, truth) << " dB)\n";
        }
    }
    return 0;
}

static int runRealtimeGPUInterpolation(
    const std::string& windowTitle,
    const GPUInterpolationSettings& gpuSettings,
    uint32_t sourceFps,
    uint32_t outputMultiplier
) {
    int exitCode = 0;
    std::thread renderThread([&]() {
        WindowHelper::attachInteractiveDesktop();

        std::cout << "\n============================================================\n"
                  << " Initializing Real-Time GPU Shader-Based Frame Interpolation\n"
                  << "============================================================\n";

        HWND targetHwnd = nullptr;
        if (!windowTitle.empty()) {
            targetHwnd = WindowHelper::findWindowByTitle(windowTitle);
            if (!targetHwnd) {
                std::cerr << "Error: Target window matching \"" << windowTitle << "\" not found!\n";
                WindowHelper::printWindowList();
                exitCode = 1;
                return;
            }
        } else {
            std::cout << "No window title specified. Listing top-level visible windows:\n";
            WindowHelper::printWindowList();
            std::cout << "Usage: motion_enhancer.exe --capture-window <window_title>\n";
            exitCode = 0;
            return;
        }

        char titleBuf[256] = {};
        GetWindowTextA(targetHwnd, titleBuf, sizeof(titleBuf));
        std::cout << "Target window: \"" << titleBuf << "\" (HWND: 0x" << std::hex << (uintptr_t)targetHwnd << std::dec << ")\n" << std::flush;

    // 1. Initialize Direct3D 11 Context
    std::cout << "[1/5] Initializing Direct3D 11 device and context...\n" << std::flush;
    auto d3dContext = std::make_shared<D3D11Context>();
    if (!d3dContext->initialize()) {
        std::cerr << "Error: Failed to initialize Direct3D 11 hardware device.\n";
        exitCode = 1;
        return;
    }
    std::cout << "      Direct3D 11 hardware context ready.\n" << std::flush;

    // 2. Create Overlay Window
    std::cout << "[2/5] Creating click-through tracking presentation window...\n" << std::flush;
    OverlayWindow overlay;
    if (!overlay.create(targetHwnd, "Motion Enhancer GPU Overlay")) {
        exitCode = 1;
        return;
    }

    // 3. Initialize Windows Graphics Capture (WGC)
    std::cout << "[3/5] Starting Windows Graphics Capture (WGC API) session...\n" << std::flush;
    WGCCapture capture;
    if (!capture.initialize(d3dContext->device.Get())) {
        exitCode = 1;
        return;
    }

    if (!capture.startCapture(targetHwnd)) {
        exitCode = 1;
        return;
    }

    // 4. Match the back buffer to the WGC texture. DXGI scales it to the
    // overlay client area during presentation when window borders differ.
    std::cout << "[4/5] Initializing hardware overlay presentation surface ("
              << capture.getWidth() << "x" << capture.getHeight() << ")...\n" << std::flush;
    RealtimePresenter presenter;
    if (!presenter.initialize(
            d3dContext, overlay.hwnd, capture.getWidth(), capture.getHeight(),
            sourceFps, outputMultiplier)) {
        exitCode = 1;
        return;
    }

    // 5. Initialize GPU Shader Interpolator (compiles HLSL compute shaders)
    std::cout << "[5/5] Compiling and loading HLSL Compute Shaders...\n" << std::flush;
    GPUInterpolator gpuInterpolator;
    gpuInterpolator.setSettings(gpuSettings);
    if (!gpuInterpolator.initialize(d3dContext)) {
        exitCode = 1;
        return;
    }
    if (!gpuInterpolator.resizeBuffers(capture.getWidth(), capture.getHeight())) {
        exitCode = 1;
        return;
    }

    std::ofstream gpuDebug("gpu_debug.txt", std::ios::trunc);
    if (gpuDebug) {
        uint32_t flowWidth = (capture.getWidth() + 15) / 16;
        uint32_t flowHeight = (capture.getHeight() + 15) / 16;
        gpuDebug << "GPU block matching debug\n"
                 << "Block size: 16x16 pixels\n"
                 << "Matching support: 16x16 pixels\n"
                 << "Flow format: R32G32_FLOAT\n"
                 << "Flow grid: " << flowWidth << "x" << flowHeight
                 << " vectors (" << static_cast<uint64_t>(flowWidth) * flowHeight << ")\n"
                 << "Directions: forward + backward\n"
                 << "Pyramid levels requested: " << gpuSettings.pyramidLevels << "\n"
                 << "Coarse search radius: " << gpuSettings.coarseSearchRadius << "\n"
                 << "Refine search radius: " << gpuSettings.refineSearchRadius << "\n"
                 << "Capture size: " << capture.getWidth() << "x" << capture.getHeight() << "\n";
        gpuDebug.flush();
    }

    std::cout << "\n============================================================\n"
              << " >>> Real-Time GPU Frame Interpolation Active!\n"
              << "  Capture Engine: Windows Graphics Capture (WGC API - Zero-Copy GPU VRAM)\n"
              << "  Source Window:  \"" << titleBuf << "\" (" << capture.getWidth() << "x" << capture.getHeight() << ")\n"
              << "  GPU Settings:   " << gpuSettings.pyramidLevels << " levels, refine to L"
              << gpuSettings.minRefineLevel << ", radii " << gpuSettings.coarseSearchRadius
              << "/" << gpuSettings.refineSearchRadius << ", smoothness "
              << gpuSettings.smoothnessWeight << "\n"
              << "  Source Cadence: "
              << (sourceFps == 0 ? "auto (detecting)" : std::to_string(sourceFps) + " FPS") << "\n"
              << "  FPS Multiplier: " << (outputMultiplier == 0 ? "display max" : std::to_string(outputMultiplier) + "x") << "\n"
              << "  Output Clock:   " << std::fixed << std::setprecision(2)
              << presenter.outputRate() << " Hz (display " << presenter.refreshRate() << " Hz)\n"
              << "  Compute Engine: FidelityFX-style 16x16 Block MSAD64 Optical Flow (DirectCompute)\n"
              << "  Flow Grid:      " << ((capture.getWidth() + 15) / 16) << "x"
              << ((capture.getHeight() + 15) / 16) << " vectors\n"
              << "  Presentation:   Paced Click-Through DWM Presenter\n"
              << "  Hotkeys:        [Ctrl+Alt+F1] Toggle | [Ctrl+Alt+Esc] Exit\n"
              << "============================================================\n\n" << std::flush;

    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    uint64_t lastPresentedIndex = 0;
    double totalPresentMs = 0.0;
    double totalTrackMs = 0.0;
    double totalMsgMs = 0.0;
    double totalUpdateMs = 0.0;
    HANDLE frameEvent = capture.getFrameEvent();
    HANDLE presentationEvent = presenter.frameLatencyHandle();
    HANDLE pacingEvent = presenter.pacingHandle();
    HANDLE waitHandles[] = { frameEvent, presentationEvent, pacingEvent };
    bool latencyReady = false;
    bool pacingReady = false;
    if (!frameEvent || !presentationEvent || !pacingEvent) {
        std::cerr << "Error: Required capture or presentation synchronization handle is unavailable.\n";
        exitCode = 1;
        return;
    }

    while (true) {
        DWORD waitResult = MsgWaitForMultipleObjectsEx(
            3,
            waitHandles,
            INFINITE,
            QS_ALLINPUT,
            MWMO_INPUTAVAILABLE
        );

        auto tm0 = std::chrono::high_resolution_clock::now();
        if (!overlay.processMessages()) break;
        auto tm1 = std::chrono::high_resolution_clock::now();
        totalMsgMs += std::chrono::duration<double, std::milli>(tm1 - tm0).count();

        auto tr0 = std::chrono::high_resolution_clock::now();
        overlay.updateTracking();
        auto tr1 = std::chrono::high_resolution_clock::now();
        totalTrackMs += std::chrono::duration<double, std::milli>(tr1 - tr0).count();

        if (waitResult == WAIT_OBJECT_0) {
            auto tu0 = std::chrono::high_resolution_clock::now();
            ID3D11Texture2D* destination = presenter.acquireCaptureTarget();
            int64_t timestamp100ns = 0;
            if (capture.copyLatestFrame(
                    d3dContext->context.Get(), destination, sourceFps, timestamp100ns)) {
                presenter.setSourceFps(sourceFps > 0 ? sourceFps : capture.getDetectedSourceFps());
                presenter.commitCapturedFrame(timestamp100ns);
            } else {
                presenter.cancelCaptureTarget();
            }
            auto tu1 = std::chrono::high_resolution_clock::now();
            totalUpdateMs += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
        }

        if (waitResult == WAIT_OBJECT_0 + 1) latencyReady = true;
        if (waitResult == WAIT_OBJECT_0 + 2) pacingReady = true;
        if (!latencyReady && WaitForSingleObject(presentationEvent, 0) == WAIT_OBJECT_0) {
            latencyReady = true;
        }
        if (!pacingReady && WaitForSingleObject(pacingEvent, 0) == WAIT_OBJECT_0) {
            pacingReady = true;
        }
        if (latencyReady && pacingReady) {
            auto frameStart = std::chrono::high_resolution_clock::now();
            if (!presenter.presentNext(gpuInterpolator)) {
                std::cerr << "Error: Realtime queued presentation failed.\n";
                break;
            }
            auto frameEnd = std::chrono::high_resolution_clock::now();
            totalPresentMs += std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
            latencyReady = false;
            pacingReady = false;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsedSec = std::chrono::duration<double>(now - lastFpsTime).count();
        if (elapsedSec >= 2.0) {
            uint64_t presented = presenter.presentedFrameIndex();
            uint64_t intervalFrames = presented - lastPresentedIndex;
            double fps = static_cast<double>(intervalFrames) / elapsedSec;
            double n = intervalFrames > 0 ? static_cast<double>(intervalFrames) : 1.0;
            std::cout << "[GPU Overlay] FPS: " << std::fixed << std::setprecision(1) << fps
                      << " fps | GPU Pipeline: " << std::setprecision(2) << gpuInterpolator.getLastGpuTimeMs() << " ms"
                      << " | Frame Slot: " << (totalPresentMs / n) << " ms"
                      << " | Queue: " << presenter.queueDepth()
                      << " | Source: "
                      << (sourceFps > 0 ? sourceFps : capture.getDetectedSourceFps()) << " fps"
                      << " | Target: " << std::setprecision(1) << presenter.outputRate() << " fps"
                      << " | Alpha: " << presenter.interpolationFactor()
                      << " | R/Q/P: " << presenter.renderedFrameIndex() << "/"
                      << presenter.queuedFrameIndex() << "/"
                      << presenter.presentedFrameIndex() << "\n" << std::flush;
            lastPresentedIndex = presented;
            totalPresentMs = 0.0;
            totalTrackMs = 0.0;
            totalMsgMs = 0.0;
            totalUpdateMs = 0.0;
            lastFpsTime = now;
        }
    }

        capture.stopCapture();
        std::cout << "Real-time GPU frame interpolation terminated.\n";
    });

    renderThread.join();
    return exitCode;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    WindowHelper::attachInteractiveDesktop();

    if (argc < 2 || (argc == 2 && std::string(argv[1]) == "--gui")) {
        return runGuiApplication();
    }

    std::string input0Path;
    std::string input1Path;
    std::string outputPath;
    std::string flowOutputPath;
    std::string occOutputPath;
    std::string captureWindowQuery;
    std::string groundTruthPath;
    bool gpuOffline = false;
    uint32_t sourceFps = 0;
    uint32_t outputMultiplier = 2;
    GPUInterpolationSettings gpuSettings;

    InterpolatorParams params;
    params.matcherParams.pyramidLevels = 8;
    params.matcherParams.blockSize = 8;
    params.matcherParams.adaptiveBlockSize = true;
    params.matcherParams.minBlockSize = 6;
    params.matcherParams.maxBlockSize = 14;
    params.matcherParams.searchRadiusCoarse = 8;
    params.matcherParams.searchRadiusRefine = 4;
    params.matcherParams.gridStep = 2;
    params.matcherParams.subpelMode = SUBPEL_QUARTER;
    params.matcherParams.useSpatialPredictors = true;
    params.matcherParams.useGaussianWeights = true;
    params.bidirectional = true;

    float t = 0.5f;
    bool runTest = false;
    bool listWindows = false;

    std::vector<std::string> positionalArgs;

    std::string currentArgument;
    try {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        currentArgument = arg;
        if (arg == "--test") {
            runTest = true;
        } else if (arg == "--list-windows") {
            listWindows = true;
        } else if (arg == "--capture-window" && i + 1 < argc) {
            captureWindowQuery = argv[++i];
        } else if (arg == "--source-fps" && i + 1 < argc) {
            std::string sourceRate = argv[++i];
            if (sourceRate == "auto") {
                sourceFps = 0;
            } else {
                try {
                    int parsedFps = std::stoi(sourceRate);
                    if (parsedFps == 24 || parsedFps == 30 || parsedFps == 60) {
                        sourceFps = static_cast<uint32_t>(parsedFps);
                    } else {
                        throw std::invalid_argument("unsupported source rate");
                    }
                } catch (const std::exception&) {
                    std::cerr << "Warning: source FPS must be auto, 24, 30, or 60; using auto\n";
                    sourceFps = 0;
                }
            }
        } else if (arg == "--gpu-offline") {
            gpuOffline = true;
        } else if (arg == "--ground-truth" && i + 1 < argc) {
            groundTruthPath = argv[++i];
        } else if (arg == "--gpu-levels" && i + 1 < argc) {
            gpuSettings.pyramidLevels = std::clamp(std::stoi(argv[++i]), 1, 8);
        } else if (arg == "--gpu-min-refine" && i + 1 < argc) {
            gpuSettings.minRefineLevel = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--gpu-coarse-radius" && i + 1 < argc) {
            gpuSettings.coarseSearchRadius = std::clamp(std::stoi(argv[++i]), 0, 8);
        } else if (arg == "--gpu-refine-radius" && i + 1 < argc) {
            gpuSettings.refineSearchRadius = std::clamp(std::stoi(argv[++i]), 0, 8);
        } else if (arg == "--gpu-smoothness" && i + 1 < argc) {
            gpuSettings.smoothnessWeight = std::clamp(std::stof(argv[++i]), 0.0f, 0.1f);
        } else if (arg == "--multiplier" && i + 1 < argc) {
            std::string multiplier = argv[++i];
            if (multiplier == "max") {
                outputMultiplier = 0;
            } else {
                if (!multiplier.empty() && multiplier.back() == 'x') multiplier.pop_back();
                int parsedMultiplier = std::stoi(multiplier);
                if (parsedMultiplier < 2 || parsedMultiplier > 4) {
                    throw std::invalid_argument("multiplier must be 2, 3, 4, or max");
                }
                outputMultiplier = static_cast<uint32_t>(parsedMultiplier);
            }
        } else if (arg == "--time" && i + 1 < argc) {
            t = std::stof(argv[++i]);
        } else if (arg == "--levels" && i + 1 < argc) {
            params.matcherParams.pyramidLevels = std::stoi(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            params.matcherParams.blockSize = std::stoi(argv[++i]);
        } else if (arg == "--search-radius" && i + 1 < argc) {
            params.matcherParams.searchRadiusRefine = std::stoi(argv[++i]);
        } else if (arg == "--coarse-radius" && i + 1 < argc) {
            params.matcherParams.searchRadiusCoarse = std::stoi(argv[++i]);
        } else if (arg == "--grid-step" && i + 1 < argc) {
            params.matcherParams.gridStep = std::stoi(argv[++i]);
        } else if (arg == "--smoothness" && i + 1 < argc) {
            params.matcherParams.smoothnessWeight = std::stof(argv[++i]);
        } else if (arg == "--occlusion-threshold" && i + 1 < argc) {
            params.occlusionThreshold = std::stof(argv[++i]);
        } else if (arg == "--photo-threshold" && i + 1 < argc) {
            params.photoGateThreshold = std::stof(argv[++i]);
        } else if (arg == "--photo-sigma" && i + 1 < argc) {
            params.photoGateSigma = std::stof(argv[++i]);
        } else if (arg == "--subpel" && i + 1 < argc) {
            std::string smode = argv[++i];
            if (smode == "none") params.matcherParams.subpelMode = SUBPEL_NONE;
            else if (smode == "parabolic") params.matcherParams.subpelMode = SUBPEL_PARABOLIC;
            else if (smode == "half") params.matcherParams.subpelMode = SUBPEL_HALF;
            else if (smode == "quarter") params.matcherParams.subpelMode = SUBPEL_QUARTER;
            else std::cerr << "Warning: unknown subpel mode '" << smode << "', using quarter\n";
        } else if (arg == "--no-spatial-pred") {
            params.matcherParams.useSpatialPredictors = false;
        } else if (arg == "--no-gaussian-weight") {
            params.matcherParams.useGaussianWeights = false;
        } else if (arg == "--no-adaptive-block") {
            params.matcherParams.adaptiveBlockSize = false;
        } else if (arg == "--min-block-size" && i + 1 < argc) {
            params.matcherParams.minBlockSize = std::stoi(argv[++i]);
        } else if (arg == "--max-block-size" && i + 1 < argc) {
            params.matcherParams.maxBlockSize = std::stoi(argv[++i]);
        } else if (arg == "--no-bidirectional") {
            params.bidirectional = false;
        } else if (arg == "--no-photo-gate") {
            params.enablePhotometricGating = false;
        } else if (arg == "--no-hole-inpaint") {
            params.enableHoleInpainting = false;
        } else if (arg == "--no-detail-restore") {
            params.enableDetailRestoration = false;
        } else if (arg == "--no-color-clamp") {
            params.enableColorClamping = false;
        } else if (arg == "--adaptive-block" && i + 1 < argc) {
            params.matcherParams.adaptiveBlockSize = std::stoi(argv[++i]) != 0;
        } else if (arg == "--spatial-pred" && i + 1 < argc) {
            params.matcherParams.useSpatialPredictors = std::stoi(argv[++i]) != 0;
        } else if (arg == "--gaussian-weight" && i + 1 < argc) {
            params.matcherParams.useGaussianWeights = std::stoi(argv[++i]) != 0;
        } else if (arg == "--bidirectional" && i + 1 < argc) {
            params.bidirectional = std::stoi(argv[++i]) != 0;
        } else if (arg == "--photo-gate" && i + 1 < argc) {
            params.enablePhotometricGating = std::stoi(argv[++i]) != 0;
        } else if (arg == "--hole-inpaint" && i + 1 < argc) {
            params.enableHoleInpainting = std::stoi(argv[++i]) != 0;
        } else if (arg == "--detail-restore" && i + 1 < argc) {
            params.enableDetailRestoration = std::stoi(argv[++i]) != 0;
        } else if (arg == "--color-clamp" && i + 1 < argc) {
            params.enableColorClamping = std::stoi(argv[++i]) != 0;
        } else if (arg == "--detail-strength" && i + 1 < argc) {
            params.detailStrength = std::stof(argv[++i]);
        } else if (arg == "--save-flow" && i + 1 < argc) {
            flowOutputPath = argv[++i];
        } else if (arg == "--save-occ" && i + 1 < argc) {
            occOutputPath = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg.rfind("--", 0) != 0) {
            positionalArgs.push_back(arg);
        }
    }
    } catch (const std::exception& error) {
        std::cerr << "Error: invalid value for " << currentArgument << ": " << error.what() << "\n";
        return 1;
    }

    params.matcherParams.pyramidLevels = std::clamp(params.matcherParams.pyramidLevels, 1, 8);
    params.matcherParams.blockSize = std::clamp(params.matcherParams.blockSize, 1, 64);
    params.matcherParams.minBlockSize = std::clamp(params.matcherParams.minBlockSize, 1, 64);
    params.matcherParams.maxBlockSize = std::clamp(params.matcherParams.maxBlockSize, 1, 64);
    if (params.matcherParams.minBlockSize > params.matcherParams.maxBlockSize) {
        std::swap(params.matcherParams.minBlockSize, params.matcherParams.maxBlockSize);
    }
    params.matcherParams.searchRadiusCoarse = std::clamp(params.matcherParams.searchRadiusCoarse, 0, 64);
    params.matcherParams.searchRadiusRefine = std::clamp(params.matcherParams.searchRadiusRefine, 0, 16);
    params.matcherParams.gridStep = std::clamp(params.matcherParams.gridStep, 1, 8);
    params.matcherParams.smoothnessWeight = std::clamp(params.matcherParams.smoothnessWeight, 0.0f, 0.1f);
    params.occlusionThreshold = std::clamp(params.occlusionThreshold, 0.0f, 100.0f);
    params.detailStrength = std::clamp(params.detailStrength, 0.0f, 1.0f);
    params.photoGateThreshold = std::clamp(params.photoGateThreshold, 0.0f, 255.0f);
    params.photoGateSigma = std::clamp(params.photoGateSigma, 0.01f, 255.0f);

    if (listWindows) {
        WindowHelper::printWindowList();
        return 0;
    }

    if (!captureWindowQuery.empty()) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runRealtimeGPUInterpolation(captureWindowQuery, gpuSettings, sourceFps, outputMultiplier);
    }

    if (runTest) {
        return runSyntheticTest(params);
    }

    if (positionalArgs.size() >= 3) {
        input0Path = positionalArgs[0];
        input1Path = positionalArgs[1];
        outputPath = positionalArgs[2];
    } else {
        std::cerr << "Error: missing required positional arguments <frame0.png> <frame1.png> <output.png>\n";
        printUsage(argv[0]);
        return 1;
    }

    if (gpuOffline) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runOfflineGPUInterpolation(input0Path, input1Path, outputPath,
                                          flowOutputPath, groundTruthPath, t, gpuSettings);
    }

    std::cout << "Loading input frames:\n"
              << "  Frame 0: " << input0Path << "\n"
              << "  Frame 1: " << input1Path << "\n";

    Image frame0 = Image::load(input0Path);
    if (frame0.empty()) return 1;

    Image frame1 = Image::load(input1Path);
    if (frame1.empty()) return 1;

    if (frame0.width != frame1.width || frame0.height != frame1.height) {
        std::cerr << "Error: frame dimensions mismatch (" 
                  << frame0.width << "x" << frame0.height << " vs "
                  << frame1.width << "x" << frame1.height << ")\n";
        return 1;
    }

    std::string subpelName = "quarter-pel bilinear";
    if (params.matcherParams.subpelMode == SUBPEL_NONE) subpelName = "none";
    else if (params.matcherParams.subpelMode == SUBPEL_PARABOLIC) subpelName = "parabolic peak fit";
    else if (params.matcherParams.subpelMode == SUBPEL_HALF) subpelName = "half-pel bilinear";

    std::cout << "Image resolution: " << frame0.width << "x" << frame0.height 
              << ", channels: " << frame0.channels << "\n";
    std::cout << "Configuration:\n"
              << "  Pyramid levels:      " << params.matcherParams.pyramidLevels << "\n"
              << "  Block size:          " << params.matcherParams.blockSize << "x" << params.matcherParams.blockSize
              << (params.matcherParams.adaptiveBlockSize ? " (adaptive [" + std::to_string(params.matcherParams.minBlockSize) + ".." + std::to_string(params.matcherParams.maxBlockSize) + "])" : "") << "\n"
              << "  Gaussian weighting:  " << (params.matcherParams.useGaussianWeights ? "enabled" : "disabled") << "\n"
              << "  Spatial predictors:  " << (params.matcherParams.useSpatialPredictors ? "enabled (EPZS)" : "disabled") << "\n"
              << "  Subpixel mode:       " << subpelName << "\n"
              << "  Coarse search:       [-" << params.matcherParams.searchRadiusCoarse << ", +" << params.matcherParams.searchRadiusCoarse << "]\n"
              << "  Refinement search:   [-" << params.matcherParams.searchRadiusRefine << ", +" << params.matcherParams.searchRadiusRefine << "]\n"
              << "  Grid step:           " << params.matcherParams.gridStep << "\n"
              << "  Bidirectional:       " << (params.bidirectional ? "enabled" : "disabled") << "\n"
              << "  Photometric gating:  " << (params.enablePhotometricGating ? "enabled" : "disabled") << "\n"
              << "  Hole inpainting:     " << (params.enableHoleInpainting ? "enabled" : "disabled") << "\n"
              << "  Detail restoration:  " << (params.enableDetailRestoration ? ("enabled (strength: " + std::to_string(params.detailStrength) + ")") : "disabled") << "\n"
              << "  Color clamping:      " << (params.enableColorClamping ? "enabled" : "disabled") << "\n"
              << "  Target time (t):     " << t << "\n";

    FrameInterpolator interpolator(params);
    FlowField forwardFlow, backwardFlow;
    Image occMask;

    auto start = std::chrono::high_resolution_clock::now();

    Image output = interpolator.interpolate(
        frame0, frame1, t,
        flowOutputPath.empty() ? nullptr : &forwardFlow,
        nullptr,
        occOutputPath.empty() ? nullptr : &occMask
    );

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Interpolation completed in " << std::fixed << std::setprecision(2) << elapsedMs << " ms\n";

    if (!output.savePNG(outputPath)) {
        std::cerr << "Failed to save interpolated frame to " << outputPath << "\n";
        return 1;
    }
    std::cout << "Saved interpolated frame to: " << outputPath << "\n";

    if (!flowOutputPath.empty()) {
        Image flowVis = forwardFlow.toColorImage();
        flowVis.savePNG(flowOutputPath);
        std::cout << "Saved motion flow visualization to: " << flowOutputPath << "\n";
    }

    if (!occOutputPath.empty()) {
        Image occVis(frame0.width, frame0.height, 3);
        for (int y = 0; y < frame0.height; ++y) {
            for (int x = 0; x < frame0.width; ++x) {
                float v = occMask.get(x, y, 0) * 255.0f;
                occVis.set(x, y, 0, v);
                occVis.set(x, y, 1, v);
                occVis.set(x, y, 2, v);
            }
        }
        occVis.savePNG(occOutputPath);
        std::cout << "Saved occlusion mask to: " << occOutputPath << "\n";
    }

    return 0;
}
