#include "Image.h"
#include "Pyramid.h"
#include "MotionVector.h"
#include "ZNCCMatcher.h"
#include "FrameInterpolator.h"
#include "RealtimeSession.h"
#include "WindowHelper.h"
#include "GuiApp.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cctype>

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
              << "  " << progName << " --gpu-offline <f0.png> <f1.png> <out.png> [--ground-truth gt.png] [--save-flow f.png] [--gpu-offline-prev p.png]\n"
              << "                                        # Run the real-time GPU shader pipeline on two PNGs\n"
              << "  " << progName << " --test [options]                    # Synthetic verification benchmark\n"
              << "  " << progName << " --d3d12-self-test                   # Native dispatch and shared-texture probe\n\n"
              << "Real-Time GPU Overlay Options:\n"
              << "  --capture-window <title> Target window title substring to capture via WGC\n"
              << "  --source-fps <auto|24|30|60>  Source content rate (default: auto)\n"
              << "  --multiplier <2|3|4|max>      Output FPS multiplier (default: max = display rate)\n"
              << "  --gpu-levels <1-8>       Real-time pyramid levels (default: 8)\n"
              << "  --gpu-min-refine <n>     Finest searched level, 0 to levels-1 (default: 0)\n"
              << "  --gpu-coarse-radius <0-8>   Coarse search radius (default: 8)\n"
              << "  --gpu-refine-radius <0-8>   Finest-level search radius (default: 8)\n"
              << "  --gpu-smoothness <0-0.1>    Predictor smoothness weight (default: 0.0005)\n"
              << "  --flow-engine <ffx|msad>    Optical flow engine (default: ffx; MSAD is\n"
              << "                              the automatic and forced fallback)\n"
              << "  --adapter <auto|igpu|dgpu>  Graphics adapter preference (default: auto = dgpu)\n"
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
    std::string previousFramePath;
    bool gpuOffline = false;
    uint32_t sourceFps = 0;
    uint32_t outputMultiplier = 0;   // Max: present at the display refresh rate
    GraphicsAdapterPreference adapterPreference = GraphicsAdapterPreference::Auto;
    FlowSettings gpuSettings;

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
    bool d3d12SelfTest = false;

    std::vector<std::string> positionalArgs;

    std::string currentArgument;
    try {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        currentArgument = arg;
        if (arg == "--test") {
            runTest = true;
        } else if (arg == "--d3d12-self-test") {
            d3d12SelfTest = true;
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
        } else if (arg == "--gpu-offline-prev" && i + 1 < argc) {
            previousFramePath = argv[++i];
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
        } else if (arg == "--flow-engine" && i + 1 < argc) {
            std::string engine = argv[++i];
            std::transform(engine.begin(), engine.end(), engine.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (engine == "ffx" || engine == "amd") {
                gpuSettings.preferFfxOpticalFlow = true;
            } else if (engine == "msad" || engine == "native") {
                gpuSettings.preferFfxOpticalFlow = false;
            } else {
                throw std::invalid_argument("flow engine must be 'ffx' or 'msad'");
            }
        } else if (arg == "--adapter" && i + 1 < argc) {
            std::string adapter = argv[++i];
            std::transform(adapter.begin(), adapter.end(), adapter.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (adapter == "auto") {
                adapterPreference = GraphicsAdapterPreference::Auto;
            } else if (adapter == "igpu" || adapter == "integrated") {
                adapterPreference = GraphicsAdapterPreference::Integrated;
            } else if (adapter == "dgpu" || adapter == "discrete") {
                adapterPreference = GraphicsAdapterPreference::Discrete;
            } else {
                throw std::invalid_argument("adapter must be 'auto', 'igpu', or 'dgpu'");
            }
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

    if (d3d12SelfTest) {
        return runD3D12SelfTest(adapterPreference);
    }

    if (!captureWindowQuery.empty()) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        RealtimeOptions options;
        options.windowTitle = captureWindowQuery;
        options.flow = gpuSettings;
        options.sourceFps = sourceFps;
        options.outputMultiplier = outputMultiplier;
        options.adapter = adapterPreference;
        return runRealtimeSession(options);
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
        return runOfflineGpuInterpolation(input0Path, input1Path, outputPath,
                                          flowOutputPath, groundTruthPath, t, gpuSettings,
                                          adapterPreference, previousFramePath);
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
