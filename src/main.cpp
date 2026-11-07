#include "Image.h"
#include "Pyramid.h"
#include "MotionVector.h"
#include "ZNCCMatcher.h"
#include "FrameInterpolator.h"
#include "RealtimeSession.h"
#include "NeuralFlow.h"
#include "WindowHelper.h"
#include "GuiApp.h"
#include "FrameTimeline.h"

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
              << "  " << progName << " --fsr-fg-offline <f0.png> <f1.png> <out.png> [--ground-truth gt.png] [--gpu-offline-prev p.png] [--fsr-depth 0-1] [--fsr-frames 1-4]\n"
              << "                                        # Experimental: AMD FSR frame generation fed with this flow\n"
              << "  " << progName << " --fsr-realtime-test <f0.png> <f1.png> <f2.png>... [--ai-depth --fsr-depth-span s] [--fsr-save dir]\n"
              << "                                        # The live --fsr-fg path on frames in order\n"
              << "  " << progName << " --test [options]                    # Synthetic verification benchmark\n"
              << "  " << progName << " --d3d12-self-test                   # Native dispatch and shared-texture probe\n"
              << "  " << progName << " --timeline-self-test                # CPU-only output clock and source queue checks\n"
              << "  " << progName << " --projection-self-test              # High-resolution GPU projection regression\n"
              << "  " << progName << " --ai-blend-mask-self-test           # Learned-mask synthesis regression\n\n"
              << "  " << progName << " --ai-refinement-self-test           # Paired AI/DIS refinement regression\n"
              << "  " << progName << " --train-neural <model.bin> <video.mp4|image.png> [more files...]  # Train the convolutional estimator\n"
              << "  " << progName << " --neural-epochs <n>                 # Training epochs (default: 24)\n"
              << "  " << progName << " --neural-gradient-check             # Verify the analytic gradient\n"
              << "  " << progName << " --neural-parity-check <f0.png> <f1.png>  # Verify the shaders match the model\n"
              << "  " << progName << " --gpu-offline-sequence <prefix>     # Replay <prefix>_f000.png... as a temporal chain\n"
              << "  " << progName << " --gpu-flow-eval <prefix>...         # Per-level vector error vs <prefix>_flow.bin\n\n"
              << "Real-Time GPU Overlay Options:\n"
              << "  --capture-window <title> Target window title substring to capture via WGC\n"
              << "  --source-fps <auto|24|30|60>  Source content rate (default: auto)\n"
              << "  --multiplier <2|3|4|max>      Exact source FPS multiplier (default: max = display-rate FPS)\n"
              << "                                60 FPS x2 always schedules 120 FPS, regardless of monitor Hz\n"
              << "  --switch-width <w>            Hand-over from frame0's warp to frame1's: 0 = pure warping\n"
              << "                                (default), up to 0.45 = spread over 0.5 +/- w of the pair,\n"
              << "                                1 = the old full blend (same as --allow-crossfade)\n"
              << "  --gpu-levels <1-10>      Real-time pyramid levels (default: 8)\n"
              << "  --gpu-min-refine <n>     Finest searched level, 0 to levels-1 (default: 0)\n"
              << "  --gpu-coarse-radius <0-8>   Coarse search radius (default: 4)\n"
              << "  --gpu-refine-radius <0-8>   Finest-level search radius (default: 6)\n"
              << "  --gpu-smoothness <0-0.1>    Predictor smoothness weight (default: 0.0005)\n"
              << "  --flow-engine <dis|sad|ffx|neural|ai>  Engine (default: dis = Dense Inverse Search;\n"
              << "                             ai = the trained model in models\\ai_flow.onnx)\n"
              << "  --ai-blend-mask          Use AI's learned blending mask (AI only, experimental; default: off)\n"
              << "  --ai-refinement          Integrated AI refinement instead of DIS (paired refined export required)\n"
              << "  --ai-temporal-ui         Trained temporal UI candidates (AI pyramid + UI mask; separate ai_flow_ui.onnx)\n"
              << "  --ai-temporal-ui-self-test  Verify temporal state chaining and reset on DirectML\n"
              << "  --no-ai-refinement       Backbone-only AI + DIS refinement (default; independent of blending mask)\n"
              << "  --no-ui-mask             Interpolate static UI too (default: HUD/subtitles kept sharp)\n"
              << "  --no-ui-persistence      UI mask without HUD persistence (screen-fixed structure, no model)\n"
              << "  --ai-ui-regions          Native UI mask only inside AI-detected regions (optional ui_detect.onnx)\n"
              << "  --ui-detector <type>     auto|icons|matrix; icon detection only, not flow\n"
              << "  --ui-detect-test <in.png> <out.json>  Test UI boxes and worker latency; no capture\n"
              << "  --ui-regions-self-test   Verify native ROI gating, decoder and expiry rules\n"
              << "  --fsr-fg                 Experimental: AMD FSR frame generation beside the normal interpolation;\n"
              << "                           its middle frame is shown near the middle of each pair (all\n"
              << "                           generated frames at 2x on every monitor). Off by default.\n"
              << "  --fsr-window <0-0.5>     How near the middle FSR's frame is used (default: 0.1);\n"
              << "                           fixed multipliers use only the exact midpoint, never real frames\n"
              << "  --fsr-depth-span <0-1>   Give FSR the AI depth (needs --ai-depth), spread over this much of\n"
              << "                           its depth range (default: 0.25); 0 = flat depth\n"
              << "  --fsr-camera <off|spin|flow>  Experimental FSR camera (default: off = fixed). spin: a constant\n"
              << "                           turn, which FSR measured better with; flow: turned by the flow's\n"
              << "                           overall motion; exact = flow, offline tests only\n"
              << "  --fsr-camera-fov <deg>   The game's assumed vertical field of view for it (default: 60)\n"
              << "  --latency-min <0-50>     Minimum buffering margin above one source frame (default: 4 ms)\n"
              << "                           Default fixed budget: two source periods + 10 ms; no stall adaptation\n"
              << "  --latency-max <ms|auto>  Cap on the fixed buffering budget, 20-300 ms\n"
              << "                           (default: auto = 3 source frames + 60 ms)\n"
              << "  --frame-latency <1-2>    Frames the swap chain may hold (default: 1); 2 absorbs a slow\n"
              << "                           frame at one refresh more delay\n"
              << "  --capture-queue <2-8>    Captured frames that may wait to be processed (default: 4)\n"
              << "  --vsync <on|off>         Legacy tearing switch: on = disabled (default), off = allowed\n"
              << "                           Both use the FPS clock and Present(0), with no vsync waits\n"
              << "  --adapter <auto|igpu|dgpu>  Graphics adapter preference (default: auto = dgpu)\n"
              << "  --list-windows           List all active top-level desktop windows\n\n"
              << "Offline / Algorithm Options:\n"
              << "  --time <float>          Intermediate timestamp in (0.0, 1.0) (default: 0.5)\n"
              << "  --levels <1-10>         Number of CPU pyramid levels (default: 8)\n"
              << "  --block-size <int>      Nominal block matching window size (default: 8)\n"
              << "  --search-radius <int>   Refinement search radius per level (default: 4)\n"
              << "  --coarse-radius <int>   Coarsest level search radius (default: 8)\n"
              << "  --grid-step <int>       Grid step: 1 = dense, 2 = 2x2 grid (default: 2)\n"
              << "  --subpel <mode>         Subpixel mode: none, parabolic, half, quarter (default: quarter)\n"
              << "  --smoothness <float>    Predictor smoothness weight in [0, 0.1]\n"
              << "  --no-spatial-pred       Disable multi-candidate spatial predictor evaluation (EPZS)\n"
              << "  --no-gaussian-weight    Disable 2D Gaussian block window weighting\n"
              << "  --no-adaptive-block     Disable pyramid level-adaptive block sizing\n"
              << "  --min-block-size <int>  Min block size at fine levels (default: 6)\n"
              << "  --max-block-size <int>  Max block size at coarse levels (default: 14)\n"
              << "  --no-bidirectional      Disable backward flow estimation\n"
              << "  --no-detail-restore     Disable high-frequency detail restoration\n"
              << "  --no-color-clamp        Disable local color bounding box clamping\n"
              << "  --detail-strength <flt> Detail restoration strength in [0.0, 1.0] (default: 0.65)\n"
              << "  --save-flow <path>      Save color-coded forward flow field to PNG\n"
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

    std::cout << "Interpolating intermediate frame at t = 0.5 using " << params.matcherParams.pyramidLevels << " pyramid levels...\n";
    auto start = std::chrono::high_resolution_clock::now();

    Image interpolated = interpolator.interpolate(frame0, frame1, 0.5f, &forwardFlow, &backwardFlow);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    interpolated.savePNG("test_interpolated.png");
    Image flowVis = forwardFlow.toColorImage();
    flowVis.savePNG("test_flow.png");

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
              << "  Saved Outputs:           test_interpolated.png, test_flow.png\n";

    if (errX < 0.5f && errY < 0.5f) {
        std::cout << ">>> TEST PASSED (Motion vector accurately captured within < 0.5 px tolerance!)\n\n";
        return 0;
    } else {
        std::cout << ">>> Warning: Motion estimation error exceeds 0.5 px tolerance\n\n";
        return 1;
    }
}


// Frame pacing on a source whose timestamps land on the refresh grid (a
// game rendering at a rate that does not divide the display refresh reaches
// the screen, and WGC's timestamps, after 2, 4, 3... refreshes) while its
// content advances evenly. Every displayed frame's position along the
// source (pair index + alpha) should advance by the same step.
static int runTimelineSelfTest() {
    constexpr int64_t second = 10000000, ms = 10000, start = 1000000000;
    bool ok = true;
    auto push = [&](FrameTimeline& timeline, int k, int64_t ts, int64_t arrival) {
        TimelineFrame frame;
        frame.slot = k;
        frame.tsRaw = ts;
        frame.arrival = arrival;
        frame.flowEntry = 0;
        timeline.push(frame);
    };

    // Prebuffered clock tests isolate output from capture and scanout. All
    // deadlines and k/N positions must be identical on every monitor, even
    // when requested FPS exceeds display Hz. No GPU or live capture is used.
    int cases = 0, failures = 0;
    for (uint32_t source : {24u, 30u, 60u}) {
        for (uint32_t n : {2u, 3u, 4u}) {
            for (double hz : {60.0, 75.0, 120.0, 144.0, 165.0, 240.0, 360.0}) {
                FrameTimeline timeline;
                timeline.configure(source, hz, n);
                for (int k = 0; k < static_cast<int>(source * 6 + 2); ++k) {
                    const int64_t ts = start + static_cast<int64_t>(static_cast<double>(k) * second / source + 0.5);
                    push(timeline, k, ts, ts + 3 * ms);
                }
                const double budget = timeline.stats().latencyBudgetMs;
                int64_t now = start, epoch = 0;
                bool pass = true;
                double worst = 0.0;
                for (uint32_t k = 0; k < source * n * 5; ++k) {
                    const auto sel = timeline.nextOutput(now);
                    if (k == 0) epoch = sel.presentTime;
                    const double expected = epoch + static_cast<double>(k) * second / (source * n);
                    worst = std::max(worst, std::fabs(sel.presentTime - expected));
                    const float alpha = static_cast<float>(k % n) / n;
                    if (!sel.valid || !sel.fixedCadence || sel.f0.slot != static_cast<int>(k / n) ||
                        sel.alpha != alpha || worst > 1.0 ||
                        sel.canUseMidpoint(0.5f) != (alpha == 0.5f)) pass = false;
                    // Reports and wake-up jitter must not stretch the clock.
                    if (k % 31 == 0) timeline.reportStarvation();
                    now = sel.presentTime + (k % 4) * 1000;
                    timeline.retireBefore(sel.f0.index);
                }
                if (timeline.stats().latencyBudgetMs != budget || timeline.stats().missedOutputs != 0) pass = false;
                ++cases;
                failures += !pass;
                if (!pass || (source == 60 && n == 2))
                    std::cout << "Timeline self-test: " << source << " fps " << n << "x / " << hz << " Hz: "
                              << source * n << " FPS, max deadline error " << worst * 0.1 << " us: "
                              << (pass ? "PASS" : "FAIL") << "\n";
            }
        }
    }
    std::cout << "Timeline self-test: " << cases << " monitor-independent clock cases, " << failures << " failures\n";
    ok = failures == 0;

    // Max explicitly chooses display-rate FPS, but uses the same absolute
    // timer rather than bringing the deleted compositor pipeline back.
    for (double hz : {120.0, 144.0, 165.0, 240.0}) {
        FrameTimeline timeline;
        timeline.configure(60, hz, 0);
        for (int k = 0; k < 180; ++k) {
            const int64_t ts = start + static_cast<int64_t>(static_cast<double>(k) * second / 60);
            push(timeline, k, ts, ts + 3 * ms);
        }
        int64_t now = start, epoch = 0;
        bool pass = true;
        for (int k = 0; k < static_cast<int>(hz * 2); ++k) {
            const auto sel = timeline.nextOutput(now);
            if (k == 0) epoch = sel.presentTime;
            const double expected = epoch + static_cast<double>(k) * second / hz;
            if (!sel.valid || sel.fixedCadence || std::fabs(sel.presentTime - expected) > 1.0) pass = false;
            now = sel.presentTime + 1000;
            timeline.retireBefore(sel.f0.index);
        }
        std::cout << "Timeline self-test: Max / " << hz << " Hz on FPS timer: " << (pass ? "PASS" : "FAIL") << "\n";
        ok = ok && pass;
    }

    // Streaming auto-cadence test: quantized WGC timestamps and 1.5-3.5 ms
    // delivery jitter, but output is consumed on its own timer, not vsync.
    for (double hz : {120.0, 144.0, 165.0, 240.0}) {
        FrameTimeline timeline;
        timeline.configure(0, hz, 2);
        const int64_t refresh = static_cast<int64_t>(second / hz + 0.5);
        int next = 0, outputs = 0, wrong = 0;
        int64_t firstOutput = 0, lastOutput = 0;
        double lastContent = -1.0;
        FrameTimeline::Selection pending;
        for (int64_t now = start; now < start + 10 * second; now += 1000) {
            for (;;) {
                const int64_t present = start + static_cast<int64_t>(static_cast<double>(next) * second / 60) + 3 * ms;
                const int64_t ts = ((present + refresh - 1) / refresh) * refresh;
                const int64_t arrival = ts + 15 * ms / 10 + (next % 5) * ms / 2;
                if (arrival > now) break;
                const auto plan = timeline.onFrameArrived(ts, arrival);
                if (plan.replaceLast) timeline.popLast();
                if (plan.acceptThis) push(timeline, next, ts, arrival);
                else ++wrong; // this source has no intermediate repaints
                ++next;
            }
            if (!pending.valid) pending = timeline.nextOutput(now);
            if (!pending.valid || pending.presentTime > now) continue;
            if (now >= start + 4 * second) {
                const double content = pending.f0.slot + pending.alpha;
                if (lastContent >= 0.0 && std::fabs(content - lastContent - 0.5) > 1e-6) ++wrong;
                if (pending.alpha != 0.0f && pending.alpha != 0.5f) ++wrong;
                if (!outputs) firstOutput = pending.presentTime;
                lastOutput = pending.presentTime;
                lastContent = content;
                ++outputs;
            }
            timeline.retireBefore(pending.f0.index);
            pending = {};
        }
        const double fps = outputs > 1 ? (outputs - 1) * static_cast<double>(second) / (lastOutput - firstOutput) : 0.0;
        const bool pass = std::fabs(fps - 120.0) < 1.0 && wrong == 0 && timeline.stats().frames < 12;
        std::cout << "Timeline self-test: auto 60 fps 2x / " << hz << " Hz: " << std::fixed << std::setprecision(3)
                  << fps << " FPS, " << wrong << " wrong positions: " << (pass ? "PASS" : "FAIL") << "\n";
        ok = ok && pass;
    }

    // Late work skips expired slots once, without a burst of queued frames.
    {
        FrameTimeline timeline;
        timeline.configure(60, 144, 2);
        for (int k = 0; k < 120; ++k) {
            const int64_t ts = start + static_cast<int64_t>(static_cast<double>(k) * second / 60);
            push(timeline, k, ts, ts + 3 * ms);
        }
        const auto first = timeline.nextOutput(start);
        const int64_t late = first.presentTime + second / 2;
        const auto recovered = timeline.nextOutput(late);
        const auto following = timeline.nextOutput(std::max(late, recovered.presentTime));
        const bool pass = recovered.valid && recovered.f0.index > first.f0.index && following.valid &&
                          following.presentTime > late && timeline.stats().missedOutputs > 0;
        std::cout << "Timeline self-test: stalled output skips expired slots without catch-up burst: "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass) std::cout << "  late=" << late << " recovered=" << recovered.presentTime << " valid=" << recovered.valid
                             << " next=" << following.presentTime << " valid=" << following.valid
                             << " missed=" << timeline.stats().missedOutputs << "\n";
        ok = ok && pass;
    }

    // Discarding every queued output must still free capture slots, or
    // waiting for fresh input would deadlock a full producer ring.
    {
        FrameTimeline timeline;
        timeline.configure(60, 144, 2);
        for (int k = 0; k < 12; ++k) {
            const int64_t ts = start + static_cast<int64_t>(static_cast<double>(k) * second / 60);
            push(timeline, k, ts, ts + 3 * ms);
        }
        const auto first = timeline.nextOutput(start);
        const int64_t late = first.presentTime + second;
        const auto empty = timeline.nextOutput(late);
        const auto released = timeline.retireBefore(empty.retireBeforeIndex);
        const bool freed = !empty.valid && released.size() == 11 && timeline.stats().frames == 1;
        push(timeline, 12, late, late + 3 * ms);
        const auto resumed = timeline.nextOutput(late + 3 * ms);
        const bool pass = freed && resumed.valid && resumed.f0.slot == 12 && resumed.alpha == 0.0f;
        std::cout << "Timeline self-test: exhausted capture queue retires stale slots and resumes: "
                  << (pass ? "PASS" : "FAIL") << "\n";
        ok = ok && pass;
    }

    // A static/paused source shows its last real frame once, then sleeps.
    {
        FrameTimeline timeline;
        timeline.configure(60, 144, 2);
        push(timeline, 0, start, start + 3 * ms);
        push(timeline, 1, start + second / 60, start + second / 60 + 3 * ms);
        int64_t now = start;
        bool pass = true;
        for (int k = 0; k < 3; ++k) {
            const auto sel = timeline.nextOutput(now);
            pass = pass && sel.valid && sel.f0.slot + sel.alpha == k * 0.5f;
            now = sel.presentTime;
            timeline.retireBefore(sel.f0.index);
        }
        for (int k = 0; k < 100; ++k) {
            now += ms;
            if (timeline.nextOutput(now).valid) pass = false;
        }
        push(timeline, 2, now, now + 3 * ms);
        const auto resumed = timeline.nextOutput(now + 3 * ms);
        pass = pass && resumed.valid && resumed.f0.slot == 2 && resumed.alpha == 0.0f;
        std::cout << "Timeline self-test: pause, final real frame, resume without stale interpolation: "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass) std::cout << "  resumed valid=" << resumed.valid << " slot=" << resumed.f0.slot
                             << " alpha=" << resumed.alpha << "\n";
        ok = ok && pass;
    }
    std::cout << "Timeline self-test: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}


int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--timeline-self-test") return runTimelineSelfTest();
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
    std::string captureWindowQuery;
    std::string groundTruthPath;
    std::string previousFramePath;
    std::string stopEventName;
    bool gpuOffline = false;
    bool fsrFrameGenOffline = false;   // experimental: AMD FSR frame generation instead of InterpolateCS
    float fsrDepth = 0.5f;
    uint32_t fsrFrames = 1;
    bool fsrFrameGen = false;          // live: FSR frame generation beside the normal interpolation
    bool fsrRealtimeTest = false;      // --fsr-realtime-test <frames...>: the live FSR path offline
    float fsrWindow = 0.1f;
    float fsrDepthSpan = 0.25f;        // RealtimeOptions::fsrDepthSpan
    FsrCameraSettings fsrCamera;       // --fsr-camera, --fsr-camera-fov, --fsr-camera-flip
    RealtimeOptions queues;            // --latency-min/-max, --frame-latency, --capture-queue
    std::string fsrSaveFolder;         // --fsr-save <dir>: --fsr-realtime-test writes its middle frames there
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
    bool projectionSelfTest = false;
    bool aiBlendMaskSelfTest = false;
    bool aiRefinementSelfTest = false;
    bool aiTemporalUiSelfTest = false;
    bool uiRegionsSelfTest = false;
    std::string uiDetectInput, uiDetectOutput;
    std::string neuralModelPath;
    int neuralEpochs = 24;
    bool neuralGradientCheck = false;
    bool neuralParityCheck = false;
    std::string sequencePrefix;
    std::string depthTestInput, depthTestOutput;
    std::string aiFlowTestModel;
    bool flowEval = false;   // --gpu-flow-eval <prefix>...: positional prefixes

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
        } else if (arg == "--projection-self-test") {
            projectionSelfTest = true;
        } else if (arg == "--ai-blend-mask-self-test") {
            aiBlendMaskSelfTest = true;
        } else if (arg == "--ai-refinement-self-test") {
            aiRefinementSelfTest = true;
        } else if (arg == "--ai-temporal-ui-self-test") {
            aiTemporalUiSelfTest = true;
        } else if (arg == "--ui-regions-self-test") {
            uiRegionsSelfTest = true;
        } else if (arg == "--ui-detect-test" && i + 2 < argc) {
            uiDetectInput = argv[++i];
            uiDetectOutput = argv[++i];
        } else if (arg == "--gpu-offline-sequence" && i + 1 < argc) {
            sequencePrefix = argv[++i];
        } else if (arg == "--ai-flow-test" && i + 1 < argc) {
            aiFlowTestModel = argv[++i];
        } else if (arg == "--depth-test" && i + 2 < argc) {
            depthTestInput = argv[++i];
            depthTestOutput = argv[++i];
        } else if (arg == "--gpu-flow-eval") {
            flowEval = true;
        } else if (arg == "--neural-parity-check") {
            neuralParityCheck = true;
        } else if (arg == "--train-neural" && i + 1 < argc) {
            neuralModelPath = argv[++i];
        } else if (arg == "--neural-epochs" && i + 1 < argc) {
            neuralEpochs = std::clamp(std::stoi(argv[++i]), 1, 500);
        } else if (arg == "--neural-gradient-check") {
            neuralGradientCheck = true;
        } else if (arg == "--list-windows") {
            listWindows = true;
        } else if (arg == "--capture-window" && i + 1 < argc) {
            captureWindowQuery = argv[++i];
        } else if (arg == "--stop-event" && i + 1 < argc) {
            stopEventName = argv[++i];
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
        } else if (arg == "--fsr-fg-offline") {
            fsrFrameGenOffline = true;
        } else if (arg == "--fsr-realtime-test") {
            fsrRealtimeTest = true;
        } else if (arg == "--fsr-fg") {
            fsrFrameGen = true;
        } else if (arg == "--fsr-window" && i + 1 < argc) {
            fsrWindow = std::clamp(std::stof(argv[++i]), 0.0f, 0.5f);
        } else if (arg == "--fsr-save" && i + 1 < argc) {
            fsrSaveFolder = argv[++i];
        } else if (arg == "--latency-min" && i + 1 < argc) {
            queues.latencyMinExtraMs = std::clamp(std::stod(argv[++i]), 0.0, 50.0);
        } else if (arg == "--latency-max" && i + 1 < argc) {
            const std::string value = argv[++i];
            queues.latencyMaxMs = value == "auto" ? 0.0 : std::clamp(std::stod(value), 20.0, 300.0);
        } else if (arg == "--frame-latency" && i + 1 < argc) {
            queues.frameLatency = std::clamp(std::stoi(argv[++i]), 1, 2);
        } else if (arg == "--vsync" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value != "on" && value != "off") throw std::invalid_argument("vsync must be on or off");
            queues.vsync = value == "on";
        } else if (arg == "--capture-queue" && i + 1 < argc) {
            queues.captureQueue = std::clamp(std::stoi(argv[++i]), 2, 8);
        } else if (arg == "--fsr-camera" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "off") fsrCamera.mode = 0;
            else if (mode == "flow") fsrCamera.mode = 1;
            else if (mode == "exact") fsrCamera.mode = 2;
            else if (mode == "spin") fsrCamera.mode = 3;
            else throw std::invalid_argument("FSR camera must be off, spin, flow or exact");
        } else if (arg == "--fsr-camera-fov" && i + 1 < argc) {
            fsrCamera.fovDegrees = std::clamp(std::stof(argv[++i]), 20.0f, 120.0f);
        } else if (arg == "--fsr-camera-flip" && i + 1 < argc) {
            const std::string flip = argv[++i];
            fsrCamera.yawSign = (flip == "yaw" || flip == "both") ? -1.0f : 1.0f;
            fsrCamera.pitchSign = (flip == "pitch" || flip == "both") ? -1.0f : 1.0f;
            if (flip != "none" && flip != "yaw" && flip != "pitch" && flip != "both")
                throw std::invalid_argument("FSR camera flip must be none, yaw, pitch or both");
        } else if (arg == "--fsr-depth-span" && i + 1 < argc) {
            fsrDepthSpan = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
        } else if (arg == "--fsr-depth" && i + 1 < argc) {
            fsrDepth = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
        } else if (arg == "--fsr-frames" && i + 1 < argc) {
            fsrFrames = static_cast<uint32_t>(std::clamp(std::stoi(argv[++i]), 1, 4));
        } else if (arg == "--ground-truth" && i + 1 < argc) {
            groundTruthPath = argv[++i];
        } else if (arg == "--gpu-offline-prev" && i + 1 < argc) {
            previousFramePath = argv[++i];
        } else if (arg == "--gpu-levels" && i + 1 < argc) {
            gpuSettings.pyramidLevels = std::clamp(std::stoi(argv[++i]), 1, FlowEngine::kMaxLevels);
        } else if (arg == "--gpu-min-refine" && i + 1 < argc) {
            gpuSettings.minRefineLevel = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--gpu-coarse-radius" && i + 1 < argc) {
            gpuSettings.coarseSearchRadius = std::clamp(std::stoi(argv[++i]), 0, 8);
        } else if (arg == "--gpu-refine-radius" && i + 1 < argc) {
            gpuSettings.refineSearchRadius = std::clamp(std::stoi(argv[++i]), 0, 8);
        } else if (arg == "--no-ui-mask") {
            gpuSettings.uiMask = false;
        } else if (arg == "--no-ui-persistence") {
            gpuSettings.uiPersistence = false;
        } else if (arg == "--allow-crossfade") {
            SetEnvironmentVariableA("MOTION_ENHANCER_ALLOW_CROSSFADE", "1");
        } else if (arg == "--switch-width" && i + 1 < argc) {
            SetEnvironmentVariableA("MOTION_ENHANCER_SWITCH_WIDTH", argv[++i]);
        } else if (arg == "--ai-ui-regions") {
            gpuSettings.aiUiRegions = true;
        } else if (arg == "--ui-detector" && i + 1 < argc) {
            const std::string type = argv[++i];
            if (type == "auto") gpuSettings.uiDetector = UiDetector::Choice::Automatic;
            else if (type == "icons") gpuSettings.uiDetector = UiDetector::Choice::DirectMLIcons;
            else if (type == "matrix") gpuSettings.uiDetector = UiDetector::Choice::MatrixIcons;
            else { std::cerr << "UI detector must be auto, icons or matrix. Text/HUD detection has been removed.\n"; return 1; }
        } else if (arg == "--ai-blend-mask") {
            gpuSettings.aiBlendMask = true;
        } else if (arg == "--ai-refinement") {
            gpuSettings.aiRefinement = true;
        } else if (arg == "--ai-temporal-ui") {
            gpuSettings.aiTemporalUi = true;
        } else if (arg == "--no-ai-refinement") {
            gpuSettings.aiRefinement = false;
        } else if (arg == "--no-occlusion-fill") {
            gpuSettings.occlusionFill = false;
        } else if (arg == "--ai-depth") {
            gpuSettings.aiDepth = true;
        } else if (arg == "--ai-depth-every" && i + 1 < argc) {
            gpuSettings.aiDepthStride = std::clamp(std::stoi(argv[++i]), 1, 4);
        } else if (arg == "--no-flow-median") {
            gpuSettings.flowMedian = false;
        } else if (arg == "--no-ffx-refine") {
            gpuSettings.ffxRefine = false;
        } else if (arg == "--ffx-occlusion") {
            gpuSettings.ffxOcclusion = true;
        } else if (arg == "--no-ffx-repair") {
            gpuSettings.ffxRepair = false;
        } else if (arg == "--gpu-smoothness" && i + 1 < argc) {
            gpuSettings.smoothnessWeight = std::clamp(std::stof(argv[++i]), 0.0f, 0.1f);
        } else if (arg == "--flow-engine" && i + 1 < argc) {
            std::string engine = argv[++i];
            std::transform(engine.begin(), engine.end(), engine.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            gpuSettings.preferFfxOpticalFlow = engine == "ffx" || engine == "amd";
            gpuSettings.neuralFlow = engine == "neural";
            gpuSettings.disFlow = engine == "dis" || engine == "ai";   // DIS is the AI engine's fallback
            gpuSettings.aiFlow = engine == "ai";
            if (!gpuSettings.preferFfxOpticalFlow && !gpuSettings.neuralFlow && !gpuSettings.disFlow &&
                engine != "sad" && engine != "msad" && engine != "native") {
                throw std::invalid_argument("flow engine must be 'sad', 'dis', 'ffx', 'neural' or 'ai'");
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
        } else if (arg == "--detail-restore" && i + 1 < argc) {
            params.enableDetailRestoration = std::stoi(argv[++i]) != 0;
        } else if (arg == "--color-clamp" && i + 1 < argc) {
            params.enableColorClamping = std::stoi(argv[++i]) != 0;
        } else if (arg == "--detail-strength" && i + 1 < argc) {
            params.detailStrength = std::stof(argv[++i]);
        } else if (arg == "--save-flow" && i + 1 < argc) {
            flowOutputPath = argv[++i];
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

    params.matcherParams.pyramidLevels = std::clamp(params.matcherParams.pyramidLevels, 1, FlowEngine::kMaxLevels);
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
    params.detailStrength = std::clamp(params.detailStrength, 0.0f, 1.0f);

    if (listWindows) {
        WindowHelper::printWindowList();
        return 0;
    }

    if (d3d12SelfTest) {
        return runD3D12SelfTest(adapterPreference);
    }
    if (projectionSelfTest) return runProjectionSelfTest(adapterPreference);
    if (aiBlendMaskSelfTest) return runAiBlendMaskSelfTest(adapterPreference);
    if (aiRefinementSelfTest) return runAiRefinementSelfTest(adapterPreference);
    if (aiTemporalUiSelfTest) return runTemporalUiSelfTest(adapterPreference);
    if (uiRegionsSelfTest) return runUiRegionsSelfTest(adapterPreference);
    if (!uiDetectInput.empty()) return runUiDetectTest(uiDetectInput, uiDetectOutput, adapterPreference, gpuSettings.uiDetector);

    if (!aiFlowTestModel.empty()) return runAiFlowTest(aiFlowTestModel, adapterPreference);
    if (!depthTestInput.empty()) {
        return runDepthTest(depthTestInput, depthTestOutput, adapterPreference);
    }

    if (neuralGradientCheck) {
        return checkNeuralGradient(7, true) ? 0 : 1;
    }

    if (!neuralModelPath.empty()) {
        return trainNeuralFlowModel(positionalArgs, neuralModelPath, neuralEpochs);
    }

    if (flowEval) {
        if (positionalArgs.empty()) {
            std::cerr << "--gpu-flow-eval needs one or more <prefix> (for _a.png, _b.png, _flow.bin)\n";
            return 1;
        }
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runFlowLevelEval(positionalArgs, gpuSettings, adapterPreference);
    }

    if (fsrRealtimeTest) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runFsrRealtimeTest(positionalArgs, gpuSettings, adapterPreference, fsrDepthSpan, fsrSaveFolder,
                                  fsrCamera);
    }

    if (!sequencePrefix.empty()) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runOfflineGpuSequence(sequencePrefix, gpuSettings, adapterPreference);
    }

    if (neuralParityCheck) {
        if (positionalArgs.size() < 2) {
            std::cerr << "Error: --neural-parity-check needs <frame0.png> <frame1.png>\n";
            return 1;
        }
        return runNeuralParityCheck(positionalArgs[0], positionalArgs[1], adapterPreference);
    }

    if (!captureWindowQuery.empty()) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        RealtimeOptions options;
        options.windowTitle = captureWindowQuery;
        options.stopEventName = stopEventName;
        options.flow = gpuSettings;
        options.sourceFps = sourceFps;
        options.outputMultiplier = outputMultiplier;
        options.adapter = adapterPreference;
        options.fsrFrameGen = fsrFrameGen;
        options.fsrWindow = fsrWindow;
        options.fsrDepthSpan = fsrDepthSpan;
        options.fsrCamera = fsrCamera;
        options.latencyMinExtraMs = queues.latencyMinExtraMs;
        options.latencyMaxMs = queues.latencyMaxMs;
        options.frameLatency = queues.frameLatency;
        options.captureQueue = queues.captureQueue;
        options.vsync = queues.vsync;
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

    if (fsrFrameGenOffline) {
        gpuSettings.minRefineLevel = std::clamp(
            gpuSettings.minRefineLevel, 0, gpuSettings.pyramidLevels - 1);
        return runFsrFrameGenOffline(input0Path, input1Path, outputPath, groundTruthPath, gpuSettings,
                                     adapterPreference, previousFramePath, fsrDepth, fsrFrames);
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
              << "  Detail restoration:  " << (params.enableDetailRestoration ? ("enabled (strength: " + std::to_string(params.detailStrength) + ")") : "disabled") << "\n"
              << "  Color clamping:      " << (params.enableColorClamping ? "enabled" : "disabled") << "\n"
              << "  Target time (t):     " << t << "\n";

    FrameInterpolator interpolator(params);
    FlowField forwardFlow, backwardFlow;

    auto start = std::chrono::high_resolution_clock::now();

    Image output = interpolator.interpolate(
        frame0, frame1, t,
        flowOutputPath.empty() ? nullptr : &forwardFlow,
        nullptr
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

    return 0;
}
