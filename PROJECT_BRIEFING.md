# Motion Enhancer — Project Briefing (hand-off for Opus 5.5)

Generated from a full read of the repo at commit `c8424ca` (branch `main`). Where the README and the code disagree, **the code and CMakeLists.txt win**; discrepancies are listed in section 9.

## 1. What it is

A **Windows-only, C++17 real-time frame interpolator** with GPU motion estimation and intermediate-frame synthesis. It captures a window with Windows Graphics Capture (WGC), estimates block motion on the GPU, and presents an interpolated stream at the display refresh rate in a click-through overlay. There is also an offline mode (two images in, one interpolated image out), a native Win32 GUI, and a small trainable "neural" flow estimator.

- Executable: `motion_enhancer.exe` (CMake target `motion_enhancer`, VS 2022 x64, MSVC `/O2 /fp:fast /await`).
- Owner's GPU for benchmarks: RTX 3050 Ti laptop (timings are power-limit sensitive, see README).
- Not AMD AFMF; it only optionally uses AMD FidelityFX Optical Flow (SDK v1.1.4, vendored in `third_party/`).

## 2. Entry points (`src/main.cpp`, `int main` at line 195)

| Invocation | Path |
|---|---|
| no args / `--gui` | `runGuiApplication()` in `GuiApp.cpp` (Win32 owner-drawn dark UI; tabs: Live Capture, Offline Interpolation, Train Neural Flow) |
| `--capture-window <title>` | `runRealtimeSession()` in `RealtimeSession.cpp` (live overlay) |
| `--list-windows` | `WindowHelper.cpp` |
| `a.png b.png out.png [opts]` | CPU offline interpolator (`ZNCCMatcher` + `FrameInterpolator`, OpenMP) |
| `--gpu-offline f0 f1 out [--ground-truth gt] [--save-flow f] [--gpu-offline-prev p]` | runs the real-time GPU graph on two PNGs (`OfflineContext` in RealtimeSession.cpp) |
| `--gpu-offline-sequence <prefix>` | replays `<prefix>_f000.png…` as a temporal chain, PSNR vs `<prefix>_m000.png…` |
| `--test` | synthetic CPU benchmark with ground truth (+16,+8 px) |
| `--d3d12-self-test` | dispatch + shared-texture probe |
| `--train-neural <model.bin> <video/image…>`, `--neural-epochs`, `--neural-gradient-check`, `--neural-parity-check f0 f1` | neural trainer / verification |
| Live options | `--source-fps auto|24|30|60`, `--multiplier 2|3|4|max`, `--gpu-levels`, `--gpu-min-refine`, `--gpu-coarse-radius`, `--gpu-refine-radius`, `--gpu-smoothness`, `--flow-engine sad|ffx|neural`, `--adapter auto|igpu|dgpu`, `--stop-event <name>` (GUI graceful stop) |

The GUI launches capture and training as **separate processes** (`quoteArgument`, `trainModel`, `startCapture` in GuiApp.cpp) and stops capture via a named stop event.

## 3. Real-time architecture (the part that matters)

```
WGC (D3D11 surface, ts) -> CaptureEngine producer thread
   copy to slot ring, GPU luma diff (dup rejection + scene cut), cadence classify (FrameTimeline)
   -> FlowEngine (flow queue): luma pyramid -> SAD | FFX | neural -> 16-entry flow ring + fence
   -> FrameTimeline (media clock) -> Presenter (present queue): InterpolateCS -> flip swap chain -> overlay window
```

Components (all in `src/` + `include/`):

- **GraphicsDevice** — D3D12 device, two queues (`presentQueue`, `flowQueue`), a D3D11 device on the same adapter used *only* because WGC requires `ID3D11Texture2D`. Shared NT-handle textures and shared fences (`captureReady`, `computeDone`) order D3D11→D3D12. Holds `ShaderConstants` (1024 bytes, 64 rows of 16 B — **must match `shaders/Common.hlsli` exactly**; `static_assert`ed).
- **ComputeKernels / DispatchContext** — one root signature (t0..t5 SRV, u0 UAV, b0 CBV, s0 linear, s1 point) shared by all shaders; `enum class Kernel { Luminance, Pyramid, MotionSearch, Interpolate, PresentFrame, FfxFlowConvert, PackLuma, LearnedFeatures, FillFlow }`. DispatchContext records barriers+descriptors+constants+dispatch per call, with per-command-list regions of heap/CB ring. Shared capture textures are simultaneous-access, read via implicit COMMON promotion (no barriers across two queues — deliberate).
- **CaptureEngine** — WGC callback only enqueues; producer thread owns the D3D11 immediate context. Slot states `Free/Filling/Live/Retiring`; slots recycle only after all queues' fences pass. Env `MOTION_ENHANCER_DUMP_FRAMES=<prefix>` dumps first 8 accepted frames.
- **FlowEngine** (`FlowEngine.cpp`, 662 lines) — `submitPair(prev, cur)` runs on capture thread; ring of 16 `Entry` (forward/backward `R32G32B32A32_FLOAT` per 8x8 block: xy = vector in full-res px, zw = per-axis measured confidence). Per-slot cached pyramids (luma R16_FLOAT, packed R32_UINT, learned feature R16_FLOAT). Flow of each level is double-buffered; previous pair's flow at the same level is the temporal predictor. FFX path: single `current→previous` field on frame1's grid; `Entry::singleField=true`, forward is the negation. `setInterpolationConstants()` is the **only** correct way to fill block size/time/field-convention constants.
- **FrameTimeline** — cadence lock (mean of middle half of last 16 intervals, locks at 3/4 agreement), deferral/replace/drop of off-grid frames, timestamp regularisation, media clock `m = T_display − captureOffset − latencyBudget`, adaptive latency budget (grows on starvation, decays on slack). Times in 100 ns units.
- **Presenter** — 3-buffer flip-model swap chain, woken by `DCompositionWaitForCompositorClock` with a PLL on wake times; one command list per refresh; GPU-side fence waits only. `PresentFrameCS` for the hold/cut cases.
- **RealtimeSession** (1176 lines) — wires it all together, MMCSS thread boosts, async log, overlay window, hotkeys (`Ctrl+Alt+F1` toggle, `Ctrl+Alt+Esc` quit), status line every 2 s, offline GPU harness (`OfflineContext`).

## 4. Shaders (`shaders/*.hlsl`, compiled by DXC to `.cso` in `<build>/shaders`)

Built targets (CMake list): `Common.hlsli, LuminanceCS, PyramidCS, MotionSearchCS (cs_6_0, wave intrinsics), InterpolateCS, PresentFrameCS, PackLumaCS, LearnedFeaturesCS, FillFlowCS`, plus `FfxFlowConvertCS` when FFX is on. Others `cs_5_0`. DXC path hard-hints `C:/VulkanSDK/1.4.309.0/Bin`; if DXC is missing, a runtime `D3DCompile` fallback is mentioned.

- **MotionSearchCS** (940 lines, the core): one 8x8 thread group per 8x8 flow block. Plain **scalar SAD** over all 64 luma px (no msad4, no ZNCC, no magnitude penalty). 11 scored centres (coarse predictor, temporal, 8 neighbouring parents, zero) → one ±radius window around the winner → parabolic sub-pixel from 5 SADs. `RankKey` tie-break: within 1 luma level of best, prefer candidate nearest a reference vector (previous pair's vector only on axes the block can measure, else the coarse predictor). Radii: finest 6, coarse 4 (tapered 4/3 at top); keep every level's span ≤ 13 (cost is a step function of span due to 64 threads). Has a neural branch (cost volume over learned features, 6 descriptors → 8 ReLU → 1, softmax soft-argmax over a dense 9x9 window, anchors ranked on learned cost).
- **FillFlowCS** — runs after *every* pyramid level; fills un-measured per-axis components from confident neighbours (weighted by confidence, distance, and vector similarity). Fixed "77% of blocks measure nothing on text/chart pages".
- **LearnedFeaturesCS** — 3x3→6 ReLU, 3x3→2 + learned gain on the coarser level's features, tanh (5x5 receptive field, 238 weights, file magic `MNFN`).
- **InterpolateCS** — the entire synthesis: per pixel, trajectory-aware flow read (2 Picard iterations of `p = x − t·F(p)`), bilinear fetch from each frame, `lerp(c0,c1,t)`. **No occlusion mask, no confidence, no gating, no per-pixel vector selection.** Two field conventions selected by `FlowScale` (0 = independent fwd/bwd on own grids; ≠0 = FFX single negated field on frame1 grid).
- Shader constants hold the neural weights (`float neuralWeights[240]`, indexed via `kw*` constants mirrored in `NeuralFlow.h` namespace `NeuralFlowLayout` and `Common.hlsli`).

## 5. Neural flow (`NeuralFlow.cpp`, `NeuralFlowTrain.cpp`, `TrainingVideo.cpp`)

CPU reference inference, trainer (Adam, analytic backprop, Charbonnier EPE, cosine decay, per-level supervision with *detached* parent flow, synthetic piecewise-constant-flow data from images/video frames via Media Foundation), model file `neural_model.bin` (957 bytes, in repo root). Two self-tests keep the four implementations (CPU, trainer, shader, parity) consistent: `--neural-gradient-check`, `--neural-parity-check`. README is explicit: **neural is not default and does not beat SAD on the synthetic suite**; limit is capacity, not objective.

## 6. Flow engines

| Engine | Default | Notes |
|---|---|---|
| `sad` | yes | native 8-level pyramid, 8x8 blocks, ~2.9 ms @1080p |
| `ffx` | no | AMD FidelityFX Optical Flow v1.1.4; needs priming (SDK suppresses vectors through temporal frame 5 after reset); resets/re-primes when source texture changes; falls back to SAD on init/dispatch failure |
| `neural` | no | needs a validated `MNFN` model (`MOTION_ENHANCER_NEURAL_MODEL`) |

## 7. CPU offline path (legacy-ish but still wired)

`Image`, `Pyramid`, `ZNCCMatcher` (8-level pyramid, ZNCC, EPZS, Gaussian block weights, quarter-pel), `MotionVector.h` (flow field, colour-guided vector median, flow visualiser), `FrameInterpolator` (bidirectional flow, splatting, Laplacian detail restoration, colour-box clamp). OpenMP. This is what the README's headline text describes; the live path no longer uses ZNCC. The GUI "Offline Interpolation" tab drives it.

## 8. Build & tooling

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release        # -> build\Release\motion_enhancer.exe
```
- FFX is ON by default and **configure fails** unless `third_party/FidelityFX-SDK-v1.1.4/CMakeLists.txt` and `tools/binary_store/FidelityFX_SC.exe` exist. Disable with `-DMOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW=OFF`. Prebuilt SDK libs are checked in under `bin/ffx_sdk/`.
- Links d3d11/d3d12/dxgi/dwmapi/avrt/mfplat/mfreadwrite/windowscodecs/windowsapp etc.; `app.rc` + `app.manifest` embedded manually (`/MANIFEST:NO`).
- A POST_BUILD copy on the *shader target* ensures an HLSL-only edit still reaches `Release\shaders` (past trap: stale `.cso` made A/B measurements wrong).
- `tools/make_flow_tests.ps1` regenerates the 320x240 verification pairs (`flow_tests/`, `train_sources/` hold images).
- Diagnostics env vars: `MOTION_ENHANCER_PACING=1`, `_D3D_DEBUG=1`, `_DUMP_FRAMES=<prefix>`, `_NO_TEMPORAL=1`, `_SAD_DEBUG=1|3|5|6`, `_BENCH_ITERS=n`, `_NEURAL_MODEL=<path>`.
- No unit-test framework. Verification = PSNR on synthetic pairs (`--gpu-offline`, `--gpu-offline-sequence`), the neural gradient/parity checks, and eyeballing. **Measure temporal behaviour with `--gpu-offline-sequence`, not single pairs**; include sequences where motion starts *and stops*.
- Check GPU power state (`nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,enforced.power.limit,pstate --format=csv`) before trusting any timing; keep an unmodified engine as a control.

## 9. Repo state, gotchas and hygiene issues (important for any change)

1. **Dead code**: `D3D11Context`, `D3D12Context`, `GPUInterpolator`, `RealtimePresenter`, `WGCCapture` (`.cpp`/`.h`) are **not in CMakeLists** and referenced by nothing but each other (superseded by GraphicsDevice/CaptureEngine/Presenter/FlowEngine). Shaders `FilterFlowCS`, `UpscaleFlowCS`, `InvertFlowCS`, `BidirectionalRefineCS` are also not built/loaded. `include/MotionVector.h`/ZNCC code is live only for the CPU path.
2. **README is partly stale**: its directory tree omits most of the live files, the headline describes ZNCC (CPU path) while the live path is SAD, and it claims CMake uses `cs_5_0` for all shaders (MotionSearchCS is `cs_6_0`). It also shows mojibake in the tree (encoding).
3. **Build artefacts are committed**: ~3000 tracked files, including `build/`, `build-d3d12/`, `build-ffx/`, `build-vendored-ffx/` (objs, tlogs, `.exe`, generated FFX headers), `shared_probe.obj`, ~60 `gpu_*.png`/`output_*.png`/`ffx_*.png` debug images in the root, and several `*.png.txt` / `gpu_debug.txt` logs. **There is no `.gitignore`.** Cleaning this up (history rewrite is a separate, risky decision) is the highest-value repo-health task.
4. `MOTION_ENHANCER_DXC` hint path is machine-specific (`C:/VulkanSDK/1.4.309.0/Bin`).
5. Commit history is 5 commits (`init`, `added multiplayer`, `UI improveed`, `added amd optical flow sdk`, `jaja`); messages are uninformative. "multiplayer" does not correspond to anything visible in the current source tree.
6. Windows-only: D3D12, WGC (C++/WinRT, `/await`), DComp, MF. The non-WIN32 CMake branch is vestigial.
7. Cross-file invariants that bite: `ShaderConstants` ⇄ `Common.hlsli`; `NeuralFlowLayout` ⇄ `kw*` constants ⇄ trainer ⇄ CPU reference; `Entry::singleField` ⇄ `FlowScale` ⇄ InterpolateCS convention.

## 10. Hard-won design decisions (do not undo without measuring)

- No displacement penalty in the SAD cost; tie-breaking only inside a 1-luma-level bucket (removing the penalty without `RankKey` cost 12 dB @1080p).
- Coarse radius 4 / fine radius 6 (18.8 dB → 47.7 dB on bar-chart page; 4.67 → 3.06 ms @1080p); costs 4.6 dB on the periodic synthetic `circle` case — known and accepted.
- Previous-pair vector is a prior, used only on axes the block can measure (identical-frame test: 61.5% phantom-motion blocks, 37.5 dB, before the fix).
- Synthesis is deliberately simple. Per-pixel vector selection, per-pixel occlusion side choice, and block-vector splatting were all built, measured and **removed** (they fail on blown highlights / real footage). The remaining artifact source on real footage is upstream (flow), not synthesis.
- Trajectory-aware flow lookup in InterpolateCS (+0.7 dB), and field-convention split for FFX (+1.3 dB on pure shift).
- Neural weights live in the constant buffer (one flat array) to avoid parallel layouts.

## 11. Suggested first tasks for Opus 5.5

1. Add `.gitignore`, `git rm --cached` the build trees, root PNGs/logs/`.obj`; move test imagery into `docs/` or `flow_tests/`.
2. Delete (or move to `legacy/`) the dead D3D11Context/D3D12Context/GPUInterpolator/RealtimePresenter/WGCCapture and the four unused shaders after confirming no build config references them.
3. Rewrite README structure/overview to match the live architecture (this file is a starting point).
4. Add an automated regression script around `--gpu-offline-sequence` + `--neural-gradient-check` + `--neural-parity-check` with PSNR thresholds from the README table.
5. Remove the hard-coded VulkanSDK DXC hint (use `find_program` + env/cache var).
6. Quality work should target flow estimation (README evidence), not synthesis; neural engine needs more capacity and real-footage training before it can beat SAD.

## 12. File index (size in lines)

Core live path: `main.cpp` 558, `RealtimeSession.cpp` 1176, `GuiApp.cpp` 1440, `FlowEngine.cpp` 662, `CaptureEngine.cpp` 562, `FrameTimeline.cpp` 336, `Presenter.cpp` 274, `GraphicsDevice.cpp` 282, `ComputeKernels.cpp` 233, `NeuralFlow.cpp` 509, `NeuralFlowTrain.cpp` 992, `TrainingVideo.cpp` 298, `WindowHelper.cpp` 318, `shaders/MotionSearchCS.hlsl` 940.
CPU path: `Image.cpp` 213, `Pyramid.cpp` 36, `ZNCCMatcher.cpp` 410, `FrameInterpolator.cpp` 243, `MotionVector.h` 247.
Third party: `stb_image.h`, `stb_image_write.h`, `FidelityFX-SDK-v1.1.4/` (MIT; see `THIRD_PARTY_NOTICES.md`).
