# Motion Enhancer — Project Briefing

Hand-off notes for anyone (human or agent) starting work on this repo. Written
from a read of the code at commit `ef06184e` (2026-11-06, branch `main`).
Where the README and the code disagree, **the code and `CMakeLists.txt` win**;
known discrepancies are in section 10.

## 1. What it is

A **Windows-only C++17 real-time frame interpolator**, version 1.1.0
(Apache 2.0). It captures any window with Windows Graphics Capture (WGC),
estimates optical flow on the GPU (D3D12), and shows generated in-between
frames in a click-through overlay, either at a fixed 2x/3x/4x or up to the
display refresh rate.

Around that core:

- a Win32 owner-drawn GUI (pages: **Live Capture**, **Offline Interpolation**, **AI Model**) with an animated brand character and startup splash;
- offline GPU harnesses and self-tests built into the same executable;
- optional AI parts on ONNX Runtime + DirectML: a trainable flow network, MiDaS depth, a UI-region detector, a trained temporal-UI model;
- optional AMD FidelityFX parts: Optical Flow (SDK 1.1.4) and FSR 4 frame generation / upscaling (SDK 2.4.0);
- a Python trainer (`tools/ai-train`) for the AI flow and UI models, and an NSIS installer.

Executable: `motion_enhancer.exe` (CMake target `motion_enhancer`, MSVC x64,
`/O2 /fp:fast /W3 /permissive- /await`).

## 2. Size and layout

| Area | Lines | Notes |
|---|---|---|
| `src/` + `include/` | ~16k C++ | biggest: `RealtimeSession.cpp` 3.4k, `GuiApp.cpp` 3.0k, `FlowEngine.cpp` 1.5k |
| `shaders/` | ~6k HLSL | 41 files, all listed in CMake (no dead shaders) |
| `tools/`, `docs/tools/` | ~18k Python/PS/C++ | trainer, UI-detector and temporal-UI tooling, ~25 `test_*.py` |
| `docs/validation/` | 25 notes | dated measurement write-ups behind each design decision |

Other directories: `models/` (ONNX models + JSON metadata, ~69 MB),
`flow_tests/` (75 synthetic verification images), `packaging/` (NSIS script,
installer theme, licences), `third_party/` (see section 8), `build-project/`
(the single active build folder, tracked in git, see section 10).

## 3. Entry points (`src/main.cpp`, `main` at line 316)

| Invocation | Handler |
|---|---|
| no args / `--gui` | `runGuiApplication()` (`GuiApp.cpp`) |
| `--capture-window "<title>"` | `runRealtimeSession()` (`RealtimeSession.cpp`): the live overlay |
| `--list-windows` | `WindowHelper` |
| `a.png b.png out.png [opts]` | legacy CPU interpolator (`ZNCCMatcher` + `FrameInterpolator`, OpenMP) |
| `--gpu-offline f0 f1 out [--ground-truth gt] [--save-flow f] [--gpu-offline-prev p]` | the live GPU graph on two PNGs (`runOfflineGpuInterpolation`) |
| `--gpu-offline-sequence <prefix>` | temporal chain over `<prefix>_f###.png`, PSNR vs `<prefix>_m###.png` |
| `--gpu-flow-eval <prefixes…>` | per-pyramid-level flow evaluation |
| `--fsr-fg-offline`, `--fsr-realtime-test <frames…>`, `--fsr-upscale-test <frames…> --upscale-size WxH --fsr-save <dir>` | FSR 4 offline harnesses |
| `--ai-flow-test <model>`, `--depth-test in out`, `--ui-detect-test in out` | single-model checks |
| Self-tests | `--timeline-self-test`, `--projection-self-test`, `--ai-refinement-self-test`, `--ai-temporal-ui-self-test`, `--ai-blend-mask-self-test`, `--ui-regions-self-test`, `--d3d12-self-test`, `--test` (synthetic CPU benchmark) |
| Neural estimator | `--train-neural <model.bin> <inputs…>`, `--neural-epochs`, `--neural-gradient-check`, `--neural-parity-check f0 f1` |

Main live options (defaults in brackets):

- `--flow-engine dis|ai|ffx|sad|neural` [dis]; `--adapter auto|igpu|dgpu` [auto]
- `--source-fps auto|24|30|60` [auto]; `--multiplier 2|3|4|max` [max]
- `--vsync on|off` [on]; `--frame-latency 1|2` [1]; `--capture-queue 2..8`; `--latency-min <ms>`, `--latency-max <ms>|auto`
- quality toggles: `--no-ui-mask`, `--no-ui-persistence`, `--no-occlusion-fill`, `--no-flow-median`, `--no-ffx-refine`, `--no-ffx-repair`, `--ffx-occlusion`, `--allow-crossfade`
- AI: `--ai-refinement` / `--no-ai-refinement`, `--ai-blend-mask`, `--ai-temporal-ui`, `--ai-ui-regions`, `--ui-detector`, `--ai-depth`, `--ai-depth-every 1..4`
- FSR: `--fsr-fg`, `--fsr-window` [0.1], `--fsr-depth-span` [0.25], `--fsr-camera off|flow|exact|spin`, `--upscale off|fsr4`, `--upscale-sharpness`
- `--stop-event <name>`: graceful stop, used by the GUI

The GUI runs capture and training as **separate processes** and stops capture
through the named stop event.

## 4. Real-time architecture

```
WGC (D3D11 surface, timestamp)
  -> CaptureEngine (producer thread): slot ring, GPU luma diff (duplicate rejection, scene cut)
  -> FrameTimeline: cadence classification, timestamp regularisation
  -> FlowEngine (flow queue): pyramid -> DIS | AI+DIS | FFX | SAD | neural -> post passes -> flow ring + fence
       (optional DepthService on its own queue; UiPersist/UiMask evidence on the flow queue)
  -> FrameTimeline (media clock, latency budget) picks the pair and t for each output frame
  -> Presenter (present queue): motion projection -> InterpolateCS -> [FSR FG / FSR upscale] -> flip swap chain -> overlay
```

Capture, flow and presentation run on separate threads and GPU queues and
meet only through fences: nothing on the way to the screen waits for the flow,
the depth network, or a save.

Components (`src/` + `include/`):

- **GraphicsDevice**: D3D12 device with `presentQueue` and `flowQueue`, plus a D3D11 device on the same adapter used only because WGC needs `ID3D11Texture2D`. Shared NT-handle textures and shared fences order D3D11 -> D3D12. Holds `ShaderConstants`, which **must match `shaders/Common.hlsli` byte for byte** (`static_assert`ed).
- **ComputeKernels / DispatchContext**: one root signature for every shader; `DispatchContext` records barriers, descriptors, constants and the dispatch per call. Shared capture textures are simultaneous-access and read through implicit promotion (deliberately no cross-queue barriers).
- **CaptureEngine**: the WGC callback only enqueues; a producer thread owns the D3D11 immediate context. Slots go `Free -> Filling -> Live -> Retiring` and recycle only after every queue's fence has passed them.
- **FrameTimeline** (100 ns units): cadence lock from the mean of the middle half of the last 16 intervals (`kCadenceWindow`), deferral/replace/drop of off-grid frames, timestamp regularisation, media clock `m = T_display - captureOffset - latencyBudget`, latency budget that grows on starvation and decays on slack. With vsync off it also schedules present times itself (`nextPresentTime`).
- **FlowEngine**: `submitPair(prev, cur)` runs on the capture thread and records the pair's flow into a ring of entries (forward and backward fields, plus optional UI evidence, AI blend mask and per-pair diagnostics). Per-slot cached pyramids; up to 10 levels (`kMaxLevels`), never below a 4 px short side. `setInterpolationConstants()` (`FlowEngine.h:463`) is the **only** correct way to fill block size, time and field-convention constants.
- **Presenter**: 3-buffer flip-model swap chain. Paced by `DCompositionWaitForCompositorClock` when available (the overlay is a layered, composed window whose frame-latency waitable was seen to signal between refreshes, so that object is only back-pressure). One command list per refresh, GPU-side fence waits only. Keeps a replay buffer in video memory for `Ctrl+Alt+F2` (last 20 shown frames + the game frames they came from, saved to `captures\<date-time>\` next to the exe). Debug views 0-7 cycle on `Ctrl+Alt+F3`.
- **RealtimeSession**: wires everything together; MMCSS thread boosts, async log, overlay window, hotkeys (`Ctrl+Alt+F1` show/hide, `F2` save, `F3` debug view, `Ctrl+Alt+Esc` quit), status line every 2 s. The same file also holds every offline harness and self-test (~18 functions).

## 5. Flow engines

Engine choice is stored as booleans in `FlowSettings` (`preferFfxOpticalFlow`,
`disFlow`, `aiFlow`, `neuralFlow`, `aiRefinement`), not an enum; `--flow-engine`
sets them together (`main.cpp:567`). `ai` also sets `disFlow`, because DIS is
its fallback and its refiner.

| Engine | Default | How it works |
|---|---|---|
| `dis` | **yes** | Dense Inverse Search (Kroeger et al. 2016): 8x8 patches every 4 px (`kDisStride`) at every level, 12 inverse-compositional Lucas-Kanade steps each, mean-normalised SSD after a photometric gain correction (`Photometric.hlsli`); `DisDensifyCS` makes a per-pixel field weighted by how well each patch explains the pixel. Ring grid 2 px. README: 45.7 dB vs SAD 44.6 on the rendered pairs, 48.3 vs 37.5 on the CS2-style sequence. |
| `ai` | no | "AI pyramid": a network trained with `tools/ai-train`, exported to `models/ai_flow.onnx` + `.json`, run by ONNX Runtime/DirectML on the flow queue (`AiFlowEstimator`). Its field seeds DIS at level 4 (`kAiSeedLevel`) and DIS refines it down to level 1 with 6 steps per patch, then a 5x5 median (`FlowMedian5CS`). `--ai-refinement` instead uses the network's integrated refiner (`ai_flow.onnx`, no DIS). Falls back to DIS without a model. |
| `ffx` | no | AMD FidelityFX Optical Flow 1.1.4: one current->previous field, then optional variational refinement to a per-pixel half-res field (`FfxRefine*`), neighbour repair (`FfxRepairCS`) and a 3x3 median. Needs priming after reset. If FFX can't start it falls back to DIS (with refinement on, the default) or SAD (`FlowEngine.cpp:439`). |
| `sad` | no | the original block matcher (`MotionSearchCS`): scalar SAD over 8x8 blocks, 11 candidate centres + one window, parabolic sub-pixel, `RankKey` tie-break; coarse radius 4, fine radius 6. |
| `neural` | no | small learned-feature cost volume inside `MotionSearchCS` (weights in the constant buffer, model magic `MNFN`). README: does not beat SAD; limit is capacity. |

Known limit of the AI engine (measured 2026-11-04): it finds only 0.42-0.74x
of motions under ~2 network px (about 6 px at 1440p). With integrated
refinement this shows as a 3.1-3.3x jump at the mid-pair hand-over; AI + DIS
refinement brings it to 1.10x (plain DIS 1.05x). A fine-tune on 1440p footage
and a `--subpixel` training objective did not fix it; the latter was removed.

## 6. Synthesis and its helpers

- **Motion projection** (`ProjectClearCS`, `ProjectSplatCS`, two-field pairs only): every block of both fields splats itself to its position at time t with an atomic min over a priority key (round-trip inconsistency + brightness error; with AI depth, nearness too). This decides occlusion without a depth buffer, as FSR 3 FI and Softmax Splatting do.
- **InterpolateCS** + **Synthesis.hlsli**: trajectory-aware warp from both frames (fixed-point solve `p = x - t*F(p)`), time blend. No per-pixel vector reselection; two field conventions selected by `FlowScale` (0 = independent forward/backward fields on their own grids; non-zero = FFX single negated field).
- **OcclusionFillCS** (DIS, once per pair): blocks next to moving objects whose vector fails the round trip get a confirmed neighbour's vector, the visible surface's or the background's.
- **Static UI**: `UiPersistCS` learns where structure stays fixed while the scene moves (no model, 4x4 cells); `UiMaskCS` runs the exact warp at t = 0.5 and marks pixels the warp would damage far more than they changed, confirmed over several pairs. InterpolateCS shows those pixels in place and drops warped samples that land on them. Optional detector boxes (`--ai-ui-regions`) and the trained temporal-UI model (`--ai-temporal-ui`) only *add* candidates; native pixel evidence is still required.
- **AI depth** (`DepthService` + `DepthEstimator`, MiDaS v2.1 small fp16): per-frame nearness on the GPU, every n-th frame; used by the projection and by `DepthDilateCS` at outlines. A frame whose depth is not ready is interpolated without it.
- **AI blend mask** (`--ai-blend-mask`): the network's learned midpoint preference, experimental.

## 7. FSR 4 (FidelityFX SDK 2.4.0)

Always **beside** the native interpolation, never replacing it (owner's rule).

- **Frame generation** (`FsrFrameGen`, `FsrRealtime`, `FsrMotionCS`, `FsrDepthCS`, `FsrCameraCS`): our backward field is FSR's motion-vector input. FSR 4.0.2 makes only the t = 0.5 frame, so live (`--fsr-fg`) its frame is used only on refreshes with |t - 0.5| <= `--fsr-window`; it covers every generated frame only at `--multiplier 2`. Offline: about +1.4 dB over InterpolateCS on 16 triplets, slightly softer. Off by default.
- **Upscaling** (`FsrUpscaler`, `FsrScaledMotionCS`, `--upscale fsr4`): runs last in Presenter. Offline it only beat bilinear and was below Lanczos (captured frames carry no camera jitter). Experimental.
- Neither path applies the static-UI mask to FSR's output yet.

## 8. Build, packaging, dependencies

```powershell
cmake -S . -B build-project -A x64
cmake --build build-project --config Release --target motion_enhancer --parallel
# -> build-project\Release\motion_enhancer.exe
```

- **FFX Optical Flow is ON by default**; configure fails unless `third_party/FidelityFX-SDK-v1.1.4` has its `CMakeLists.txt` and `tools/binary_store/FidelityFX_SC.exe`. Disable with `-DMOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW=OFF`.
- **FSR 4** builds in automatically when `third_party/FidelityFX-SDK-v2.4.0/Kits/FidelityFX/api/include/ffx_api.h` exists (defines `MOTION_ENHANCER_FSR_FG`, copies the signed loader/framegen/upscaler DLLs).
- **ONNX Runtime 1.24.4 + DirectML 1.15.4** (`third_party/ai-depth`): built in when `lib/onnxruntime.lib` exists (defines `MOTION_ENHANCER_AI_DEPTH`); `onnxruntime.dll` is **delay-loaded**, so the program starts without it. The AI flow, depth and UI detector all depend on this build.
- **UI matrix detector** (`-DMOTION_ENHANCER_ENABLE_UI_MATRIX=ON`, default OFF): pinned ncnn (Vulkan cooperative matrix) from `third_party/ui-matrix`.
- **DXC** is found with `find_program` (FFX SDK's bundled DXC, then `$VULKAN_SDK/Bin`); set `-DDXC_EXECUTABLE=` otherwise. Shaders compile to `<build>/shaders/*.cso`; a POST_BUILD step on the shader target copies them to `Release\shaders` so an HLSL-only edit is never measured with stale `.cso` files. A scratch build with a different `OutDir` does **not** get the shaders copied: copy them by hand.
- `app.rc` embeds the icon, logo and the 33 MB `character-atlas.png` (most of the 39 MB exe).
- **Installer**: `tools/Build-Package.ps1` (needs an FFX-enabled build and `makensis`) -> `dist/MotionEnhancer-1.1.0-…`; install rules in `cmake/Install.cmake`. `tools/Clean-Generated.ps1` removes only obsolete build trees.
- **AI trainer**: `tools/ai-train` (PyTorch; ROCm on AMD, CUDA or CPU), GUI `app.py` with Dataset/Train/Export/Label UI pages, setup via `Setup-AITrainer.ps1`; Colab notebook `colab_train.ipynb`. The current flow model is exported at 640x352.

Diagnostic environment variables (all `MOTION_ENHANCER_…`): `PACING`,
`D3D_DEBUG`, `DUMP_FRAMES=<prefix>`, `DEBUG_VIEW`, `BENCH_ITERS`,
`BENCH_SERIAL`, `SAD_DEBUG`, `FLOW_EVAL_DUMP`, `DEPTH_REPORT`, `SWITCH_WIDTH`,
`ALLOW_CROSSFADE`; model overrides `AI_FLOW_MODEL`, `AI_UI_MODEL`,
`NEURAL_MODEL`, `UI_MODEL`, `UI_MATRIX_MODEL`, `UI_BACKEND`; UI-detector
probes `UI_PROFILE`, `UI_RAW_OUTPUT`, `UI_DEBUG_DUMP`, `UI_TEST_*`,
`UI_MATRIX_TIMING|TRACE`; GUI capture/testing `GUI_PAGE`, `GUI_SNAPSHOT`,
`GUI_TIME`, `NO_SPLASH`, `SPLASH_FRAMES`, `SPLASH_TEST`, `LOGO_FRAMES`.

## 9. How changes are verified

There is **no unit-test framework and no CI**. Verification is:

1. the built-in self-tests (section 3), especially `--timeline-self-test` for pacing and `--projection-self-test` for occlusion;
2. PSNR on synthetic pairs and sequences (`--gpu-offline`, `--gpu-offline-sequence`; regenerate pairs with `tools/make_flow_tests.ps1`). Measure temporal behaviour on **sequences, not single pairs**, including motion that starts *and stops*;
3. offline runs on saved captures (`Ctrl+Alt+F2` output) with `--fsr-realtime-test` etc.;
4. the Python `test_*.py` scripts under `tools/ai-train`, `tools/ui-detect`, `tools/ui-temporal`;
5. dated notes in `docs/validation/` recording before/after numbers.

Working rules from the owner:

- **Offline only.** Never run live screen capture or the overlay to test; use the offline harnesses and simulations.
- **Don't stress the GPU.** It is shared with model training, and probing memory limits once crashed the machine. Report quality; treat timings as unreliable unless the GPU is idle (check `nvidia-smi` clocks/power state), and keep an unmodified build as a control.
- **Git is the save.** Commit to save; ask before untracking anything.

## 10. Gotchas and repo-health issues

1. **Repo size.** The pack is ~2.2 GiB. `build-project/` (1.7 GB) is tracked on purpose as the runnable build, including ~480 `.obj`/`.pdb`/`.tlog`/`.lib` intermediates and large DLLs; every rebuild adds another copy to history. Both FidelityFX SDKs and the ncnn sources (8k files) are vendored, plus a stray 6.7 MB `.mp4` in `third_party/`.
2. **Docs drift.** The README (~98 KB) mixes user guide and changelog. `CMakeLists.txt:144-146` still says SDK 2.4.0 is "not in git" (it is tracked since `ef06184e`), and the `Presenter.h` class comment still says pacing comes from the frame-latency waitable alone (it is the compositor clock, see `Presenter.cpp:172`).
3. **`RealtimeSession.cpp`** holds the live session (`runRealtimeSession`, ~520 lines in one lambda) *and* every offline harness. Prefer adding new harnesses elsewhere.
4. **Engine flags, not an enum** (section 5): impossible combinations can be represented; `usingNeural()` already guards against FFX.
5. **Cross-file invariants that bite:** `ShaderConstants` <-> `Common.hlsli`; `NeuralFlowLayout` <-> `kw*` constants <-> trainer <-> CPU reference; `Entry::singleField` <-> `FlowScale` <-> InterpolateCS convention; `Synthesis.hlsli` is shared by InterpolateCS and UiMaskCS so the mask tests exactly the warp that is shown; AI model `.onnx` <-> its `.json` (network size) <-> optional `.dis.onnx` companion.
6. **Choose-before-capture settings:** AI refinement, AI UI regions and the temporal-UI model bind tensors at session start; changing them needs a new capture session.
7. **Lost/unmerged work (as of 2026-11-07):** a later session made pacing fixes in `FrameTimeline` (trimmed-mean cadence over 96 intervals, `phaseTolerance`, slewed latency changes), per-position "own frame times" for fixed multipliers, FSR upscaler jitter, and replaced upscaling with FSR 4 native AA (`--aa fsr4`). **None of it is in this tree or in git** (no matching identifiers; reflog ends at `ef06184e`). It may exist only in the owner's `MotionEnhancer2.rar` backup. Check with the owner before redoing it.

## 11. Hard-won design decisions (do not undo without measuring)

- **Synthesis stays simple.** Per-pixel vector selection, per-pixel occlusion side choice and block splatting into colour were built, measured and removed; artifacts on real footage come from the flow. Occlusion is handled by projection priority and OcclusionFill instead.
- **DIS is the default** over SAD (section 5 numbers); AI + DIS refinement over integrated AI refinement for even motion.
- **DIS 5x5 median** after AI+DIS refinement: speckled flow pixels 6.9% -> 1.8%, roughness -36%, PSNR -0.02 dB, ~+0.2 ms.
- **SAD:** no displacement penalty, only a 1-luma-level tie-break bucket; coarse radius 4 / fine 6 (bar-chart page 18.8 -> 47.7 dB); the previous pair's vector is a prior only on axes the block can measure.
- **Trajectory-aware warp** (+0.7 dB) and the separate FFX field convention (+1.3 dB on pure shift).
- **UI regions are additive:** making detector boxes the *only* allowed places emptied the mask in games (0% of a synthetic HUD protected vs 19% without).
- **FSR stays optional and additive**, and only the middle frame comes from FSR.
- **Neural weights live in the constant buffer** (one flat array) to avoid parallel layouts.

## 12. Good next tasks

1. Recover or redo the 2026-11-07 pacing work (section 10.7) and add its 144/100 fps cases to `--timeline-self-test`.
2. A single offline regression script (`tools/Run-Checks.ps1`): the self-tests + `--gpu-offline-sequence` on `flow_tests/` with PSNR thresholds.
3. Split the offline harnesses out of `RealtimeSession.cpp`; replace the engine booleans with an enum.
4. Decide with the owner how much of `build-project/` to track (e.g. `Release/` only), and whether to shrink the embedded character atlas.
5. Split the README into a short user guide and a technical/changelog document.
6. Quality work should keep targeting flow (small-motion recall of the AI engine) and pacing, not synthesis.
