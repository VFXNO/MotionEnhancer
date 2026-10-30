# Integrated motion refiner — implementation and validation

> Historical measurement: this initial implementation retained DIS after the
> integrated network. The later [AI/DIS switch validation](ai-dis-refinement-toggle-2026-10-30.md)
> adds mutually exclusive refinement paths and separate new measurements.

## Scope and preservation

Implemented an **opt-in** integrated flow-only refinement variant, not an RGB
frame generator or a second runtime network. The original `pyramid-v2` and
`pyramid8-v1` checkpoint formats remain supported. The new architecture is
`pyramid-refine-v1`; existing runs cannot silently change architecture.

Per request, **training stays as it was**: triplets and original middle targets,
augmentations, reconstruction/endpoint/smoothness losses and their weights,
AdamW, scheduler and existing flash handling. No offline teacher, flow-label
generation, dataset rebuild or production training run was introduced. The
previously identified gap/source sampling issues are not changed by this pass.

The native application still generates intermediate frames with its own GPU
pipeline, including existing DIS correction. It has not been replaced by a
standalone frame-generation model, and DIS was not removed merely to improve
a benchmark number.

All experiments use isolated temporary files. Nothing was installed into
`C:\MotionEnhancer`, and the installed checkpoints/models were not replaced.
Repository executable rebuilds add diagnostic measurement paths, not a new
live synthesis algorithm. No commits or pushes.

## Architecture

Only the finest block receives the new head:

1. Reuse its existing half-resolution features; pixel-shuffle to full network
   resolution (8/10/12 channels for Small/Medium/Large).
2. Concatenate those features with the existing block input and residual.
3. A 16-channel 1x1 mix, 16-channel depthwise 3x3 filter, and 1x1 output head
   predict corrections to the four directional flow components.
4. `tanh` bounds each correction to +/-2 **network** pixels. The original
   blend-mask output is unchanged by the head.

No additional image warp, second encoder, extra pyramid output/loss, temporal
state, RGB output or external model is required. This is a local correction,
not a replacement for the coarse pyramid's large-displacement search.

| Preset | Original parameters | Added | Refined total |
|---|---:|---:|---:|
| Small | 294,516 | 724 | 295,240 |
| Medium | 715,932 | 756 | 716,688 |
| Large | 1,616,452 | 788 | 1,617,240 |

The head starts with zero output. An explicitly initialized legacy model
therefore starts with **exactly the same PyTorch outputs**, rather than
immediately perturbing learned motion. This is not a quality improvement until
trained. A small parameter count also does not make full-resolution operations
free.

## Training and checkpoint interface

- GUI: **Integrated flow refinement**, plus optional **Initial weights**.
- CLI: `--refiner`; `--init-from <checkpoint>` optionally copies same-size
  model weights into a **new** run. Optimizer/schedule/validation restart.
- `--resume` and `--init-from` are mutually exclusive. Resume restores the
  saved architecture even if the checkbox/flag differs.
- Checkpoints and ONNX metadata record the actual architecture. A missing
  refiner, wrong-size initializer or mismatched best/last architecture fails
  without overwriting saved checkpoints.
- ONNX retains one FP32 `frames` input and FP32 `flow`, `mask` outputs, with
  FP16 network math and FP32 sampling coordinates.
- The refiner remains experimental/off by default. No realtime or general
  quality advantage is assumed.

## Test environments

Tested on the existing workstation/environment, not a clean-machine install:

| Component | Actual environment |
|---|---|
| OS | Windows 11, build 26200 |
| CPU | Ryzen 5 5600X, 6 cores / 12 threads |
| GPU | Radeon RX 9070 XT, 16 GB |
| GPU driver | 32.0.32015.7029, dated 2026-10-18 |
| Python | 3.14.7, 64-bit |
| PyTorch | 2.14.0+rocm10.1.0 |
| HIP | 7.16.26385 |
| ONNX / OpenCV | 1.23.2 / 5.0.0 |
| Native ORT DLL | 1.24.20260316.9.2d92497 |
| DirectML DLL | 1.15.4 |

Environments exercised: CPU FP32, off-screen Qt, ROCm FP16 training/inference,
opset-18 FP16 ONNX export/checker, and native DirectML/D3D12. No new dependencies,
drivers, external weights or performance/power settings were installed.
CUDA, Intel/NVIDIA GPUs, other AMD GPUs, clean installation and concurrent-game
load are **not tested**. The existing MIOpen tuning-database warning is nonfatal.

## Regression and learning controls

- Original trainer suite: **33 passed**.
- Refiner suite: **24 passed**, including repeated checkpoint/resume/flash
  regressions, worker-independent exact continuation, new-run initialization,
  architecture guards, odd shapes, all presets, finite gradients and GUI options.
- C++ Release build succeeded; only the existing `/await` deprecation warning.

The GPU learning control trains fresh, identically initialized baseline/refined
Small and Medium networks for **60 steps**, batch **4**, crop **224x128**, using
the original loss and optimizer/schedule. It includes textured translations,
independently moving foreground/occlusion and original-target return flashes.
The held-out synthetic textures are separate. No optical-flow teacher or
ground-truth flow loss is used. This is a sanity experiment, **not** a full
training run or a quality benchmark of the user's games.

| Control | Baseline PSNR | Refined PSNR | Baseline step ms | Refined step ms |
|---|---:|---:|---:|---:|
| Small | 21.2642 | 21.2632 | 45.11 | 48.83 |
| Medium | 21.2001 | 21.2461 | 49.50 | 52.74 |

Both start at 20.4914 dB, the blend baseline. Both refiner heads learned finite,
nonzero weights. Step means exclude the first five steps, data generation and
host-to-GPU copy, but include forward/backward, clipping, optimizer and scaler.
The added head cost approximately **8.2% Small / 6.5% Medium** in these controls.
That is not an estimate for the user's batch-16/crop-480 production training.
The quality differences are negligible/mixed and do **not** demonstrate a
general trained-quality gain.

### ROCm continuation and DirectML numerics

The real trainer (not a substitute optimizer loop) stopped a refined Small
run at step **2/8** and continued it to **8/8**. Model weights, optimizer,
scheduler, scaler, RNG and schedule all matched the uninterrupted control
**exactly**. FP16 scale remained 65,536 and the growth tracker finished at 8.

All **10 ONNX exports** passed checker, DirectML execution, output-finiteness
and mask-range checks. They cover baseline/refined Small and Medium at
640x352 and 512x288, plus the real trained Large checkpoint and its
zero-correction refined copy at 640x352. Native FP32 input/output dumps were
compared with the exported FP16 PyTorch graph on identical real-frame input.

Worst statistics across those ten comparisons:

- Mean absolute flow error: **0.01405 network pixel**.
- 99th-percentile absolute flow-component error: **0.0625 pixel**.
- Maximum absolute flow-component error: **0.33594 pixel**.
- Mean absolute mask error: **0.000583**; maximum **0.015625**.

All pass the declared cross-provider FP16 tolerances (mean flow <=0.1 pixel,
p99 <=0.5 pixel, mean mask <=0.005). These are numerical-compatibility checks,
not flow accuracy against real-world ground truth.

The first development run of the new tensor-dump diagnostic used a command
list type that did not match the application's direct flow queue and failed
with device-removed status. The diagnostic now creates its allocator/list
using the actual queue type. The complete rerun passed; `gpu-check-v2` contains
the successful results, while the initial `gpu-check` logs are retained.

### Real native outputs and known motion

All **84 selected native midpoint runs** passed: six AI configurations plus
DIS, each on the same 12 held-out cases, including dedicated flash sources.
No silent AI-to-DIS fallback occurred. These pairs use dataset-resolution
images with `t=0.5` and UI masking off, not original full-resolution captures.

The real Large checkpoint and its zero-correction refined copy produced
**pixel-identical native PNGs on all 12 cases**. This confirms preservation on
initialization; it is deliberately not a trained-refiner quality gain.

On the nine selected moving cases, native mean PSNR was 29.0801 dB for both
legacy configurations, versus 28.4779 dB for DIS. This small diagnostic subset
is different from the previous checkpoint test's selected examples; its means
must not be presented as a new whole-dataset quality improvement.

Six exact translations were also evaluated for each of the ten exported
configurations (**60 controls**). The trained legacy model and initialized
refiner matched throughout. The 60-step Small/Medium controls remain clearly
undertrained (including roughly 11–13 pixel error on one static control), so
their generated weights are **test artifacts, not deployment candidates**.
Neither these controls nor the native examples establish an improvement from
a fully trained refiner.

### Native GPU regression checks

The isolated diagnostic executable also passed:

- `--d3d12-self-test`.
- `--projection-self-test`: nine configurations, including 1440p, 4K, 5K,
  odd dimensions, 8K, block sizes 2/8 and saturated projection priority.
- `--ai-blend-mask-self-test`, with the refined ONNX active: mask copy,
  per-pair lifetime, enable/disable, ring reuse, UI protection and visibility.

## Latency measurements and the revised 4 ms maximum

The user subsequently relaxed the original **2 ms** limit to **4 ms maximum**.
This changes the acceptance threshold, not the model, recorded times or the
meaning of maximum. The raw native diagnostic retains its historical 2 ms
counts; `benchmark_latency.py --budget-ms 4` reports the current threshold.
Five additional stdlib-only acceptance tests passed, including inclusive exact
limits, invalid samples/budgets and rechecking saved samples without GPU work.
The combined CPU/UI/reporting regression count is now **62 passed**.

The initial sweep tested 10 exported networks and both output resolutions.
Because timing varied strongly, the repeat measurement runs without a resident
PyTorch GPU context and reverses baseline/refiner order on alternate rounds:

- Small/Medium, baseline/refined, 640x352/512x288; plus DIS.
- 1920x1080 and 2560x1440 output textures.
- **3 rounds x 100 measured pairs = 300 samples per configuration**, after
  20 warm-up pairs per invocation; 54 invocations / 5,400 measured pairs total.
- Fresh frame IDs rebuild one cached luma pyramid per pair, rather than timing
  only the previously cached images. The two images alternate as a chain.
- The enlarged dataset images are **timing proxies**, not full-resolution
  ground-truth quality samples. No concurrent game load was introduced.

The table is **GPU flow-stage elapsed time plus GPU midpoint-synthesis time**.
Flow includes input conversion, network and existing DIS correction. It does
not include capture, initial upload, readback, disk IO, presentation or scanout.
GPU timestamp spans can include queue-idle gaps during host submission; they
are not an isolated sum of active neural-kernel durations. Serial wall-time
statistics, also recorded, include additional CPU/fence overhead.

| Refined model | Network | Output | Mean ms | Median ms | p95 ms | Max ms | >4 ms / 300 |
|---|---|---|---:|---:|---:|---:|---:|
| Small | 640x352 | 1080p | 3.180 | 2.214 | 8.705 | 10.790 | 60 |
| Small | 640x352 | 1440p | 3.967 | 2.970 | 9.230 | 11.366 | 75 |
| Small | 512x288 | 1080p | 3.164 | 2.200 | 8.496 | 18.611 | 64 |
| Small | 512x288 | 1440p | 3.591 | 2.722 | 9.097 | 11.326 | 56 |
| Medium | 640x352 | 1080p | 3.230 | 2.248 | 8.614 | 10.596 | 60 |
| Medium | 640x352 | 1440p | 4.176 | 3.084 | 9.252 | 10.842 | 84 |
| Medium | 512x288 | 1080p | 3.060 | 2.205 | 8.535 | 10.377 | 54 |
| Medium | 512x288 | 1440p | 6.392 | 3.723 | 15.877 | 22.118 | 144 |

Reference measurements show that the long tails are not exclusive to the new
refiner:

| Original/reference | Output | Mean ms | Median ms | p95 ms | Max ms | >4 ms / 300 |
|---|---|---:|---:|---:|---:|---:|
| Medium 640x352, original | 1080p | 2.762 | 2.121 | 6.799 | 10.375 | 41 |
| Medium 640x352, original | 1440p | 3.858 | 2.857 | 9.242 | 12.652 | 67 |
| DIS | 1080p | 2.777 | 2.617 | 3.119 | 8.935 | 8 |
| DIS | 1440p | 6.056 | 5.138 | 9.139 | 13.032 | 300 |

For Medium/refiner 640x352, serial wall medians were **2.850 ms at 1080p**
and **3.895 ms at 1440p**, with maxima **10.959 / 11.860 ms**. Even the GPU
stage sum misses 4 ms on **20% / 28%** of measured pairs. A lower resolution
does not consistently eliminate these tails. The test does not establish
whether GPU scheduling, clock state, host dispatch or another factor dominates
them; an active-kernel/queue trace has not been captured.

**Acceptance result: no tested configuration meets a strict 4 ms maximum.**
Median/best-case times cannot override measured misses. The integrated refiner
is therefore implemented and compatible, but remains experimental/off by
default, with no production deployment or claimed general quality improvement.
The next performance task is to investigate the existing pipeline's latency
tails and budget refinement/synthesis together, not add a teacher or change
the user's training losses.

The 4 ms report was computed from the existing 5,400 raw timing samples; it
required **zero new training steps or GPU benchmark runs** and preserved the
original measurement report. See `acceptance-4ms/report.json` and the original
`latency-repeats/report.json`.

## Final preservation check

All **14 protected SHA256 hashes were unchanged** after the tests: installed
best/last/config/settings and executable, repository trainer settings, dataset
index, and installed/repository runtime ONNX/JSON files. The pre-build repository
executable was also backed up. The fully tested diagnostic executable and its
DLLs/shaders are retained separately under `native-runtime` in the artifact
folder. No running application was closed to replace a locked build file.

## Reproduction and raw artifacts

Repository entry points:

```text
tools/ai-train/test_quality.py
tools/ai-train/test_refiner.py
tools/ai-train/benchmark_refiner.py
tools/ai-train/test_refiner_gpu.py
tools/ai-train/benchmark_latency.py
tools/ai-train/test_latency.py
```

See `tools/ai-train/README.md` for commands and measurement boundaries.
Raw checkpoints, ONNX exports, per-sample JSON records,
native logs, images and preservation hashes are under:

```text
C:\Users\bsher\AppData\Local\Temp\opencode\mefi-refiner-test-4w40jl6o
```
