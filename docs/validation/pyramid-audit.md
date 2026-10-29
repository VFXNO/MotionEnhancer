# Live pyramid-depth audit

Archived validation summary. The one-off runners, generated images, raw logs
and machine-readable results mentioned below were removed during packaging
cleanup. The unresolved findings in this report have not been silently fixed.

## Verdict

The native live-flow implementation can allocate and execute ten levels, but
the change is not free of issues and does not make AMD FFX itself a ten-level
estimator. No application source was changed during this audit.

Tests ran against `build-project/Release/motion_enhancer.exe`, on an AMD Radeon
RX 9070 XT. There were 98 planned executable invocations: 97 exited normally
and one neural parity invocation crashed. Three additional completed GPU-based
validation runs emitted a resource-layout diagnostic. A fourth validation run
was interrupted by the outer command timeout and is not counted as completed.

## Confirmed findings

### 1. Neural CPU/GPU pyramid mismatch can crash the parity tool

- `include/FlowEngine.h:131` permits a four-pixel short side on the GPU.
- `src/NeuralFlow.cpp:303-305` still stops the CPU reference before going
  below an eight-pixel short side (`kBlock == 8`).
- `src/RealtimeSession.cpp:1248-1250` builds that shorter CPU pyramid, but
  `src/RealtimeSession.cpp:1360-1363` indexes it using the GPU level count.
- At 320x240, the GPU builds six levels while the CPU builds five.
- Reproduction with the repository's model:

  ```powershell
  $env:MOTION_ENHANCER_NEURAL_MODEL = "$PWD\neural_model.bin"
  .\build-project\Release\motion_enhancer.exe --neural-parity-check flow_tests\cel0.png flow_tests\cel1.png
  ```

- Observed exit code: `3221225477` (`0xC0000005`, access violation), immediately
  after the message introducing the per-level comparison.
- This is a regression from relaxing the GPU size cutoff. It is a reference /
  validation-path defect; the tested live-flow shader executions did not crash.
- The parity command also creates its own default `FlowSettings`, so a CLI
  `--gpu-levels` argument is not applied to this check.
- Recommended fix: share the native pyramid-size policy between CPU and GPU,
  validate matching counts before indexing, and allow explicit depth in parity.

### 2. FFX is still a fixed seven-level estimator

- `third_party/FidelityFX-SDK-v1.1.4/src/components/opticalflow/ffx_opticalflow.cpp:231`
  fixes `OpticalFlowMaxPyramidLevels` at 7; line 535 selects seven iterations.
- `src/FlowEngine.cpp:368-370` passes resolution, not the native pyramid count,
  to the SDK context.
- FFX refinement uses native luma level 1 and brightness statistics, not a
  native coarse-to-fine search over all ten levels. Its DIS/SAD fallback can
  use the expanded native pyramid.
- The 4K translation test produced byte-identical FFX output with settings 8
  and 10. Raw FFX and the occlusion-enabled variant also ran successfully.
- Recommended fix: report native and SDK depths separately. Extending the
  SDK's algorithm is a separate change, not a matter of raising this cap.
- Similarly, the ONNX model has its own exported pyramid; the live setting
  affects its native DIS component, not the number of levels inside ONNX.

### 3. Finest-searched-level validation uses requested, not effective depth

- `src/main.cpp:523-524` clamps `minRefineLevel` to the requested depth.
- `src/FlowEngine.cpp:184-186` can subsequently reduce that depth for the image.
- `src/FlowEngine.cpp:54` disables search below `minRefineLevel`.
- An accepted configuration at 1080p, `--flow-engine sad --gpu-levels 10
  --gpu-min-refine 9`, builds only nine levels (indices 0-8), so every level
  receives zero search radius. The test still exits successfully.
- On `hd0/hd1`, that configuration scored 33.04 dB, versus 41.94 dB with
  the normal finest-level setting.
- Recommended fix: clamp or reject the setting after effective depth is known.

### 4. GPU-based validation is not clean, at either depth

- Ordinary CPU-side D3D12 validation emitted no diagnostics in the tested
  ten-level DIS, SAD, FFX and neural runs.
- GPU-based validation reported `Incompatible texture barrier layout`, with
  an interpolation-output UAV described as being in COMMON layout.
- Reproduced with DIS at eight and ten levels, FFX at eight levels, and the
  small self-test. The message is therefore not specific to ten levels.
- See `gpu_validation_dis_8.log`, `gpu_validation_dis_10.log`,
  `gpu_validation_ffx_8.log`, and `gpu_selftest_debug.log`.
- These runs returned success despite the diagnostic. The self-test's PASS
  message must not be treated as proof that GPU validation was clean.
- Root cause (resource-state handling versus driver/validation interaction)
  is not established by this audit and needs a separate investigation.

### 5. More levels are not an unconditional quality improvement

Requests above eight remain opt-in; both GUI and CLI still default to eight.
However, lowering the minimum coarse extent from eight to four changes the
effective default depth for smaller inputs too: 320x240 now builds six levels
instead of five, and 1280x720 builds eight instead of seven.

| Test | Setting 8 | Setting 10 | Notes |
| --- | ---: | ---: | --- |
| Generated 4K pan, DIS | 47.81 dB | 48.13 dB | Actual 8 vs 10 levels |
| Existing `big24`, DIS | 37.24 dB | 36.68 dB | Actual 8 vs 9 levels |
| Existing `txt`, DIS | 49.81 dB | 50.73 dB | Actual 8 vs 9 levels |
| Generated 4K exposure change, DIS | 40.74 dB | 40.62 dB | Actual 8 vs 10 levels |

The new coarse images may be smaller than the fixed 8x8 patch; shaders clamp
samples, so this does not itself cause an out-of-bounds access, but the patches
contain repeated edge pixels and provide weaker motion measurements.
Also, `statsLevel()` depends on total depth, so changing depth changes the
image used to estimate brightness statistics, not just the motion search.

## Checks that passed

| Input | Requested | Observed native levels | Coarsest image |
| --- | ---: | ---: | --- |
| 1280x720 | 10 | 8 | 10x5 |
| 1920x1080 | 10 | 9 | 7x4 |
| 2560x1440 | 10 | 9 | 10x5 |
| 3840x2160 | 10 | 10 | 7x4 |
| 4097x2049 | 10 | 10 | 8x4 |
| 3072x2047 | 10 | 9 | 12x7 |
| 3072x2048 | 10 | 10 | 6x4 |

- Resource arrays use `kMaxLevels`; no remaining eight-entry bound was found
  in the active native live pipeline. Top-level parent bindings are guarded.
- Downsampling and vector scaling consistently use factors of two, including
  odd-sized levels. Tested true ten-level odd-sized inputs completed.
- Cold-pair dispatch upper bounds, including UI and depth input work, are
  about 85 for SAD/neural and 69 for DIS, below the 146-dispatch descriptor
  capacity and 160 constant-buffer slots.
- Native DIS, SAD, neural, AI/ONNX, refined FFX, raw FFX and FFX occlusion
  completed at a native ten-level size with their intended engines active.
- Eight-frame sequences at 3072x2048 exercised motion start, stop, reversal
  and stop again, at settings 8 and 10. No process failure or cumulative
  first-to-last drift was observed on these particular sequences.
- DIS and FFX exactly preserved the tested static 4K input. SAD and neural
  introduced small static-image errors, but the same errors were present at
  eight levels, so these are not new ten-level regressions in this test.

## Limits

### Follow-up: shared synthesis has a resolution-dependent index overflow

After the cross-engine artifact report, review of `shaders/Synthesis.hlsli:135`
found that motion projection reserves only 21 bits for a source block index.
`shaders/ProjectSplatCS.hlsl:70-71` ORs the unbounded source index into the
priority key, and `shaders/Synthesis.hlsli:181-182` masks it back to 21 bits.
With a 2x2 flow grid, captures exceeding 2,097,152 blocks therefore alias source
indices and may corrupt the priority as well. There is no size guard here.

3840x2160 fits (2,073,600 blocks), but 4096x2160, 3840x2400, 5120x2160 and
5120x2880 exceed the limit. The audited 4097x2049 case also slightly exceeds
it: its successful exit and aggregate PSNR did not establish correctness of
the affected bottom rows. Native flow-array bounds remain a separate check.

This affects the projected two-field synthesis path (DIS, AI, neural, and
FFX with two-field occlusion enabled). Default single-field FFX bypasses it.
It is a confirmed encoding defect, but is not established as the cause of the
user's cross-engine symptom without the actual capture resolutions. It cannot
explain a 1080p-to-1440p regression by itself.

**Subsequently fixed:** the shared encoder/decoder now derives index width
from the grid. The new `--projection-self-test` reproduced the old failure
and passes through 8K after the fix. See [projection fix](projection-fix.md) for
the build, replay comparisons and remaining limits.

These tests replay images through the shared live GPU flow and synthesis
code. They are not a long-running WGC/overlay capture test, nor a guarantee
for all footage, GPUs, settings, memory pressure, resize events, or timing
conditions. Single-pair GPU times are not used as performance conclusions.

Test runners: `../pyramid_audit.py` and `../pyramid_audit_extra.py`.
Machine-readable results: `results.json` and `extra-results.json`.
