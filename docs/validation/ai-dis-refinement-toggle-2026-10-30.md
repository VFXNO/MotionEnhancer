# Programme AI/DIS refinement switch — 2026-10-30

## Implemented behavior

On **Live Capture**, select **AI pyramid (trained model)**, then:

- **AI flow refinement (off: DIS) checked:** load the integrated graph and
  send its bidirectional motion directly to the 2x2 ring. **No DIS patch or
  densification refinement is dispatched.** The native synthesis remains.
- **Unchecked (default):** load the same jointly trained backbone with the
  integrated head physically removed, then run the existing DIS correction.
  The estimator is still AI; this is not pure DIS engine selection.
- **AI: blending mask** remains an independent synthesis setting.
- Choose before capture; stop/restart capture to change. The checkbox is
  disabled while the capture process owns the graph. This is not a live,
  mid-pair hot switch.
- Non-refined checkpoints retain DIS mode; they cannot gain learned refiner
  weights through a runtime checkbox. Their AI-refinement control is disabled.

No training loss, optimizer, dataset, augmentation or training target changed.
No teacher or new training run was used in this implementation pass.

## Export and compatibility

The corrected repository exporter automatically produces a pair for a refined
checkpoint:

```text
ai_flow.onnx        full integrated graph
ai_flow.json        primary metadata
ai_flow.dis.onnx    backbone-only graph; no refiner operators
ai_flow.dis.json    companion metadata
```

Both graphs use the same trained backbone weights and resolution. The bypass
is **not** a separately trained baseline, and its metadata does not claim the
full model's validation PSNR. The source checkpoint is never modified.

The loader validates matching export IDs, architecture, preset, resolution and
file presence. Missing/mixed pairs produce an actionable error, rather than
running the wrong path. The trainer's copy-to-builds path copies/renames all
four files, publishing primary metadata last. Old unrefined single-file exports
still load in DIS mode. Old **refined** single-file exports need re-exporting
from their existing `.pt` checkpoint, not retraining or checkpoint conversion.

Selecting AI refinement explicitly does not silently fall back to DIS. The
existing missing-model fallback remains for ordinary unchecked AI selection.

## Builds and preservation

The existing repository executable was locked by a running application. It was
not closed, replaced or reset. The updated Release executable was linked to a
separate temporary output folder and copied to a new repository preview:

```text
build-refinement-preview/Release/motion_enhancer.exe
```

Preview executable SHA256:
`5687b449acb1ce3365e56a5ad9a242cf6cacb911e11a392543b5fd92b87827d7`

The preview includes the tested DLLs/shaders but deliberately **no test or
production model**. Export the desired checkpoint into its `models/ai_flow.onnx`
before using AI. The preview's trainer launcher finds the repository trainer.
`C:\MotionEnhancer`, existing runs, checkpoints, dataset index and runtime models
were not replaced. All **14 protected file hashes** matched at the post-native-
test check. A later check found concurrent user changes to repository trainer
settings (with trainer processes present); those settings were left as-is.
The remaining **13 protected hashes** still matched. Settings had also changed
since the previous task; no older settings snapshot was restored.

Release build succeeded. Warnings: the existing `/await` deprecation, plus
MSBuild's warning about incremental builds with a temporary output directory.
No installer/package was rebuilt. All source changes remain uncommitted.
Newly dirtied tracked build intermediates were archived and restored to their
pre-request clean baseline; pre-existing build modifications were left alone.

## Environments and tests

Existing workstation: Windows 11 build 26200, Ryzen 5 5600X, RX 9070 XT 16 GB,
driver 32.0.32015.7029. Python 3.14.7, PyTorch 2.14.0+rocm10.1.0, HIP 7.16.26385,
ONNX 1.23.2, OpenCV 5.0.0, native ONNX Runtime 1.24 / DirectML 1.15.4.
No dependencies, models or drivers were downloaded/installed.

Tests:

- **70 CPU/UI/reporting regressions passed**: original trainer/resume/flash
  tests plus head-free extraction/export, exact backbone preservation, unchanged
  masks/checkpoints, paired copying/renaming, mixed-pair rejection, legacy
  exports and counterbalanced benchmark routing.
- Both actual opset-18 ONNX graphs passed checker and DirectML execution.
  Integrated graph: **368 nodes**; head-free graph: **354 nodes**, with no
  refiner initializer names. This bypass does not compute-and-discard the head.
- FP16 DirectML versus PyTorch on identical real-frame inputs:

  | Graph | Mean flow-component error (network px) | p99 | Mean mask error |
  |---|---:|---:|---:|
  | Head-free/DIS input | 0.0132695 | 0.0390625 | 0.0001343 |
  | Integrated AI | 0.0135627 | 0.0390625 | 0.0001343 |

  All pass the existing tolerances: mean flow <=0.1 px, p99 <=0.5 px, mean
  mask <=0.005. These are numerical checks, not true-motion accuracy.
- Native self-test: **40 pairs**, exceeding the ring size in both modes.
  AI mode had **zero DIS patch passes**, unchecked mode had positive DIS
  patch passes. Both checked finite forward/backward fields, static-UI output,
  independent blend-mask enable/disable and ring reuse.
- Actual **Win32 GUI** test: control unavailable for another engine, enabled
  for a valid pair, default unchecked, clicks toggle state, mismatched export
  disables/unchecks it, Refresh restores availability after repairing the pair.
  Only the test-owned GUI process was closed.
- **48 real held-out midpoint outputs**: 12 pairs x two refinement modes x
  blend mask on/off. No silent estimator fallback; routing logs agree with the
  requested mode. All outputs are finite and correctly sized.
- Both modes reject mismatched IDs and missing companion graphs. A legacy
  single-file export works with DIS and explicitly rejects AI refinement.

### Quality limitation

The refined checkpoint used here is the previously generated **60-step Small
synthetic control**, not a fully trained production candidate. No further
training occurred. Native mean PSNR over the 12 diagnostics was 20.9858 dB
for integrated-only correction versus 27.3229 dB for backbone+DIS (blend mask
off). **Removing DIS from these undertrained weights worsens quality.** This
validates distinct routing, not an AI quality upgrade. Keep DIS as the default
and train/test the integrated model properly before choosing it for playback.

## New exclusive-path latency measurement

Model: that same Small control at **640x352**. Output textures: 1080p/1440p
enlarged dataset images, **timing proxies**, not full-resolution quality truth.
Three rounds, 20 warm-up pairs per invocation, 100 measured pairs per mode and
output each round: **1,200 measured pairs** total. AI/DIS order reverses on
alternate rounds. No resident PyTorch GPU context or concurrent game workload.
Static-UI masking and AI depth are off for these timings.

| Correction mode | Output | GPU mean ms | p95 ms | Worst ms | >4 ms / 300 |
|---|---|---:|---:|---:|---:|
| Integrated AI | 1080p | 1.293 | 1.413 | 1.657 | 0 |
| Backbone + DIS | 1080p | 1.688 | 1.829 | 1.979 | 0 |
| Integrated AI | 1440p | 1.698 | 1.816 | 1.994 | 0 |
| Backbone + DIS | 1440p | 2.375 | 2.506 | 2.660 | 0 |

These times sum the GPU flow stage (conversion, selected network, DIS only
when selected) and native midpoint synthesis. Additional serial CPU/fence
wall-time maxima were **1.804/2.172 ms** for AI at 1080p/1440p, versus
**2.250/2.855 ms** for DIS mode. Capture, initial upload, readback, disk IO,
presentation and scanout are excluded from both measurements.

**All samples in this limited sweep were within 4 ms.** This is not a maximum
guarantee under gameplay load, for Medium/Large or for other hardware. Previous
measurements had significant tails; this run does not prove their cause or
elimination. The earlier integrated-refiner report measured the then-existing
AI+DIS stack and is historical, not a test of this new exclusive switch.

## Reproduction/artifacts

```powershell
tools\ai-train\.venv\Scripts\python.exe -m unittest discover -s tools\ai-train -p "test_*.py"
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_refinement_switch_native.py --out <new-folder> --executable <updated-exe> --checkpoint <refined.pt> --cases <benchmark-cases-folder>
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\benchmark_latency.py --out <new-folder> --executable <updated-exe> --models <paired-primary.onnx> --inputs <1080p-folder> <1440p-folder> --refinement both --rounds 3 --samples 100 --budget-ms 4
```

All raw logs, exported graphs, tensor dumps, real outputs, complete per-sample
statistics and preservation manifests:

```text
C:\Users\bsher\AppData\Local\Temp\opencode\mefi-refinement-switch-j6uljemi
```
