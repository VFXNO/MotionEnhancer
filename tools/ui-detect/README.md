# Optional AI UI-region detection

The AI detects **where UI might be**. The existing native `UiMaskCS` decides
**which pixels actually need protection**, only inside those regions. Detector
rectangles are never filled as solid protection masks. The native damage,
change, multi-pair confirmation and changing-UI release tests remain intact.
An AI false positive can still admit wrong native masking inside that box;
an AI miss means that UI receives no protection. This is experimental.

Independent of DIS/AI/FFX selection, AI flow refinement, AI depth and the AI
blending mask. No training, flow-model replacement or changed training losses.

## Optional pretrained model and licence

Uses the **OmniParser V1 YOLOv8 nano icon/UI detector**, the pinned
[ONNX Community 640x640 FP16 conversion](https://huggingface.co/onnx-community/OmniParser-icon_detect_640x640/tree/799bd041b5d053ed44651c2237ced04d8fdb2777).
It has float32 I/O and ~6.2 MB of weights. It was trained for interactable web
UI/icons, **not general game HUD segmentation**; text/subtitles are not covered
by a separate detector. The Text/HUD experiment has been removed because of
false detections. Inspect icon results on each game before use.

The detector weights are **AGPL-3.0**, not covered by this project's MIT licence.
See [Microsoft's detector-specific licence](https://huggingface.co/microsoft/OmniParser/blob/main/icon_detect/LICENSE).
Distribution/integration obligations need review before shipping this model;
do not relabel it MIT or silently include it in packages. CMake/packaging do
not automatically download or bundle these weights.

First test in a temporary folder, without touching runtime models:

```powershell
python tools/ui-detect/fetch_model.py --accept-agpl --out "$env:TEMP/ui-detector-test/ui_detect.onnx"
tools/ai-train/.venv/Scripts/python.exe tools/ui-detect/evaluate.py `
  --executable build-project/Release/motion_enhancer.exe `
  --model "$env:TEMP/ui-detector-test/ui_detect.onnx" `
  --out "$env:TEMP/ui-detector-test/evaluation" screenshot.png
```

The downloader verifies SHA256
`cbc393231545eee6155291c7e2527db3eca86ed54935f4155556113344e24d00`,
retains upstream licence/provenance alongside the model and refuses to replace
a different existing file. It does not use pickle or execute downloaded code.
No screenshots leave the machine.

After reviewing accuracy/licensing, install **only the new UI detector**:

```powershell
python tools/ui-detect/fetch_model.py --accept-agpl --out build-project/Release/models/ui_detect.onnx
```

Restart the canonical programme and enable both **Keep static UI sharp** and
**AI: UI regions (exp.)** before capture. The
option defaults off. Disabling the parent UI mask also disables detection.
The flow/refinement/blending controls keep their original meanings.

### GUI detector selection

The adjacent dropdown selects the **UI detector only**, never flow:

- **DirectML: icons (default)**: preserved OmniParser V1 FP16 graph.
- **Matrix: icons (WMMA)**: explicit Vulkan cooperative-matrix icon graph.
  Select this to use verified matrix instructions from the GUI. Still slower
  than DirectML icons; not the default and not a 4 ms gaming-load guarantee.

Both **Keep static UI sharp** and **AI: UI regions (exp.)** must be checked.
Stop capture to change selection. Missing requested files or unsupported matrix
features fail setup, with no model/backend/full-screen-mask fallback. Explicit
GUI/CLI choices take precedence over the legacy backend environment variable.
Merely placing matrix files in `models/` no longer changes the default backend.

### Icon-only consistency

Both dropdown entries use the same preserved icon model. New icons still need
confidence >=0.10. An icon from the previous fresh result may continue at
>=0.075 only with a **current** proposal overlapping it at IoU >=0.60. Each
previous icon can support only one weak continuation. Supported existing icons
get priority at the 32-box limit instead of dropping out when unrelated scores
change. NMS remains 0.5 and padding remains two detector pixels. Output geometry
and confidence always come from the current proposal, never old/smoothed boxes.

This is confidence hysteresis/capacity continuity, not a promise of perfect
detection: an entirely missing proposal or score below 0.075 still drops
immediately, and model false positives remain possible. Expired/unavailable,
out-of-order or invalidated history cannot authorize continuation. There is no
text/OCR path, extra detector, guessed HUD strip or hold of missing/stale boxes.
`--ui-detector icons|matrix|auto` selects execution; the former `text` choice
fails explicitly. Old text probability-map models are rejected even through
`MOTION_ENHANCER_UI_MODEL`. Text weights/licence/provenance were moved intact
out of the runtime to the before-change archive, not destroyed.

`test_consistency.py --executable <exe> --image <local.png> --out <new-folder>`
checks repeated icon raw-output checksums and boxes within/across fresh
DirectML and matrix instances. It uses the supplied local image, not screen
capture; no models or input files are modified. This checks same-input
stability, **not** accuracy or freedom from flicker as the game scene changes.
See `docs/validation/ai-ui-icons-only-2026-11-01.md` for this removal's checks.
`test_temporal.py` adds generated changing-input ONNX fixtures through the real
GPU preprocessing/DirectML/worker path: threshold crossings, no weak acquisition,
missing/below-floor immediate release, current coordinates and capacity ranking.
These controlled regression fixtures are not trained models or live-game tests.
See `docs/validation/ai-ui-continuity-2026-11-01.md` for continuity validation.

### Live debug output

Press **Ctrl+Alt+F3** until **view 7** (AI UI regions / mask), after the AI
blend-mask view. Cyan outlines = detector areas, yellow = native evidence
without protection, green = actual supported mask coverage. The dim picture
is a source-frame backdrop, not a warped-output quality comparison. No boxes
are filled as masks. No additional inference runs for the debug view.

The top-left legend reports **MASK OFF**, **AI OFF**, **NO PAIR**, **NO RESULT**
(not ready, expired or failed), or **REGIONS: 00**–**32**. A valid zero-region
result is visibly different from waiting for inference. The boxes and status
are saved with each flow pair and match the displayed mask, even when a newer
detection completes. With AI off, the existing native mask is still shown.
Switching views also redraws held/static frames. Press the hotkey again after
view 7 for normal output; **Ctrl+Alt+F2** saves the debug output in captures.

Offline: set `MOTION_ENHANCER_DEBUG_VIEW=7` before `--gpu-offline` or
`--gpu-offline-sequence`. The first pair can show NO RESULT; offline replay
does not wait for the detector just to make a debug image.

`test_debug_view.py --executable build-project/Release/motion_enhancer.exe
--out <new-folder>` exercises the native GPU view and saves synthetic diagnostic
previews for boxes/masks and each status. It checks supported vs isolated mask
pixels, stale-box suppression, distinct status text, offline view selection,
repeatable normal output and unchanged runtime models. `--ui-regions-self-test`
also runs those GPU assertions without writing previews.

CLI: `--ai-ui-regions`, with the usual `--flow-engine dis` or `ai`, etc.
`--no-ui-mask` overrides it. `MOTION_ENHANCER_UI_MODEL` selects an alternative
file without replacing a runtime model. A missing/incompatible requested
detector stops session setup; it never silently falls back to full-screen
native masking.

## Runtime and safety

- RGB input is letterboxed on the GPU to 640x640. With DirectML, float NCHW
  stays in GPU memory; only the icon proposals (168 KB) return to the worker CPU.
- A separate compute queue/worker requests detection at **at most 5 Hz**.
  One job at a time, no backlog. Capture does not wait for inference/readback.
- ONNX model execution is DirectML-only: unsupported nodes fail session setup
  rather than silently falling back to the CPU. Box decoding/NMS still runs
  on the worker CPU as before. This is not a guarantee of specific matrix ISA.
- At most 32 regions, new-icon confidence >=0.10 (matched continuation >=0.075
  with IoU >=0.60), IoU-NMS 0.5, two detector-pixel edge padding.
  Degenerate/nonfinite/padding-only or >50%-of-frame boxes are rejected.
  The size filter can miss large menus, intentionally preferring less masking.
- Until a result exists, when no regions are found, after failure, or more
  than **600 ms after input submission**, the native mask is all zero. Old
  pixel evidence outside current regions is cleared immediately. Resize
  restarts the service; detected frame-ID gaps invalidate its cache.
- GPU work still competes with the game. Asynchronous/lower-rate detection
  is **not a guarantee of a 4 ms total interpolation budget**. Measure live
  pacing and game load as well as detector time before calling it fast enough.

`--ui-regions-self-test` checks CPU box decoding, letterboxing, NMS, invalid
values, cap and expiry, plus actual native GPU gating: only in-region evidence
survives, empty/lost regions clear evidence, rectangles alone do not create
masks, changing UI still releases, and legacy masking stays unchanged.
`--ui-detect-test input.png output.json` exercises GPU preprocessing,
asynchronous inference, readback, buffer reuse and invalidation. It reports 21
worker wall-time samples after three warmups (excluding session creation and
input preprocessing), plus per-inference `region_history` for offline temporal
regressions; it does not measure end-to-end live pacing.

## Offline-only text alternatives

The saved Arabic quest-text miss is diagnosed in
[`ai-ui-text-miss-2026-11-01.md`](../../docs/validation/ai-ui-text-miss-2026-11-01.md).
Separately authorized PP-OCRv6 tiny/small and CRAFT experiments are documented in
[`ai-ui-text-alternatives-2026-11-01.md`](../../docs/validation/ai-ui-text-alternatives-2026-11-01.md).
**None was activated or accepted as a reliable <=4 ms solution.** This does not
restore the removed Text/HUD dropdown/runtime path.

Optional diagnostic tools, not linked into the application:

- `prepare_text_probe.py --source <local.onnx> --sha256 <pinned-hash> --kind
  db|craft --size 640 [--height 640] --out <new-folder>` creates static FP32/FP16
  copies with source-specific normalization. Requires `onnx`, `numpy` and the
  existing isolated ORT FP16 converter; do not change the trainer environment.
  This is conversion, not training/calibration or model installation.
- `offline_text_probe.cpp` runs one-input/one-output float32-I/O static models
  using existing DirectML/ORT DLLs and GPU-bound input/output resources. CPU EP
  fallback is explicitly disabled. Build its standalone `offline-probe/` CMake
  project into a **new temporary diagnostic directory**, not an application
  build. It does not modify `build-project/Release` or capture the screen.
- Probe arguments: `<existing-runtime-dir> <static-model.onnx>
  <input-manifest.txt> <NEW-out-dir> <4..500-runs> <profile:0|1>`. The manifest
  lists absolute paths to contiguous NCHW RGB float32 `[0,1]` tensors, prepared
  with center-aligned bilinear letterboxing and `114/255` padding. Save model
  provenance/input geometry with each experiment. Use separate profiled placement
  runs and unprofiled cost runs. The probe checks repeated same-input maps and
  first-input restoration after buffer reuse.
- `inspect_text_probe.py --run-dir <probe-output> --images <originals-folder>
  --kind db|craft --size <width> [--height <height>]
  [--references <manual-rectangles.json>] --out <new-folder>` writes annotated
  copies and box-reference coverage. This is an explicitly approximate
  axis-aligned contour/component decoder, not exact upstream rotated-polygon
  decoding, OCR recognition, semantic HUD detection or native masking.
- `test_text_probe.py` covers empty maps, score admission, geometry, reference
  unions, source/output preservation and conversion order. Opt-in GPU fixtures
  use `MOTION_ENHANCER_TEST_TEXT_PROBE=<standalone-probe.exe>` plus
  `MOTION_ENHANCER_TEST_RUNTIME=<existing-runtime-dir>` to check source-specific
  BGR normalization/readback and rejection of a CPU-only model.

Probe worker time includes GPU-bound Run, normalization and output readback,
wait and copy. It excludes resize/input preparation/upload, session creation,
component decoding, concurrent icon detection, native masking, flow and
presentation. CPU Python decode is measured separately. Warm probe averages
and asynchronous scheduling do not establish a gaming-load deadline.

## INT8 / matrix-acceleration investigation

**FP16 remains selected.** On the RX 9070 XT, the tested static INT8 versions
were slower than FP16 in counterbalanced native runs. The downloaded dynamic
INT8 version was slower still. INT8 proposals also differed; baseline agreement
is not ground-truth HUD accuracy. No INT8 model was activated or copied over
the original. See [measured results](../../docs/validation/ai-ui-int8-2026-11-01.md).

DirectML/AMD's driver chooses the kernels. GPU-only ONNX execution and genuine
8-bit tensors do **not** prove that the selected shader uses WMMA/matrix units.
There is no “force matrix cores” switch in the DirectML integration. Its
instruction selection remains unverified, and no 4 ms total-budget guarantee
is claimed. The separate explicit Vulkan backend is described below.

Reproducible opt-in tools (all outputs must be **new directories**):

- `quantize_model.py`: reads the hash-pinned upstream FP32 UI detector plus
  explicitly supplied local screenshots. Creates a separate static INT8 ONNX,
  licence and provenance. MinMax calibration only, **no training**. Supports
  QOperator/unsigned activations or QDQ/signed activations with per-channel
  INT8 weights; it never processes/replaces flow models or datasets. Requires
  `onnx`, `onnxruntime`, `numpy`, Pillow and OpenCV. CPU ONNX Runtime is needed
  only for offline calibration, not application playback.
- `benchmark_int8.py --executable <exe> --fp16 <original.onnx> --int8
  <candidate.onnx> --out <new-folder> <screenshots...>`: obtains separate ORT
  placement traces, rejects CPU-provider model nodes, then runs unprofiled
  FP16/INT8 and INT8/FP16 rounds. Reports per-image worker times and box
  agreement without activating either file or claiming matrix-unit usage.
- `MOTION_ENHANCER_UI_PROFILE=<new-prefix>` enables native ORT tracing;
  `--ui-detect-test` closes the profile and prints its path as `UI_PROFILE`.
  It is off during normal capture. Profiled times are not benchmark times.
- `MOTION_ENHANCER_UI_TEST_RUNS=127` adjusts only the offline detector test's
  repetitions (4–500, first three excluded). Default remains 24/21 measured.

The tested calibration runtime was installed only into the approved temporary
diagnostic directory, not the existing trainer environment. Native/UI tests
and model hashes passed after the GPU-only guard was added.

## Explicit AMD matrix backend

**Implemented and ISA-verified, but not selected by default.** The original
readback/upload bridge measured 3.63–3.76 ms worker means. The newer shared GPU
input removes that transfer and is selectable from the GUI. DirectML icons
remain faster. See the original evidence in
`docs/validation/ai-ui-matrix-2026-11-01.md` and the newer comparison in
`docs/validation/ai-ui-improve-2026-11-01.md`.

The optional ncnn/Vulkan backend uses **FP16 cooperative matrix multiplies with
FP32 accumulation**, not the slower experimental INT8 ONNX graph. Matrix units
also support FP16; quantization is not required to use them.

It checks the actual pipeline: the submitted shader must contain
`OpCooperativeMatrixMulAddKHR`, its AMD disassembly must contain `v_wmma_`, and
those verified pipelines must be dispatched during each successful inference.
Every dispatched cooperative-matrix pipeline must pass the ISA check. Missing
shader/driver evidence means no result, not silent CPU/ordinary-shader fallback.
Activation, pooling, small output heads, etc. still use ordinary GPU shaders.
Box decoding/NMS remain native CPU worker code in either backend.

- Dependency: `setup_matrix.py` obtains hash-pinned ncnn 20261020 source.
  Enable with `cmake -S . -B build-project
  -DMOTION_ENHANCER_ENABLE_UI_MATRIX=ON -DMOTION_ENHANCER_STAGE_AI_FLOW_MODEL=OFF`,
  then build the same canonical Release target. No extra application build.
- `convert_matrix.py --model <pinned-FP32-source.onnx> --pnnx <pnnx.exe>
  --licence <detector-AGPL.txt> --out <new-folder>` creates a separate graph.
  Requires the hash-pinned pnnx 20261020 Windows x64 binary; no training/calibration. Keep the licence/provenance and
  validate before staging. The original FP16 ONNX is preserved.
- Runtime: `models/ui_detect.matrix.param` plus `.bin`, `.LICENSE.txt` and
  `.source.json`. CMake does not implicitly download/stage/package these.
   Its presence does not select it automatically. Choose **Matrix: icons (WMMA)**
   beside the still-default-off **AI: UI regions (exp.)** GUI option, or use
   `--ui-detector matrix` explicitly.
- `MOTION_ENHANCER_UI_BACKEND=matrix` requires the matrix backend;
  `MOTION_ENHANCER_UI_BACKEND=directml` selects the preserved ONNX path.
  `MOTION_ENHANCER_UI_MATRIX_MODEL=<path.param>` selects a separate candidate.
  An ONNX `MOTION_ENHANCER_UI_MODEL` override keeps selecting that ONNX graph
  unless `UI_BACKEND=matrix` is explicit. Missing requested files fail setup.
- Vulkan adapter LUID must match D3D12. This backend requires an AMD driver
  exposing cooperative matrix and `VK_AMD_shader_info`. No driver/SDK updates.
- The input buffer is a dedicated shared D3D12 resource imported by Vulkan on
  the matching adapter via `VK_KHR_external_memory_win32`. No CPU input readback,
  memcpy or upload. GPU preprocessing still copies the planar texture into
  that buffer, and GPU packing still converts it for the graph: not a fully
  copy-free pipeline. The worker waits for the D3D12 producer, and the retained
  Vulkan graph acquires/releases external ownership every execution. Capture
  never waits for detector inference/readback; failed jobs prevent buffer reuse.
- The fixed-shape Vulkan command buffer and all intermediate buffers remain
  alive for replay. Each inference submits the same ISA-verified graph with
  freshly produced shared GPU input, waits on its own completion fence, invalidates
  the mapped output and returns fresh proposals. Intermediates use additional
  VRAM. CPU-only or feature-masked graph layers are rejected before execution.
  First-worker time (graph recording/allocation) is excluded from
  warm-run statistics but retained in `all_worker_ms_including_cold`.
- `benchmark_matrix.py --executable <exe> --fp16 <original.onnx> --matrix
  <candidate.param> --out <new-folder> <screenshots...>` compares native raw
  proposals, boxes and counterbalanced worker times. ISA dumps are separate
  from timed runs. `MOTION_ENHANCER_UI_MATRIX_TRACE=<new-folder>` retains AMD
  disassembly. `MOTION_ENHANCER_UI_RAW_OUTPUT=<new-file.f32>` saves one raw
  result for offline comparison and refuses an existing destination.
- Replay regression: `MOTION_ENHANCER_UI_TEST_CHECKSUM=1` enables offline
  per-result fingerprints; `MOTION_ENHANCER_UI_TEST_ALTERNATE=<same-size.png>`
  alternates inputs in `--ui-detect-test`. Tests compare all replay results
  against separately initialized graphs to detect stale/accumulated outputs.
  These diagnostics are disabled in normal capture and timing comparisons.

The native per-pixel mask, flow/refinement/blending controls, 5 Hz/one-job
scheduling, 600 ms expiry and no-full-screen-fallback remain unchanged.
Explicit matrix use alone does not prove the 4 ms gaming-load budget or fix
OmniParser's game-HUD domain mismatch.
