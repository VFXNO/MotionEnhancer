# UI detector speed, text coverage and GUI selection — 2026-11-01

> Historical experiment. The Text/HUD option and runtime support were subsequently
> removed at the user's request because of false detections. Current behavior is
> icon-only; see [removal and consistency checks](ai-ui-icons-only-2026-11-01.md).

## Delivered behavior

Canonical build: `build-project/Release/motion_enhancer.exe`. No installation,
training, calibration, dataset processing, flow-model replacement, commit or
push. **AI: UI regions (exp.)** remains off, requires **Keep static UI sharp**,
and never changes the motion engine, AI/DIS refinement or blending mask.

The adjacent GUI selector now exposes:

1. **DirectML: icons (default)** — preserved OmniParser V1 FP16.
2. **DirectML: text / HUD** — separate PP-OCRv5 mobile text detector.
3. **Matrix: icons (WMMA)** — explicit ncnn/Vulkan matrix icon detector.

The selector is disabled unless both mask/detector checkboxes are on, and while
capture is running. Explicit GUI selection becomes `--ui-detector icons|text|matrix`
in the capture child; it does not mutate process-wide environment variables.
The same CLI flag works in offline detector/integration tests. `auto` retains
legacy environment overrides, but a matrix file merely being present no longer
changes the default. Missing/invalid requested models still fail setup.

Detection regions remain allow-lists, **not filled masks**. The original native
per-pixel evidence, cross-pair confirmation and changing-UI release are intact.
Detection stays asynchronous, at most 5 Hz, one job in flight, 600 ms expiry,
with no full-screen fallback after missing/empty/stale/failed detection.

## Matrix optimization and correctness

The 4,915,200-byte RGB tensor is now a dedicated shared D3D12 buffer imported
into Vulkan via `VK_KHR_external_memory_win32` / `D3D12_RESOURCE_BIT` on the
LUID-matched adapter. The worker waits for D3D12 producer completion before
submitting. Every retained command-graph execution acquires external ownership
before reading input and releases it before its completion fence. Capture does
not wait for inference/readback. A failed job forbids further buffer reuse.

Removed: D3D12 input readback, CPU input mapping/memcpy and Vulkan input upload.
Still present: GPU texture-to-buffer preprocessing copy, GPU packing conversion,
GPU output readback, CPU box decoding and retained intermediates/VRAM. Thus this
is shared-GPU input, **not a completely copy-free pipeline**. External-memory
support is required for this explicit backend; failure does not select another
backend. No system driver, SDK or Vulkan-loader replacement.

Matrix execution still requires cooperative-matrix SPIR-V, compiled AMD
`v_wmma_` ISA and successful submission/completion of that verified graph.
FP16 weights/storage and FP32 matrix accumulation are unchanged. There are
**60 verified cooperative/WMMA dispatches per inference**. CPU/feature-masked
layers and GPU-only/no-WMMA graphs remain rejected.

All three same-input raw-output files are **byte-identical** to the previous
matrix replay implementation. Against DirectML, the GUI frame retains 31/32
one-to-one box matches at IoU >=0.5, median matched IoU 1.0; score mean absolute
delta 0.0000554646, maximum 0.00573730, relevant coordinate maximum 1 detector
pixel. Agreement is not ground-truth accuracy. Both icon backends still return
zero regions for both saved game frames.

## Measurements

AMD Radeon RX 9070 XT, same installed driver as the earlier investigation.
No simultaneous benchmark processes. Separate diagnostic ISA/provider runs;
timed runs have tracing/checksums/raw-output diagnostics disabled. Each table
entry combines two reversed-order rounds of 127 calls, excluding the first
three calls in each group: **248 warm samples per input/backend**. Cold calls
are retained separately rather than silently discarded.

### Icons: DirectML versus shared-input matrix

| Input | DirectML mean | Matrix mean | Matrix p95 | Matrix max | Matrix >4 ms |
|---|---:|---:|---:|---:|---:|
| GUI | 1.651 ms | 2.177 ms | 2.684 ms | 2.982 ms | 0/248 |
| game_001 | 1.603 ms | 2.241 ms | 2.854 ms | 3.102 ms | 0/248 |
| game_007 | 1.614 ms | 2.165 ms | 2.680 ms | 3.029 ms | 0/248 |

The earlier readback/upload implementation measured 3.632–3.756 ms means,
171/744 warm overruns. Those are **historical runs, not a controlled
old/new simultaneous-load comparison**. Current DirectML remains faster than
matrix. Matrix first-worker times are **20.358–34.229 ms**: this is not a 4 ms
maximum guarantee, even before accounting for a game workload.

### Text: full versus pooled probability map

PP-OCRv5 network input remains 640x640. A 2x2 GPU max-pool produces a 320x320
map instead of 640x640, reducing float32 readback from 1,638,400 to 409,600 bytes
and CPU scanning cells fourfold. No input downscaling, training or calibration.

| Input | Icons mean | Full-map text mean | Pooled text mean | Pooled p95 | Pooled max |
|---|---:|---:|---:|---:|---:|
| GUI | 1.638 ms | 2.772 ms | 2.311 ms | 2.944 ms | 3.286 ms |
| game_001 | 1.608 ms | 2.548 ms | 2.233 ms | 2.935 ms | 3.228 ms |
| game_007 | 1.617 ms | 2.585 ms | 2.242 ms | 2.868 ms | 3.557 ms |

Pooled text has 0/744 warm calls >4 ms in this run; full-map text has 1/744;
icons have 1/744 in this separate comparison (maximum 5.401 ms). Pooled text
first-worker times are **4.561–5.646 ms**. Separate ORT traces contain only
`DmlExecutionProvider` model events, no CPU provider. CPU component decoding
is intentionally separate. DirectML ISA itself remains unverified.

All timings are **worker wall time** including inference, output readback and
CPU decoding; matrix also includes its D3D12 producer wait. They exclude model
setup, capture, flow-queue preprocessing, flow estimation and presentation.
This is **not controlled gaming load** and establishes neither end-to-end
latency nor a maximum 4 ms deadline for either backend.

## Accuracy scope and text contract

The original icon model missed both saved game frames. Text localization finds
quest-text regions and the objective marker/number in those same frames. These
are **two related views of one scene**, not independent games or an accuracy
benchmark. Some glyph edges/regions remain incomplete, and component merging
changes with pooling. The model identifies text-like areas, not HUD semantics;
scene text/textures can be false positives and non-text HUD elements can be
missed. There is no OCR recognition. Desktop text localization does not replace
icon detection, so both choices remain available and icons remain the default.

The native text decoder accepts only declared float32 maps `[1,1,640,640]` or
`[1,1,320,320]`, static float32 RGB input `[1,3,640,640]`, with
`motion_enhancer.ui.kind=text-db-v1`. Eight-connected finite probability
components use a 0.3 pixel threshold, conservative mean confidence >=0.7,
small-component rejection, bounded edge expansion and inverse letterboxing.
Padding-only and >50%-frame boxes are rejected; at most 32 ROI proposals.
An initial 0.5-confidence experiment admitted a low-score water/character-edge
region. The conservative 0.7 rule suppresses it in these fixtures, but this
does **not** establish general false-positive performance. Native static-pixel
evidence remains necessary inside every authorized region.

## Artifacts, provenance and preservation

Diagnostic root:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-improve-20261101-alxh68qb\`.

- `matrix-comparison/report.json`: counterbalanced samples, numerical agreement,
  cold worker calls, separate ISA dumps and raw outputs.
- `text-comparison/report.json`: three-way reversed text/icon timings, retained
  boxes and separate ORT provider traces.
- `text-candidate/`: separate full-map candidate, not the selected GUI text graph.
- `text-candidate-pooled/`: selected optional text graph and original licence.
- `protected-before.json`: all 20 protected original hashes, verified before work.

Pooled text graph SHA256:
`4115e736af3de09126a53c8588cf6479bdf8be8cbbdce6da53fef5a9529b337c`.
Source model SHA256:
`4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae`.
Source: hash-pinned RapidOCR v3.10.0 `ch_PP-OCRv5_det_mobile.onnx`.
Licence: original PaddleOCR **Apache-2.0**, SHA256
`3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e`.
RGB-to-BGR normalization, static shape, FP16 conversion and max pooling are
deployment graph changes, not retraining. OmniParser/matrix weights remain
separately **AGPL-3.0**; redistributing them still requires licence review.
No model is implicitly bundled by CMake/packaging.

Before-work executable/models/shaders/source copies are preserved under
`backup/2026-11-01_ui-improve-before-s253_il7/`. Existing model bytes, checkpoints,
dataset index, shader binaries, branding and `C:\MotionEnhancer` are protected.
Only optional new UI model files may be staged; nothing overwrites the original
icon/flow/depth models. No training or new dataset preparation was started.

### Final acceptance

- Release build succeeded in the sole canonical build directory, with matrix
  enabled and `MOTION_ENHANCER_STAGE_AI_FLOW_MODEL=OFF`.
- **32/32** downloader, conversion, numeric and native safety tests passed,
  including CPU-provider rejection, invalid text contract, CPU-feature-masked
  matrix rejection, no-WMMA graph rejection and changed/restored replay input.
- Existing eight native/integration checks passed: D3D12, projection, independent
  blend mask, both exclusive AI/DIS refinement routes, UI ROI decoder/gating,
  missing-model fail-closed and parent-mask-off behavior.
- New text component checks cover finite/confidence bounds, landscape/portrait
  inverse letterbox, empty/padding/oversized maps, 32-box cap and pooled mapping.
- Debug status images, pair ownership, stale-box suppression,
  offline view 7 and repeatable normal rendering passed the existing regression.
- Owned GUI test passed default-off, three choices/default DirectML, checkbox
  dependencies, AI/DIS/refinement independence and non-overlapping layout.
- All four CLI choices passed with runtime files, including `auto` selecting
  **DirectML despite the staged matrix graph**. Explicit matrix reports 60
  verified WMMA dispatches. Text/matrix missing-model rejection, parent-off
  bypass and native-flow integration passed, as did invalid-choice rejection.
- Seven **new** optional model/licence/provenance files are staged only under
  `build-project/Release/models/`: three `ui_text.*` files and four
  `ui_detect.matrix.*` files. None is selected/enabled by default.
- `preservation-final.json` confirms **20/20 protected original hashes** and
  **31/31 existing shader binaries** unchanged. All unrelated pre-existing
  `src/` and `include/` files also match the before-work archive.

Final executable SHA256:
`655326e951e12e79470b281854f5024d2506da956d02e8029be3eb8ea307e487`.
Test logs, `choices-final.json`, `staged-new-ui-files.json`, final text previews
and `gui-final.png` remain in the diagnostic root. Changes are uncommitted and
unpushed; the installed program is unchanged.
The final relink also reran all 32 tests and native/debug/GUI regressions
(`*-rebuild-final.log`). Re-running the final preparation tool reproduced the
staged pooled text graph byte-for-byte.
