# UI labeling tools implementation

**Implemented:** a **Label UI** trainer page for manual dense pixel annotation,
optional local AI guidance, and reviewed-only immutable `me-ui-labels-1` dataset
export. JSON, per-file hashes and grayscale coverage/known-pixel PNGs are created
by the tool; the user does not need to hand-write a manifest.

Scope: the user requested labeling tools with automatic guidance from other AI
models or manual GUI editing. This implementation does not authorize or run a
real training/data-preparation job. Only generated diagnostic fixtures were
processed during development. No real footage/checkpoint/model inference,
download, package/environment change or application activation occurred.

## Implemented workflow

- Existing Dataset, Train and Export page indices/routes remain unchanged.
  Label UI is appended as a fourth page and is also linked from the Train UI
  options. Merely opening the editor loads no source footage or AI model.
- Explicit video/window or three-image import; no folder scans. Video windows
  decode consecutive ordinals from the beginning rather than relying on
  approximate keyframe seeking. User-confirmed image chronology is not treated
  as independently proven. Prepared copies use native valid dimensions or
  explicitly selected resizing. Original inputs are read-only.
- UI/background/unknown brushes, single-pixel radius, antialiased coverage
  values, zoom/scroll, previous-frame view-only inspection and bounded undo/redo.
  Remaining-background marking and reset require confirmation. Unsaved edits
  get save/discard/cancel prompts; saving while switching samples keeps the list
  selection and canvas in sync.
- Recording-level training/validation assignment, exact duplicate-frame checks,
  identical renamed-video checks, protected regression-input guard, relative
  file/hash validation and refusal to replace existing nonempty directories.
  Re-encoded or related recordings still need human provenance review. No
  editable held-out test split is offered.
- Optional **local binary-segmentation ONNX** inference using installed OpenCV
  DNN CPU, with explicit normalization/output semantics and checksum pinning.
  A compatible model must be supplied by the user; none is bundled or selected
  automatically. Model bytes are pinned when inspected/loaded. External tensor
  files, custom domains/functions and incompatible I/O contracts are rejected.
- Alternatively, explicitly import an external AI probability PNG in the exact
  prepared-frame coordinates. Model suggestions remain separate until applied.
  Applying uses >=0.85 UI / <=0.15 background, touches unknown pixels only and
  keeps uncertainty unknown. Confidence is not opacity/coverage or validation.
  These are editing-draft bands, not changes to runtime detector thresholds.
- Every accepted draft still needs human correction/review. Editing clears the
  review checkbox; unreviewed samples do not export. Guidance identity/settings
  and source/review provenance are retained separately from the strict training
  manifest. Text-only/object segmenters cannot be assumed to understand HUD.
- Slow decoding/inference runs in cancellable disposable subprocesses, with a
  five-minute operation limit, no automatic retry and no dataset-wide inference.
  Current worker results are accepted only after success; cancellation, startup
  failure, unsupported output and model mutation discard the result.
- Frame copies and annotation revisions are immutable. Reviewed exports are
  separate copies with `labels.json` published last; later project editing does
  not mutate earlier exports. A failed new export may leave incomplete new
  files, but does not replace old data. Existing external project edits are
  checked before metadata writes/export; this is not a multi-editor database.
- Sending an exported manifest to Train only fills its path; it does not choose
  a mode, run training, change budgets, export runtime graphs or activate models.

## Files

| File | Purpose |
|---|---|
| `tools/ai-train/mefi/ui_label_gui.py` | Label UI page, canvas, review/undo controls, safe worker ownership |
| `tools/ai-train/mefi/ui_labeling.py` | Local projects, dense edits, input/split guards and immutable label export |
| `tools/ai-train/mefi/ui_label_assist.py` | Strict checksum-pinned local ONNX guidance adapter |
| `tools/ai-train/mefi/ui_label_worker.py` | Single requested image/video/model operation subprocess |
| `tools/ai-train/test_ui_labeling.py` | Generated-fixture annotation, export, model and GUI/worker tests |
| `tools/ai-train/app.py` | Fourth-page and Train shortcut integration; close/snapshot handling |
| `tools/ai-train/UI_LABELING.md` | Plain-language walkthrough and exact supported AI contract |
| `cmake/Install.cmake` | Include the labeling/training guides; existing recursive Python install includes the modules |

Training usage/README and the staged plan were updated to distinguish tool
implementation from actual dataset preparation/training approval. No flow model,
loss, trainer, loader, runtime source/shader or active model file was changed.

## Verification

**85 checks passed:**

- **35 labeling tests:** unknown-by-default editing, single-pixel/soft coverage,
  undo/redo, confidence/unknown handling, manual-label preservation, directory
  protection, recording/frame/video leakage guards, chronology and resize,
  strict-loader export compatibility, immutable exports, content/path/review
  validation, exact generated-video window decoding, model contract/pinning,
  worker output protection, GUI review/navigation/unsaved-state handling,
  exports and disposable subprocess round trips.
- **22 existing UI-training no-update checks.** Optimizer updates remain
  forbidden/mocked by that suite; only generated fixtures/checkpoints are used.
- **25 existing temporal synthetic checks.** No actual learning is performed.
- **3 existing flow/refiner GUI regressions.** Job launches/settings writes are
  mocked/forbidden during interface/render checks.

The AI fixture is a tiny generated ONNX model selecting the input red channel.
It verifies normalization, RGB order, coordinate restoration and subprocess
delivery—not text/HUD segmentation quality. The generated MJPG fixture verifies
frame ordinals; it is not a game capture. No actual pretrained model, real
recording, GPU training, optimizer update or active-model replacement was used.
The installed OpenCV build emits a target-selection warning for its newer graph
engine; tests still pass using the requested OpenCV CPU backend with OpenCL
disabled in the operation process. No DirectML/matrix placement claim is made.

Offscreen Label UI top/bottom views were checked using generated images and
process-local loading of existing Windows fonts, without installing fonts or
writing trainer settings. All controls are vertically reachable and the control
panel has no horizontal overflow in the recorded 1593×980 rendered window.

Protected before/after hashes cover the earlier protected source/assets,
checkpoint/settings, runtime executable/models, CMake cache and original external
binary. Only the explicitly authorized trainer UI/docs/install-list source edits
are excluded from preservation equality. Flow-model staging remains **OFF**;
`build-project` remains the sole active application build/deployment destination.
No build/install/package/commit/push or installed-application modification ran.

Evidence root:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-labeling-implementation-20261101-d5rck2zt\`.
It retains test logs, generated diagnostic rendering project/images, offscreen
screenshots, source fingerprints and protected-file verification.

## Remaining limits

Automatic assistance is **not** a ready-made universal HUD detector. It needs a
semantically appropriate compatible local model or an externally produced dense
map. Arbitrary OCR/box/prompt/multiclass models need additional adapters. No real
model has yet established Arabic/multilingual HUD coverage, label quality,
throughput or reliability. Human review remains required before export.

Real sequence selection/preparation, label quality review and a training budget
still need agreement before the agent processes user data or starts learning.
Runtime UI export/integration, native-mask evaluation and the strict combined
4 ms gaming-load AI-latency target remain separate and unverified.
