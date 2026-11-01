# EfficientSAM-Ti labeling assistance

**Implemented and downloaded:** smart click/box mask selection and bounded
automatic candidate scanning in the existing Label UI page. This is offline,
human-reviewed labeling assistance, not a game runtime/UI-protection integration.

## Authorization and preservation

The user requested an Internet model or automatic process, then answered
**“add both”** to a question explicitly offering EfficientSAM-Ti smart selection
and automatic candidates, including its pinned model download and GUI adapter.
This did not approve processing existing recordings/datasets, training, package
installation or runtime-model replacement. Development inference used generated
shapes only. No user footage, flow/UI training checkpoint inference, optimizer
update, teacher generation, build/install, commit or push occurred.

The model is separate from application assets and training runs:

`%LOCALAPPDATA%\MotionEnhancer\label-models\efficientsam-ti\`

- Source: `opencv/image_segmentation_efficientsam` on Hugging Face, revision
  `1e791988b34b2098736003057da0f3e6bc826259` (OpenCV Zoo export).
- File: `image_segmentation_efficientsam_ti_2025april.onnx`, **48,312,857 bytes**.
- SHA256: `4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77`.
- License: **Apache-2.0**, retained as `LICENSE`, with pinned size/hash validation.
  License SHA256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- `source.json` retains immutable source revision, URL, size/hash/license and
  labeling-only purpose. No upstream Python code was downloaded/executed.

Opening the GUI performs no network operation or inference. Another machine can
use the explicit, confirmed Download/Verify button. Existing valid files are
reused without network requests; mismatched files are refused, not overwritten.
Downloads are streamed to unique partial files, checked against size/SHA256 and
published exclusively. Killing an in-progress download can leave an unaccepted
partial; it is never loaded as a model. No new package/environment setup is needed.

## Exact behavior

- `tools/ai-train/mefi/ui_sam.py` implements the pinned downloader, model adapter,
  prompt contract, bounded candidate grid and pixel-mask duplicate suppression.
- `ui_label_worker.py` adds explicit `sam_download`, `sam_smart` and `sam_auto`
  subprocess actions. No automatic dataset-wide inference or retry is introduced.
- `ui_label_gui.py` adds an image interaction selector, positive/negative clicks,
  drag-box prompts, model status/download, scan choice and candidate preview list.
  Every candidate starts unchecked; previewing changes no labels. Generic dense
  ONNX and probability-PNG guidance remain available in an expandable section.
- `Annotation.apply_foreground` in `ui_labeling.py` accepts only selected
  foreground pixels that were unknown. Outside pixels remain unknown, and known
  manual background/soft-edge/UI labels are preserved. The generic probability
  adapter's separate background-threshold behavior is not applied to SAM masks.
- Accepting selected masks creates an undoable **draft**, clears the review flag
  and records model/prompt/selection provenance. Human correction and explicit
  review are still required before the existing reviewed-only export.

Model contract, verified directly from the pinned graph:

| Tensor | FP32 shape |
|---|---|
| `batched_images` | `[1,3,1024,1024]` |
| `batched_point_coords` | `[1,1,6,2]` |
| `batched_point_labels` | `[1,1,6]` |
| `output_masks` | `[1,1,3,1024,1024]` logits |
| `iou_predictions` | `[1,1,3]` predicted mask quality |

Prepared RGB frame copies are stretched to 1024² and scaled to `[0,1]`; prompt
coordinates are transformed consistently. Positive clicks use 1, **negative
clicks use 0**, unused padding uses -1, and box corners use 2/3. Original files
are not resized/modified by inference. Binary masks threshold logits at zero and
restore prepared-frame coordinates by nearest-neighbor interpolation. Explicit
positive/negative clicks are also checked against restored masks. A box is only
a prompt; it is not rasterized as a solid label rectangle.

Automatic scans perform 9 (3×3) or 25 (5×5) evenly spaced positive prompts on the
selected current frame, independently of any old box/clicks. Nonempty shapes
with predicted IoU >=0.50 and image-area fraction 0.00005–0.80 are considered;
near-duplicates are suppressed at pixel-mask IoU >=0.85. Up to 48 are retained,
reduced by a 64 MiB retained-mask budget for large frames. These are heuristic
proposal filters, **not calibrated HUD thresholds**. Generic scenery can score
highly. No selection is silently marked as UI or background.

Inference explicitly selects installed OpenCV DNN with the OpenCV backend/CPU
target, OpenCL disabled and at most four OpenCV threads in its subprocess.
The installed new graph engine emits a warning that target selection is not
supported; the default OpenCV path runs successfully. No DirectML/matrix/GPU
placement or real-time latency guarantee is claimed. Cancellation/startup errors,
bad tensors and the existing five-minute operation limit accept no current result.

## Verification

**112 repeatable generated/mock/no-update checks passed:**

- 27 new SAM checks: prompt scaling, padding/negative/box semantics, malformed
  prompts, RGB/shape contracts, immutable inputs, output restoration/validation,
  positive-only draft admission, manual-label preservation, undo, download
  pinning/reuse/failures/non-overwrite, candidate grid/filter/dedup/progress,
  worker routing, GUI click/box behavior, unchecked previews, no auto-review,
  explicit download confirmation, empty candidates and sample-change cleanup.
  Network calls are forbidden/mocked; no downloaded model is needed by this suite.
- 35 existing labeling generated-fixture/GUI/worker checks.
- 22 UI-training no-update generated checks; optimizer updates forbidden/mocked.
- 25 temporal synthetic checks.
- 3 existing flow/refiner GUI regressions with jobs/settings writes blocked/mocked.

**Additional actual pinned-model checks used generated 192×128 circle/rectangle
images only**, with no training or actual capture processing:

| Operation | Observed time | Candidates |
|---|---:|---:|
| Direct smart positive click | 2.737 s | 2 |
| Direct smart box | 2.597 s | 1 |
| Direct 3×3 scan, 9 prompts | 24.305 s | 4 |
| Direct 5×5 scan, 25 prompts | 50.112 s | 9 |
| GUI → subprocess smart click | 3.120 s | 2 |
| GUI → subprocess smart box | 2.878 s | 1 |
| GUI → subprocess 3×3 scan | 21.737 s | 4 |

The GUI check also accepted one deliberately checked generated mask, confirmed
only its 3,655 foreground pixels became known while outside remained unknown,
undid it back to unknown, saved only a draft, and verified cancellation of a new
scan accepted no result (about 0.12 s cancellation round-trip in that check).
Candidate checkboxes were all initially unchecked, and model bytes remained
identical after inference. These are operational smoke checks, **not text/HUD
accuracy measurements, throughput guarantees or 4 ms gaming-load evidence**.

Final offscreen views were checked with prior generated model outputs replayed
and jobs/settings writes blocked/mocked. All left controls were vertically
reachable with zero horizontal control overflow at 1600×1050. Empty-project
canvas text now distinguishes a ready project awaiting frames from no project.
Viewport unused space is dark, and selecting brush controls returns to Paint.

Evidence directory:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-sam-implementation-20261101-kg8gj45y\`
contains download/source-contract records, test logs, generated masks/project,
actual direct/GUI smoke results, screenshots and before/after preservation hashes.
Existing user-modified trainer settings were included in the before snapshot and
were not rewritten by validation. Runtime/model/checkpoint/source protection was
checked separately from authorized labeling/docs changes; model staging stays
OFF and no other application build/deployment directory was introduced.

## Remaining limits

EfficientSAM is an object-shape proposal tool, **not a pretrained HUD detector**.
Coarse grids can miss thin text, and even high-quality-looking shapes can be
scenery. Arabic/multilingual HUD coverage, antialiased stroke quality, reliability
and productivity on real game recordings are untested. Review/correction is
mandatory. No processing of the locked Arabic regression capture occurred.

Existing approval gates remain for choosing/processing actual training sources,
label quality review, training budgets, actual learning, runtime UI export and
integration, native-mask evaluation and strict combined-AI gaming-load latency.
