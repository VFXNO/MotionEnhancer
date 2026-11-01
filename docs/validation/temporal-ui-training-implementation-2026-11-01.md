# Temporal UI training implementation

**Implemented in the repository trainer:** opt-in GUI/CLI support for training
flow + UI **from scratch**, or training only the UI head on a pinned frozen
backbone. **No real training job was started.** Dense UI labels are still
required; the existing flow triplets/capture rectangles are not UI truth.

User authorization: **“now implement it for training.”** This superseded the
offered no-training trained-checkpoint parity experiment; that experiment was
not run. Existing trained weights were not copied, executed or re-exported.

## Behavior

- Train page: **Temporal UI candidate training → Flow + UI**. Clear Initial
  weights and its SHA256 for both branches to start from scratch. **UI only**
  instead freezes the checksum-pinned source checkpoint. **Off** retains the
  original flow-only command/workflow. The selector is off by default.
- Independent `mefi.train_ui`, not an implicit modification of `mefi.train`.
  Original flow triplets, sampling, augmentation, validation and loss function
  are reused unchanged. UI windows/labels are separate. UI features are detached
  and gradient clipping is separate, so UI loss does not alter flow gradients or
  their original clipping scale. Training cost includes two extra pair passes.
- Explicit dense-label manifest, per-file hashes, source-disjoint train/val/test
  declarations, frame-copy leakage checks, consecutive frame IDs and conservative
  ignored-pixel handling. No folder scan, OCR, teacher, pseudo-labeling, image
  resizing or new Arabic/synthetic asset preparation. Test entries are not read.
- New/empty run directories only. Separate versioned `last_ui.pt` / `best_ui.pt`
  retain the whole flow+head state; last also retains optimizer, scheduler,
  scaler, RNG, batch/validation position, source identities and recorded caps.
  Continue restores settings rather than converting or replacing old flow runs.
- UI validation loss selects best; candidate precision/recall/IoU are also
  recorded per source. Joint flow PSNR is separate. These are training metrics,
  not proof that final native protection is accurate or safe.
- Explicit cumulative completed-step and recorded wall-clock caps. No automatic
  extension, continuation, nonfinite retry, staging/export or activation. A
  pending operation may finish after a cap; abrupt termination can lose unsaved
  time/step accounting. Caps do not authorize a job or a retry budget.
- UI checkpoints are non-deployable and rejected by the existing runtime export
  loader; the GUI does not offer them on the Export page. No head is silently
  discarded to make an old two-output model appear UI-capable.

## Files

| File | Purpose |
|---|---|
| `tools/ai-train/app.py` | Opt-in mode/labels/pin/cap controls, separate job routing, UI metrics and continuation |
| `tools/ai-train/mefi/ui_model.py` | Shared previously prototyped head/tap plus ignored-pixel candidate loss |
| `tools/ai-train/mefi/ui_data.py` | Explicit dense-label allow-list, content/split guards, aligned causal UI sampling |
| `tools/ai-train/mefi/train_ui.py` | Scratch/frozen runner, original flow objective, separate clipping, validation, capped continuation |
| `tools/ai-train/test_ui_training.py` | Generated CPU fixtures, gradient checks and fully mocked no-update loop/GUI tests |
| `tools/ai-train/UI_TRAINING.md` | GUI/CLI usage, exact label schema, checkpoint and approval limits |
| `tools/ui-temporal/model.py`, `training.py` | Compatibility imports for the shared model/loss; isolated diagnostics remain available |

Original `mefi/model.py`, `train.py`, `data.py`, `checkpoint.py` and `export.py`
remain unchanged. The shared head's state-dictionary names/architecture were
preserved when moved out of the research-only directory.

## Checks and limits

**50 tests passed:**

- 25 existing synthetic temporal model/history/export-safety tests after the
  shared-model move.
- 22 new training tests: dense labels/chronology/content guards, locked test
  access, scratch/frozen setup, original flow loss and independently clipped
  gradients, UI validation, refusal of legacy deployment/loading, run protection,
  caps, interrupted pending validation, changed-label rejection and GUI routing.
- 3 existing flow/refiner GUI regression tests. All subprocess job launches
  were mocked/forbidden and trainer settings writes were mocked.

The loop tests replace the objective/validation and AMP step dispatch, forbid
`AdamW.step` independently, and assert that every saved weight remains identical
to initialization. They also compare the joint path's first original flow batch
and augmentation with the unchanged flow data path. Gradient-only checks use
generated CPU tensors without optimizer updates. **These tests do not validate
real learning, optimizer-moment evolution, real GPU/AMP continuation or the
throughput/memory cost of an actual job.**

The expanded controls were also rendered offscreen at an ordinary window size,
with job/device actions and settings writes disabled. Existing Windows fonts
were loaded into the diagnostic process only because the offscreen platform
initially lacked font discovery; no font installation or package change. Scroll
controls keep the training form reachable. This is not a loaded-game UI test.

Protected-file verification covers 126 existing files, including original flow
trainer/model/export source, existing checkpoint and settings, application
source/shaders, runtime models/executable, CMake cache and `Lossless.dll`. The
authorized trainer GUI source edit is intentionally outside that protected set.
No existing dataset/capture images were read/processed, no real training or GPU
benchmark ran, and no package, runtime model, active executable or installed
`C:\MotionEnhancer` file was modified. `build-project` remains the sole active
application build/deployment destination; flow-model staging remains OFF.
Existing unrelated working-tree changes remain uncommitted; no commit/push.

Evidence root:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-training-implementation-20261101-c19fnb9l\`.
It retains test logs, offscreen render checks, source fingerprints and before/
after preservation results. See [training usage](../../tools/ai-train/UI_TRAINING.md)
and the [staged plan](../plans/temporal-ui-head-2026-11-01.md).

Actual dataset/label preparation and training still need approved sources and
budget. Runtime integration, trained-wrapper parity, multilingual HUD quality,
native-mask safety and the strict 4 ms gaming-load target remain unverified.
