# AI trainer quality foundation (2026-10-30)

Scope: the PyTorch/ROCm trainer in `tools/ai-train`, not the separate native
neural-flow trainer. No model architecture, loss weights or runtime synthesis
changed, and no replacement model was trained or installed.

## Read-only audit

The configured dataset contains 45,872 available triplets, with 41,268 training
and 4,586 held-out samples. Its legacy index has nine source groups even though
the input list has ten videos: `videoplayback.mp4` and `videoplayback.webm`
share a basename-only identity. Those historical entries cannot safely be
unmixed automatically.

The previous 96-sample validation subset covered only six groups, including
40 samples from the merged `videoplayback` group. Balanced selection now covers
all nine indexed groups, with 3–14 samples each, while retaining 72 moving and
24 still samples. These are deliberately different validation subsets.

Re-scored the existing `run1/best.pt` on the RX 9070 XT:

| Validation subset | Moving model PSNR | Moving plain-blend PSNR |
|---|---:|---:|
| Previous selection | 29.4096 dB | 26.5733 dB |
| Balanced selection | 29.6506 dB | 27.4567 dB |

This is **not a training improvement**: identical existing weights were used.
The broader selection exposes flash-only footage the previous sample missed.
For example, three white-flash targets score about 0.0684 dB with either the
model or a plain blend, consistent with middle frames that cannot be inferred
from their endpoints. This exposes a limitation of the current flow/blend
output model, not a reason to exclude the user's requested flash footage.

Existing best/last checkpoint, settings and dataset-index SHA256 hashes were
unchanged by the audit. Original frames, videos, models and runs were not edited.

## Changes

- New datasets use canonical-path source identities, including the extension,
  in both metadata and sample names. Repeat builds stay idempotent; missing
  generated images can be restored. Ambiguous appends to basename-only legacy
  sources fail clearly without changing that dataset; legacy data still trains.
- Repeated dataset paths and duplicate source/frame/stride samples from new
  datasets are not double-weighted. Renamed video copies and legacy/rebuilt
  mixtures are not content-deduplicated.
- Per the user's correction, strong A–B–A flashes are **retained**, tagged in
  new entries and reported as kept flashes. They bypass cut/minimum-motion
  heuristics even with zero endpoint motion. Other cut, duplicate and
  maximum-motion filters remain. Original middle-frame targets are unchanged.
  Mild linear brightness changes remain accepted without a return-flash tag.
- Held-out selection is deterministic, balances source videos, spans each
  selected temporal section and fills the full budget when one motion group
  is short. Train/validation source-frame intervals cannot overlap, even with
  long strides. This is within-video hold-out, not unseen-game validation.
- Checkpoints/exports record the validation protocol and sample-selection
  fingerprint. Legacy/changed scoring is re-evaluated on resume rather than
  comparing incompatible best scores (see continuation below). Missing or
  incomplete checkpoints and changed versioned training selections are refused
  without overwriting the run. New experiments refuse to overwrite checkpoints.
- Tiny datasets no longer silently drop their only batch. NumPy augmentation
  is seeded alongside Python and PyTorch. The UI reports scored versus available
  held-out counts and preserves actionable dataset error messages.

## Verification

- `tools/ai-train/test_quality.py`: all **33 tests passed**: source separation, repeat-build idempotence,
  legacy preservation/loading, duplicate loading, flash retention/training, temporal
  separation, balanced/full-budget selection, scoring identity, checkpoint
  safeguards, seeded CPU repeatability, real CPU training/resume and off-screen
  trainer UI regression checks.
- Flash-retention correction: a real FFmpeg-decoded A–B–A test video was kept
  under default builder thresholds with zero endpoint motion. Its original
  bright middle frame reached the training loader. A flash-only synthetic CPU
  training run took an optimizer step, saved checkpoints and produced a finite
  all-clips validation score; the UI labels that score as all-clips rather than
  motion-only. This verifies inclusion/execution, not successful reconstruction
  of an unobservable flash.
- Real FFmpeg decoding of two generated same-stem videos: eight distinct
  triplets, two source identities, unchanged repeat-build index and recovery of
  a deliberately removed test image all passed.
- ROCm mixed-precision smoke run: three isolated epochs on generated frames;
  actual optimizer updates, finite weights, validation and checkpoint selection
  passed. The existing MIOpen warning about an unreadable architecture tuning
  database remained nonfatal.
- ROCm/FP16 continuation on RX 9070 XT (`2.14.0+rocm10.1.0`): stop after step
  2 of 8, load scaler/RNG/optimizer/scheduler, and finish the remaining steps.
  Final weights matched the uninterrupted control exactly (maximum difference
  0.0), with matching scheduler/scaler state and FP16 growth tracker 8. These
  small synthetic runs establish continuation correctness, not full-data quality.
- Generated datasets/checkpoints were created only in temporary test folders
  and removed. No user run was continued or overwritten. Python syntax and
  `git diff --check` passed; no commits or pushes were made.

## Checkpoint continuation correction

The previous loop saved only an epoch number, then resumed at `epoch + 1`
even if Stop had interrupted that epoch. It also omitted RNG/FP16 scaler
state, rejected an extended epoch target, and the new validation guard blocked
the user's legacy `run1`. The corrected continuation:

- Records the consumed batch cursor/global step, running loss, actual training
  config, optimizer/scheduler, scaler, RNGs and a training-selection fingerprint.
- Uses deterministic per-epoch/per-sample seeds independent of worker prefetch;
  resumes without loading/skipping already-consumed samples. CPU interrupted
  and uninterrupted **two-epoch/eight-step** runs finish with identical model,
  optimizer, scheduler, scaler and RNG state, with both zero and one worker.
- Saves atomically every 100 steps, on Stop and at epoch end. A simulated
  partial-write failure preserves the prior checkpoint. Recovery from a saved
  final batch before validation is also tested.
- Restores saved experiment settings; allows a larger **total** epoch target
  from the saved LR. An already-completed target is a read-only no-op with
  instructions to increase it, not a fake successful retraining.
- Re-evaluates legacy best/last weights on the current selection, backing up
  both original files before migration. Legacy batch position comes from the
  saved scheduler step; absent historical RNG/scaler state cannot be recovered.
  Changed versioned training data still requires restoring the original dataset
  or a new experiment, not silently reshuffling a half-finished epoch.
- Shows restored progress/settings and paused/completed state in the UI. Stop
  targets the active run folder, even if the field is subsequently edited;
  per-launch stop tokens avoid discarding an initialization-time Stop or
  immediately stopping a later launch on a stale request.

These tests use isolated temporary runs. The user's `run1` has not been resumed,
re-scored in place or changed by this correction. No full real-data continuation
or quality improvement is claimed.

A read-only load of the actual `run1/last.pt` verified compatible model and
optimizer state and restoration of large/batch 16/crop 480/initial LR 0.0007.
With the audited 2,579 steps per epoch, scheduler step 10,316 places it at the
start of **epoch 5 of 6**, with current LR 0.0001855. Original best/last/config,
trainer settings and dataset-index SHA256 hashes remained unchanged.

## Next quality experiment

Build a **new dataset folder** from representative gameplay **including flash
scenes**, keeping the legacy data intact, and use a **new run folder**. Establish a fixed balanced
hold-out baseline before testing loss/sampling changes on matched seeds and
training budgets. A full real-data retraining or demonstrated improvement to
interpolated output has **not** been performed in this pass. Existing exports
and installer/portable artifacts have not been replaced by this trainer-only
source update.
