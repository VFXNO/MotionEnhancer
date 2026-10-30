# AI Trainer — your own pyramid flow model

Train the **AI pyramid** flow engine on your own videos, with PyTorch on ROCm (AMD GPUs on Windows), and export it for Motion Enhancer. Open it from the program's **AI Model** page (**OPEN AI TRAINER**), or run:

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\app.py
```

The window has three pages: **Dataset → Train → Export**. Each long job runs as its own process (`mefi.dataset`, `mefi.train`, `mefi.export`), so a crash in a job cannot take the window down, and every page can also be run from the command line.

## Setup (once)

Installed releases include `Setup-AITrainer.cmd` and a Start menu **Set up AI
Trainer** shortcut. These create a private environment and install Python 3.14,
FFmpeg/FFprobe and the selected training backend. Internet access and several
GB of space are required. From this folder you can also run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Setup-AITrainer.ps1
# Validate an existing environment without installing/downloading anything:
powershell -NoProfile -ExecutionPolicy Bypass -File Setup-AITrainer.ps1 -CheckOnly
```

Automatic AMD selection covers RX 9070 (gfx1201); use `-Backend amd
-AmdTarget <your-gfx-target>` for another AMD architecture supported by AMD's
wheel index, `-Backend cuda` for CUDA 13.0 wheels, or `-Backend cpu` for CPU.
Compatible GPU drivers must already be installed. Installed and portable
application folders are detected as export destinations, not just build trees.

Python 3.11–3.14 and the AMD Adrenalin driver. For an RX 9070 XT (gfx1201):

```powershell
py -3.14 -m venv tools\ai-train\.venv
tools\ai-train\.venv\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0" "torchvision[device-gfx1201]==0.29.0a0+rocm10.1.0"
tools\ai-train\.venv\Scripts\python.exe -m pip install -r tools\ai-train\requirements.txt
```

The ROCm PyTorch download is about 1.6 GB. For other Radeon GPUs, replace `gfx1201` with your GPU's target (see AMD's "Install PyTorch for ROCm" page). Building a dataset also needs `ffmpeg` and `ffprobe` on the `PATH`. Check the install with:

```powershell
tools\ai-train\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 1. Dataset

A dataset is a set of **triplets**: three frames in a row from your videos. The model sees the first and last and learns to rebuild the middle one, so no motion labels are needed.

Add videos like what you will play: the same games, camera speeds and HUDs. Frames are decoded already scaled to the **frame height** (360 matches the network's view of 1440p capture), at their displayed shape: a phone video stored landscape with a rotation flag comes out portrait, and non-square pixels are corrected. A triplet is skipped when it is not a clean sample of motion:

- **Repeated frames.** Recordings of games that ran below the recording rate repeat frames, and then the middle frame is not the true middle.
- **Scene cuts.**
- **Stills.** Less than the least motion.
- **Too fast.** More than the most motion.

**Flash scenes are included.** Strong return flashes (A→B→A: similar endpoints
surrounding a very different middle frame) are retained and tagged `flash` in
new dataset entries. They bypass the cut/minimum-motion heuristics that would
otherwise discard them, even when endpoint optical flow is zero. The builder
reports these as **kept flashes**, not rejected samples. Other scene-cut,
duplicate and maximum-motion filters remain in place; this is a targeted
return-flash detector, not a detector for every type of exposure change.

**Frame gaps** 1 and 2 take consecutive frames and every other frame. The current builder tries selected gaps in ascending order and keeps the first passing candidate: selecting multiple gaps does **not** create a balanced mixture. Gap size is a temporal interval, not a guarantee of small/large motion. Each video's triplets are spread over all of it: the spacing between them grows with a long video so that its limit is reached at its end, not in its first minute. Building into an existing folder adds to it. Triplets already there are kept and not made twice, and triplets whose images were deleted are made again.

New datasets identify each source by its full canonical path, including its
extension. Two files named `videoplayback.mp4` and `videoplayback.webm`, or two
folders' `clip.mp4`, no longer share sample names or a temporal split. Adding
the same dataset twice, or the same source/frame/stride from two new datasets,
does not double-weight it during training. This is path identity, not
content-based detection of renamed copies.

**Legacy datasets remain readable for training.** Their basename-only indices
cannot safely distinguish such sources after the fact. To rebuild their videos
with the new identities, choose a **new dataset folder**.
The builder refuses ambiguous appends instead of relabelling or duplicating
existing triplets. Do not combine a legacy dataset with a rebuilt copy: its
missing source identities make cross-copy deduplication impossible.

```powershell
python -m mefi.dataset --out datasets\mygame --height 360 game1.mp4 game2.mkv
```

## 2. Train

The model is **PyramidFlow** (`mefi/model.py`). It works coarse to fine over a **pyramid whose depth follows the resolution**, as RIFE's IFNet does over its scales. Level 0 is the network resolution and each level is half the one below. Levels are added until the top is 4–8 px on its short side, so even a fast camera turn is a pixel or two there:

| Network resolution | Levels |
|---|---|
| 640×352 (¼ of 1440p) | 7 |
| 256 px training crop (464×256) | 7 |
| 1280×720 | 8 |
| 1920×1088 | 9 |

At each level a block sees both frames, both frames warped by the motion so far, that motion and the mask. It refines the motion **in both directions** (frame0 → frame1 and frame1 → frame0) plus a blend mask.

Every level above level 3 shares one small block: a level's motion is in its own pixels, so the same refinement fits every scale. That also lets one set of weights run any number of levels, and fewer weights generalise better from small datasets. Levels 3 to 0 each have their own, wider block. Every block starts at zero output, so an untrained model is "no motion, an even blend". With random starting outputs, noise from the top level was doubled at each level down, and training spent its first 19 epochs undoing it. Warps sample in FP32 even in the FP16 model: FP16 resolves only half a pixel at x = 640. The program's GPU synthesis then makes frames at any time from the two directional fields, with its own occlusion handling and HUD protection.

Training scores, at every scale:

- **The middle frame.** How well the two flows rebuild it, with a Charbonnier loss plus a census (local structure) loss.
- **Each direction.** How well each direction alone maps one end frame onto the other.
- **Smoothness.** A little smoothness of the flow where the picture is flat.

Training uses mixed precision on the GPU, AdamW (learning rate 1e-3) and a cosine learning-rate decay. Every pyramid level's loss counts the same: halving it per level left the coarse levels, the only ones that can find a large motion, almost untrained. Training adds flips, time reversal and mild colour changes, plus two optional kinds of motion:

- **A synthetic camera pan** (`--pan`, off by default). The three crops are taken from shifted windows, so the clip moves linearly on top of its own motion. It measured worse: with pans, training fell back to a plain blend (29.98 dB against 32.64), and gentler pans still cost 0.7–1.8 dB. A model learns large motion from footage that has it.
- **Optional occlusion masking** (`--occlusion-start`, off by default). The "map one frame onto the other" loss skips points hidden in the other frame (UnFlow's forward–backward check). It measured no better, and fading it in early stalled training.
- **A zoom** of 0.6× to 1.6×, which scales every motion with it.

Quality is measured as **PSNR of the rebuilt middle frame on the last tenth of
each video**. Videos with fewer than ten triplets are training-only. Two
boundary triplets are dropped, and any additional training triplet whose source
frame interval overlaps the held-out section is removed, including long strides.
This is temporal hold-out within the same footage, not a test of unseen games.

Up to **96 held-out samples** are scored each epoch, balanced across source
videos and spread across each one's held-out section. About three quarters are
moving samples; spare slots are filled when either group is small. Long videos
can no longer crowd smaller clips out of checkpoint selection. The trainer
reports both the scored count and the total available hold-out count.

The headline score counts **moving** triplets (90th-percentile motion of 3 px
or more). Zero endpoint motion does not imply an easy target: return flashes
may have identical endpoints and a very different middle. Flash samples train
with their original targets and count in the **all-clips** score. If no
validation sample moves, the headline and UI use that all-clips score instead.
Scores are compared with a plain blend. The preview shows a different moving
clip each epoch:

- **Row 1:** frame 0, the true middle, frame 2.
- **Row 2:** the model's middle, a plain blend, and the error.
- **Row 3:** both motion fields and the occlusion view.

If few validation clips move, the trainer says so. It needs footage with real camera and object motion.

The current model predicts flow and a blend mask, not new scene brightness.
Keeping flash scenes allows them to participate in training and evaluation,
but does not make a hidden middle-frame flash inferable from identical
endpoints or guarantee perfect flash reconstruction. No retained flash targets are
replaced with blends, suppressed or relabelled as ordinary motion.

Model sizes (parameter count is not a latency prediction):

| Size | Original parameters | With integrated refiner |
|---|---:|---:|
| small | 294,516 | 295,240 |
| medium | 715,932 | 716,688 |
| large | 1,616,452 | 1,617,240 |

Measure the actual exported model and whole pipeline at the intended capture
resolution. Historical best-case/network-only times are not a 2 ms deadline
guarantee; GPU load, clock state, conversion, DIS correction and synthesis matter.

### Optional integrated flow refinement

On the Train page, **Integrated flow refinement** selects the experimental
`pyramid-refine-v1` architecture. It is off by default so old experiments do not
silently change. It reuses the finest block's existing half-resolution features,
pixel-shuffles them to full network resolution and applies a narrow 1x1 +
depthwise 3x3 + 1x1 head. Its four outputs correct **motion vectors only**, by
at most +/-2 network pixels per component. It adds no RGB generator, second
encoder, extra image warp, or extra pyramid-loss evaluation. Native interpolation
remains in use. Playback can select this head **instead of** DIS refinement;
the separate blending-mask checkbox does not select either refiner.

**Training stays the same:** original triplets/targets, losses, augmentation,
optimizer, scheduler and flash inclusion. No teacher, offline optical-flow
labels or additional dataset preparation. The head still adds some training
and inference work; benchmark it rather than assuming a tiny parameter count
makes it free. It cannot fix a completely missed large displacement by itself.

To add it to existing weights, choose a **new run folder**, the **same model
size**, enable refinement, and select the old checkpoint under **Initial
weights**. The new head starts with exactly zero correction; all original
weights are copied, while optimizer/schedule/validation start a new experiment.
The source file is never rewritten. Example from `tools/ai-train`:

```powershell
python -m mefi.train --data datasets\mygame --run runs\refined-experiment --preset medium --refiner --init-from runs\original-medium\best.pt --epochs 60
```

`--init-from` and `--resume` are mutually exclusive. Continue always restores
the checkpoint's architecture: ticking refinement cannot convert a legacy
run in place, and unticking it cannot discard a trained refiner. Export detects
either architecture and keeps the existing `frames` → `flow`, `mask` ONNX
interface. A Medium model cannot initialize from Large weights.

Checkpoints record their architecture. One from an older model is refused with a message, not a crash: train a new run.

**Stop** finishes the current step and saves `last.pt`. **Continue this run**
resumes at its next unconsumed batch, including the rest of a partially trained
epoch; it does not skip to the next epoch. `best.pt` holds the model with the
best held-out quality, not the optimizer state needed to continue training.

To continue: select the original **Run folder**, tick **Continue this run**,
and click **START TRAINING**. **Total epochs includes epochs already trained**:
to extend a completed 6-epoch run by 4, set it to **10**, not 4. If the target
is already complete, the trainer explains this without rewriting checkpoints.
Extending the target continues smoothly from the saved learning rate instead
of resetting it to the initial rate. The checkpoint restores the model size,
dataset paths, batch, crop, initial learning rate, seed and loss settings;
changing those controls does not silently turn Continue into a new experiment.

New checkpoints include the optimizer, LR schedule, FP16 scaler, random states,
training-sample identity and batch position. Deterministic per-epoch shuffling
and per-sample augmentation allow resumed batches to match uninterrupted
training even with prefetched loader workers. CPU regression tests check exact
weights and optimizer state; bit-for-bit cross-device/GPU reproducibility is
not promised. Saves are atomic and occur every **100 steps**, on Stop and at
epoch end (`--save-every` changes the interval). An unexpected termination can
only recover through the latest saved step, not an unsaved in-flight batch.

Legacy runs, including `run1`, can continue without throwing away learned
weights. Their original `last.pt` and `best.pt` are first preserved as numbered
`*.pt.before-resume-NNN` backups. When validation metadata is missing or changed,
the saved best **and** last models are re-scored on the current hold-out before
selecting a comparable best score. Old scores are not reused. The saved
scheduler infers the legacy batch position, but old files lack random/FP16
states, so the original augmentation cannot be reproduced exactly for that
one-time migration. Re-scoring may take time before training resumes.

Use a **new run folder** for a genuinely different dataset/experiment: starting
without Continue never overwrites existing checkpoints. Versioned continuation
refuses a changed training-sample selection rather than silently skipping or
repeating batches. Protocol/selection metadata is also retained on export.
Scores from old and new validation subsets are not direct quality gains.

If a small dataset has fewer samples than the selected batch size, its one
batch is trained with a warning instead of dropping every step and saving an
untrained model. Tiny datasets still do not establish useful generalisation.

```powershell
python -m mefi.train --data datasets\mygame --run runs\mygame --preset medium --epochs 60
```

## 3. Export

Export writes the checkpoint as ONNX in FP16 with FP32 input and output. Output is `models\ai_flow.onnx` next to `motion_enhancer.exe`, plus a `.json` with its resolution, size and score. **Measure speed** runs `motion_enhancer.exe --ai-flow-test` on it.

The network resolution is fixed at export. The Export page offers sizes **from your monitors**: a quarter (recommended), a third or half of each, with the width rounded to a multiple of 64 (the program copies whole 256-byte rows) and the height to 16. For a 2560×1440 monitor that is 640×352, 832×480 or 1280×720. Build the dataset at about the same height (the Dataset page defaults to a quarter of your monitor). The model learns motion at the scale of its dataset frames, and the export warns when the two are far apart. The model is written as one file with its weights inside. If its shape differs a lot from the captured window's, the program warns that motion is stretched.

```powershell
python -m mefi.export runs\mygame\best.pt --out ..\..\build\Release\models\ai_flow.onnx --width 640 --height 352
```

Then choose **AI pyramid (trained model)** as the flow engine on the Live Capture page, or use `--flow-engine ai` on the command line. Without a model, the engine falls back to DIS.

### Programme switch: AI refinement or DIS

**AI flow refinement (off: DIS)** on Live Capture chooses the correction path:

| Checkbox | AI motion estimator | Correction stage |
|---|---|---|
| Checked | Trained pyramid with integrated head | Integrated AI correction; **no DIS refinement** |
| Unchecked (default) | Same jointly trained backbone, head removed | Existing **DIS refinement** |

The AI motion estimator remains active in both cases; unchecked does **not**
switch the entire engine to pure DIS. **AI: blending mask** remains independent.
Choose the setting before starting capture. Stop/restart capture to change it;
the checkbox is disabled while a capture session owns the loaded graph.

Train with **Integrated flow refinement** to enable this choice. Exporting a
refined checkpoint automatically writes:

```text
ai_flow.onnx       integrated head included
ai_flow.json       primary metadata
ai_flow.dis.onnx   identical backbone weights, refiner operators removed
ai_flow.dis.json   companion metadata
```

Keep all four files together. The trainer's copy-to-builds option copies the
whole pair, including when renaming the primary file. Matching export IDs,
resolution and preset are checked before selection. Missing/mixed pairs fail
with an instruction to re-export, rather than stacking AI+DIS or running a
different refiner silently. Older single-file refined exports must be
**re-exported**; their `.pt` checkpoints need no conversion or retraining.

Models trained **without** integration remain single-file exports and use DIS;
their AI-refinement checkbox is unavailable. A checkbox cannot add learned
weights that the checkpoint never contained. Bypassing a jointly trained head
may change accuracy, so test both modes on your footage. Neither mode guarantees
a 4 ms maximum, and the bypass metadata does not reuse the full model's PSNR.

CLI equivalents:

```powershell
motion_enhancer.exe --flow-engine ai --ai-refinement --capture-window "Game"
motion_enhancer.exe --flow-engine ai --no-ai-refinement --capture-window "Game"
```

## Files

| Path | What |
|---|---|
| `app.py` | the trainer window (PySide6) |
| `mefi/dataset.py` | triplets from videos |
| `mefi/data.py` | training/validation data, augmentation |
| `mefi/model.py` | the pyramid flow model and its exported form |
| `mefi/train.py` | training loop, losses, previews |
| `mefi/checkpoint.py` | atomic saves, continuation state and learning-rate schedule |
| `mefi/export.py` | ONNX export |
| `mefi/bundle.py` | complete paired-export copying |
| `datasets/`, `runs/`, `.venv/` | your data, training runs and Python (not in git) |

## Quality regression tests

From the repository root (no existing dataset or checkpoint is modified):

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_quality.py
```

Tests cover source collisions/idempotence, flash retention/training, duplicate loading,
temporal separation, balanced/full-budget validation, checkpoint safety,
mid-epoch/worker resume parity, legacy migration, extending a finished run,
and trainer UI progress/stop handling. They check correctness, not a measured
improvement from a full retraining run.

## Refiner test environments and measurements

Run the original suite plus the refiner's CPU/FP32, gradient, checkpoint,
worker-continuation, flash and off-screen GUI regressions:

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_refiner.py
```

The opt-in GPU benchmark requires the existing ROCm environment, FFmpeg-free
indexed dataset access, ONNX exporter dependencies and a rebuilt DirectML
application. Its output folder **must be new**. It trains only short, matched
synthetic controls with the unchanged training loss, then tests FP16 ONNX and
native pipeline timing. It also checks exact initialization from a copy of real
legacy weights. No user dataset, run, settings or installed model is replaced:

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\benchmark_refiner.py --out C:\path\to\new-benchmark --executable build-project\Release\motion_enhancer.exe --checkpoint C:\path\to\best.pt --data C:\path\to\dataset
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_refiner_gpu.py --out C:\path\to\new-gpu-check --benchmark C:\path\to\new-benchmark --executable build-project\Release\motion_enhancer.exe
```

Outputs include environment versions, raw samples, mean/median/p95/p99/max,
legacy 2 ms miss counts, training timings/losses, ONNX files and preservation hashes.
The second command checks exact ROCm interrupted/uninterrupted continuation
and FP16 PyTorch-versus-DirectML flow/mask numerics. Cross-provider tolerances
are mean flow error <=0.1 network pixel, p99 <=0.5 pixel, mean mask error <=0.005.
It also evaluates known translations and selected real held-out native outputs.

For repeated, counterbalanced latency checks **without a resident PyTorch GPU
context**, use `benchmark_latency.py`. Pass the exported baseline/refined ONNX
files under `--models`, the benchmark's `timing-1920x1080` and
`timing-2560x1440` folders under `--inputs`, plus a new `--out` and the same
`--executable`. It reverses model order on alternate rounds and aggregates
the raw samples rather than averaging percentile summaries. Add `dis` under
`--models` for the native non-neural comparison.
For a paired refined export, pass `--refinement both` to counterbalance and
measure AI-only correction versus backbone+DIS; raw samples stay separate.

The current acceptance limit is **4 ms maximum**, not mean/median. Pass
`--budget-ms 4` (the default) to count all samples exceeding it, including tail
latency. This does not guarantee future deadlines or alter the model/runtime.
To apply the limit to existing raw samples without running the GPU or training
again, use:

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\benchmark_latency.py --out C:\path\to\new-4ms-report --from-report C:\path\to\latency-repeats\report.json --budget-ms 4
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_latency.py
```

The native offline benchmark is enabled by `MOTION_ENHANCER_BENCH_SERIAL=1`
with `MOTION_ENHANCER_BENCH_ITERS=100`. It warms up 20 pairs and alternates the
two frames with fresh IDs so one luma pyramid is rebuilt per pair. `flow_gpu`
includes input conversion and the selected network, plus DIS correction only
in unchecked mode; `synthesis_gpu`
is the program's midpoint shader work. `pipeline_gpu_sum` adds those GPU stages;
`pipeline_serial_wall` also includes CPU submission/fence waits. Neither includes
capture, initial upload, readback, disk IO, scanout or a concurrent game's load.
Enlarged dataset images are **timing proxies**, not full-resolution quality truth.
Do not present these controls as a full retraining or a proven gameplay-quality
improvement. A strict deadline is not established by average or best-case time.

### AI/DIS switch regressions

`test_refinement_switch.py` covers exact backbone extraction, unchanged
checkpoint/mask, head-free export, paired metadata, complete copying/renaming,
legacy exports and the trainer's copy-to-builds path. The opt-in
`test_refinement_switch_native.py` exports and checks both real graphs,
compares DirectML outputs with PyTorch, exercises the actual Win32 checkbox,
tests real held-out synthesis in both modes with blending masks on/off, and
rejects invalid bundles. It runs `--ai-refinement-self-test`, which checks
zero DIS patch passes in AI mode, positive DIS passes in unchecked mode, static
UI protection, independent mask enable/disable, finite fields and ring reuse.
Run it with a new `--out`, `--executable`, a refined `--checkpoint`, and the
`--cases` folder from `benchmark_refiner.py`. It performs **no training**.
