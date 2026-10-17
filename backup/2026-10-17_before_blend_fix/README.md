# AI Trainer — your own pyramid flow model

Train the **AI pyramid** flow engine on your own videos, with PyTorch on ROCm (AMD GPUs on Windows), and export it for Motion Enhancer. Open it from the program's **AI Model** page (**OPEN AI TRAINER**), or run:

```powershell
tools\ai-train\.venv\Scripts\python.exe tools\ai-train\app.py
```

The window has three pages: **Dataset → Train → Export**. Each long job runs as its own process (`mefi.dataset`, `mefi.train`, `mefi.export`), so a crash in a job cannot take the window down, and every page can also be run from the command line.

## Setup (once)

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

**Frame gaps** 1 and 2 take consecutive frames and every other frame, so the model sees both small and large motion. Building into an existing folder adds to it.

```powershell
python -m mefi.dataset --out datasets\mygame --height 360 game1.mp4 game2.mkv
```

## 2. Train

The model is **PyramidFlow** (`mefi/model.py`). It works coarse to fine over an **8-level pyramid**, as RIFE's IFNet does over its scales. Level 0 is the network resolution and each level is half the one below, down to 5×3 px at 640×352, so even a fast camera turn is a pixel or two at the top. At each level a block sees both frames, both frames warped by the motion so far, that motion and the mask. It refines the motion **in both directions** (frame0 → frame1 and frame1 → frame0) plus a blend mask.

The four coarsest levels share one small block: a level's motion is in its own pixels, so the same refinement fits every scale, and fewer weights generalise better from small datasets. The four finest levels each have their own, wider block. Every block starts at zero output, so an untrained model is "no motion, an even blend". With random starting outputs, noise from the top level was doubled at each level down, and training spent its first 19 epochs undoing it. Warps sample in FP32 even in the FP16 model: FP16 resolves only half a pixel at x = 640. The program's GPU synthesis then makes frames at any time from the two directional fields, with its own occlusion handling and HUD protection.

Training scores, at every scale:

- **The middle frame.** How well the two flows rebuild it, with a Charbonnier loss plus a census (local structure) loss.
- **Each direction.** How well each direction alone maps one end frame onto the other.
- **Smoothness.** A little smoothness of the flow where the picture is flat.

Training uses mixed precision on the GPU, AdamW and a cosine learning-rate decay. It adds flips, time reversal and mild colour changes.

Quality is measured as **PSNR of the rebuilt middle frame on videos the model never trains on**. Validation is split by whole video; with fewer than three videos, the last tenth of each video is held out instead. The trainer shows this against a plain 50/50 blend of the two frames. A model worth using is well above the blend.

Model sizes, network time inside Motion Enhancer at 640×352 on an RX 9070 XT:

| Size | Parameters | Network time | Notes |
|---|---|---|---|
| small | 0.29 M | about 1.3 ms | |
| medium | 0.72 M | about 1.3 ms | recommended |
| large | 1.62 M | about 1.5 ms | |

The whole engine (scaling the 1440p frames in, the network, the flow out to the ring) measured **1.54 ms** with the small model, against DIS's 3.2 ms. On a small test set the 8-level model reached 34.1 dB where the earlier 3-level one reached 33.5.

Checkpoints record their architecture. One from an older model is refused with a message, not a crash: train a new run.

**Stop** finishes the current step and keeps the checkpoints. **Continue this run** resumes from `last.pt`. `best.pt` is the checkpoint with the best held-out quality.

```powershell
python -m mefi.train --data datasets\mygame --run runs\mygame --preset medium --epochs 60
```

## 3. Export

Export writes the checkpoint as ONNX in FP16 with FP32 input and output. Output is `models\ai_flow.onnx` next to `motion_enhancer.exe`, plus a `.json` with its resolution, size and score. **Measure speed** runs `motion_enhancer.exe --ai-flow-test` on it.

The network resolution is fixed at export, with a width that is a multiple of 64 (the program copies whole 256-byte rows). Use a quarter of the capture: 640×352 for 1440p, 512×288 for 1080p.

```powershell
python -m mefi.export runs\mygame\best.pt --out ..\..\build\Release\models\ai_flow.onnx --width 640 --height 352
```

Then choose **AI pyramid (trained model)** as the flow engine on the Live Capture page, or use `--flow-engine ai` on the command line. Without a model, the engine falls back to DIS.

## Files

| Path | What |
|---|---|
| `app.py` | the trainer window (PySide6) |
| `mefi/dataset.py` | triplets from videos |
| `mefi/data.py` | training/validation data, augmentation |
| `mefi/model.py` | the pyramid flow model and its exported form |
| `mefi/train.py` | training loop, losses, previews |
| `mefi/export.py` | ONNX export |
| `datasets/`, `runs/`, `.venv/` | your data, training runs and Python (not in git) |
