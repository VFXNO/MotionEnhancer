# Installed PyTorch checkpoint test — 2026-10-30

## Checkpoint and scope

Tested the user's completed, installed run without training or modifying it:

- Checkpoint: `C:\MotionEnhancer\tools\ai-train\runs\run1\best.pt`.
- SHA256: `126bdcfe2f182ec51bbf0b99107d9ca7296b770b6712396810e4253d67c7dfda`.
- Architecture: `pyramid-v2`, Large, **1,616,452 parameters**, saved after epoch
  **6**, dataset height **480**, saved moving PSNR **29.497951083713108 dB**.
- Hardware: RX 9070 XT; PyTorch `2.14.0+rocm10.1.0`, ROCm FP16 autocast.
- All checkpoint weights and every evaluated flow/mask output were finite.
- `best.pt` and `last.pt` contain identical model weights for this completed run.

Used a snapshotted checkpoint in the approved temporary directory. No model
was installed or replaced, no training was started, and no commits/pushes were
made. The separate native neural-feature trainer was not involved.

## Full held-out evaluation

Read the installed trainer's exact dataset loader/split to reproduce the GUI,
then evaluated **every one of the 4,586 held-out triplets**, not only its
96-sample preview/validation selection. There are 45,872 available triplets,
41,268 training triplets and no missing indexed images. The updated
overlap-safe splitter gives the same hold-out and training count on this data.

Scoring matches the trainer: RGB normalized to 0–1, full frames cropped to
multiples of 16, FP16-autocast network, FP32 midpoint reconstruction clamped
to 0–1, per-triplet PSNR capped at 100 dB, then arithmetic mean of those dB
scores. "Moving" means indexed 90th-percentile endpoint motion >= 3 pixels.
No flash scenes were removed or targets replaced.

| Selection | Count | Model PSNR | Plain-blend PSNR | Gain |
|---|---:|---:|---:|---:|
| Original GUI moving subset | 72 | 29.4980 | 26.5733 | +2.9246 |
| Original GUI all selected samples | 96 | 32.0853 | 29.3073 | +2.7780 |
| Balanced moving subset | 72 | 29.8192 | 27.4567 | +2.3624 |
| Balanced all selected samples | 96 | 30.6711 | 28.3768 | +2.2943 |
| **Entire moving hold-out** | **4,052** | **30.1314** | **27.3148** | **+2.8166** |
| Low-motion hold-out | 534 | 38.6424 | 36.3261 | +2.3164 |
| **Entire hold-out** | **4,586** | **31.1224** | **28.3640** | **+2.7584** |

The original GUI moving score reproduced the checkpoint's saved score
**exactly** (difference 0.0 dB). Other rows use different selections: their
different means are not evidence of further training or a quality increase.
The full evaluation took approximately **154 seconds**, including startup and
image loading.

Using a +/-0.01 dB tie band, the model beats blending on **3,565 / 4,052 moving
samples (88.0%)**, loses on **478**, and ties on **9**. Median moving gain is
**+2.4555 dB**; the 5th percentile is **-0.8251 dB**. Improvements are common but
not universal.

### Per-source moving results

Names are legacy indexed groups, not independently verified source identities.
In particular, the two historical `videoplayback` files remain inseparably
merged; this evaluation does not relabel or unmix their frames.

| Indexed group | Moving count | Model PSNR | Blend PSNR | Gain |
|---|---:|---:|---:|---:|
| BLACKBIRD gameplay | 55 | 28.8382 | 27.0187 | +1.8195 |
| Download (2) | 2 | 31.0257 | 29.9632 | +1.0625 |
| Flash/flicker background | 5 | 25.5594 | 29.4049 | **-3.8455** |
| Lily gameplay | 237 | 30.3360 | 26.4942 | +3.8418 |
| Professional Immortal Player | 749 | 27.9785 | 26.2673 | +1.7112 |
| Wait THIS GAME | 810 | 28.9398 | 25.4193 | +3.5205 |
| Wuthering Waves | 522 | 32.8428 | 29.7545 | +3.0883 |
| videoplayback (merged legacy group) | 1,672 | 30.8527 | 28.0572 | +2.7955 |

The nine held-out samples from the two dedicated flash/strobe sources average
**14.2300 dB for the model versus 16.3665 dB for blending**. All were retained.
Four strong return-flash samples have approximately **0.0684 dB** for both:
their bright middle target is not observable from the similar dark endpoints.
This does not establish performance on all ordinary gameplay flashes; the
dedicated-source test is small and includes deliberately edited flash footage.

### Learned-mask diagnostic

On all 4,052 moving samples, a neutral mask with the same predicted fields
scores **30.0072 dB**, versus **30.1314 dB** with the learned mask. Its average
contribution is **+0.1242 dB** in this training reconstruction. On the balanced
moving subset it instead costs **0.0308 dB**. This diagnostic is not the
native application's mask toggle: native synthesis/refinement is different.

## ONNX and native application checks

- Exported only the temporary checkpoint copy to **832x480**, matching the
  installed model's network resolution, with embedded weights and opset 18.
- ONNX checker passed. The export has 71 FP16 initializers; FP32 coordinate
  operations remain intentional. The installed ONNX has identical FP16
  initializer names/values, but the whole ONNX graph is not byte-identical.
- A standalone export initially hit Windows console code-page encoding in
  PyTorch's progress output; retrying with UTF-8, as the GUI already configures,
  succeeded without changing exporter code.
- DirectML loaded the exported model successfully. The existing
  `--ai-flow-test` reported **4.5272 ms mean / 1.2967 ms best**, over 200 runs,
  fence wait included. Timing varied substantially; this is not an end-to-end
  capture/frame-generation latency measurement or proof of matrix-core use.

Selected **12 diagnostic held-out cases**: median-gain representatives from
each of the nine legacy source groups, plus best/worst moving cases. These
are deliberately selected examples, **not an unbiased whole-dataset runtime
benchmark**. Each ran through:

1. AI + native DIS refinement, default learned mask off.
2. The same path with the experimental learned mask on.
3. DIS alone.

All **36 native GPU runs passed**, with saved outputs, no silent AI-to-DIS
fallback, and ground-truth comparisons. Tests used `t=0.5`, disabled temporal
UI masking for the independent pairs, and left the remaining offline defaults
unchanged. Input frames were the dataset's cropped decoded images, not original
full-resolution captures. Three representative AI cases also passed using the
actual installed executable and were pixel-identical to the repository build.

| Selected runtime cases | Count | Raw PyTorch | Native AI | Native AI + mask | Native DIS | Blend |
|---|---:|---:|---:|---:|---:|---:|
| All selected moving cases, including two flicker cases | 11 | 31.6032 | 31.9302 | 31.6549 | 31.1549 | 30.2996 |
| Selected non-flash moving cases | 9 | — | 33.9416 | 33.9207 | 34.1222 | — |

Across all 12 cases, default AI beat DIS by >0.01 dB in five, lost in five,
and was within 0.01 dB in two. The selected non-flash subset favors DIS by
**0.1806 dB** on average. AI helps some cases substantially, but these tests do
not establish a general quality advantage over DIS. Learned-mask effects are
also mixed. Raw trainer PSNR is not interchangeable with native midpoint PSNR.

Visible failures remain around animated weapons, disocclusions and building
edges. These are present in saved comparison sheets; a successful load/export
does not mean artifact-free interpolation.

## Known-motion controls

Six synthetic camera translations of one real textured 832x480 image were
evaluated without training. Endpoint error (EPE) is measured against the exact
forward displacement, excluding a border wider than the displacement. This is
a narrow sanity test, not a multi-scene optical-flow benchmark. Reported units
are **network pixels**, not full-resolution display pixels.

| True forward motion | Median predicted vector | Mean forward EPE | Median forward EPE |
|---|---|---:|---:|
| (0, 0) | (0.009, 0.003) | 0.0282 | 0.0243 |
| (4, 0) | (3.892, -0.008) | 0.3461 | 0.1233 |
| (8, 2) | (7.906, 1.780) | 0.9224 | 0.3528 |
| (16, 6) | (15.873, 5.923) | 1.3529 | 0.3573 |
| (32, 0) | (31.918, -0.056) | 1.8837 | 0.3868 |
| (64, 16) | (59.563, 15.592) | 10.7711 | 4.6916 |

The model can estimate ordinary translations and is not stuck at zero flow.
The largest-displacement control exposes much larger/localized errors. These
results do not prove whether architecture, data coverage or optimization is
the dominant remaining bottleneck.

## Preservation and artifacts

SHA256 checks after evaluation/native tests confirmed **all ten protected
files unchanged**: installed best/last/config/settings, repository trainer
settings, dataset index, and installed/repository runtime ONNX + JSON files.
Original frames and source videos were read only. No GPU drivers or packages
were installed; the existing nonfatal MIOpen tuning-database warning remained.

Generated scripts, CSV metrics, aggregate JSON, temporary checkpoint/ONNX and
12 labeled visual comparisons remain under:

```text
C:\Users\bsher\AppData\Local\Temp\opencode\mefi-pt-test-pkyjcsgu
```

Useful artifacts:

- `full-validation.csv` and `full-validation-summary.json` — all 4,586 scores.
- `runtime-results.json` and `runtime-summary.json` — native diagnostics.
- `runtime-cases\case-07\comparison.png` — ordinary camera-motion example.
- `runtime-cases\case-06\comparison.png` — animated weapon/occlusion failure.
- `runtime-cases\case-03\comparison.png` — flicker example.
- `known-motion-results.json` — exact-displacement controls.
- `protected-hashes.json` — input preservation manifest.

Not performed: RIFE comparison, a full 4,586-case native-engine evaluation,
full-resolution capture tests, long temporal playback tests, or medium-vs-large
training comparisons. No claim of further trained-model improvement is made.
