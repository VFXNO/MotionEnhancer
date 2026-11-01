# Offline text alternatives — 2026-11-01

## Outcome

**No candidate was activated.** None of the tested model/resolution combinations
reliably covers the supplied Arabic quest text, and none establishes the user's
maximum 4 ms AI budget. The existing icon detector, native per-pixel mask and
icon-only runtime remain unchanged. The rejected Text/HUD path remains removed.

The user authorized testing alternative pretrained text detectors offline,
including candidate downloads and cost evaluation, not training or activation.
The eight original game frames in
`build-project/Release/captures/20261101-124214/` are a **text-miss** example, not
a reproduction of the existing icon model's separately reported wrong boxes.
Wrong scenery boxes mentioned below are from these new offline candidates.

## Candidate provenance

Weights were fetched without executing downloaded code, pinned by repository
revision and verified against the host's LFS SHA256. Upstream licences/model
cards and source metadata were retained beside the temporary weights.

| Candidate | Repository / pinned revision | Source ONNX SHA256 | Licence |
|---|---|---|---|
| PP-OCRv6 tiny | `PaddlePaddle/PP-OCRv6_tiny_det_onnx` / `2ba1506c0380b8f0b03dd142459aac66d4421f6c` | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` | Apache-2.0 |
| PP-OCRv6 small | `PaddlePaddle/PP-OCRv6_small_det_onnx` / `28fe5895c24fd108c19eb3e8479f4ab385fbfc62` | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` | Apache-2.0 |
| CRAFT MLT | `inference4j/craft-mlt-25k` / `290e4d5e3abcaf319e20289816ae75532b4556a6` | `28c05f8b5e97f3c5765d00bca556ec2397c86c954d4ac5ef73aa10043db16388` | MIT |

These are text-localization experiments, not OCR transcription. The PP-OCRv6
cards describe English/Chinese detection; no general Arabic-language capability
was inferred from their model names. CRAFT's card identifies SynthText and
ICDAR 2013/2017 MLT training. Actual results on the supplied Arabic HUD, not
language claims, determine the recommendation here.

## Method and limits

- Separate standalone C++ DirectML probe using the existing application DLLs
  read-only; no application build or runtime routing change.
- RX 9070 XT, GPU-bound ONNX input/output. CPU EP fallback is explicitly disabled.
  Separate FP16 placement traces for all eight model/shape combinations contain
  only `DmlExecutionProvider` model events. This is not WMMA/ISA verification.
- Source-specific GPU normalization: RGB ImageNet for CRAFT; RGB-to-BGR then
  ImageNet for PP-OCRv6, matching its `inference.yml`. CPU input copies use
  center-aligned bilinear sampling, not Pillow's antialiased downsampling.
- Full original frames, aspect-preserving letterboxing. Shapes: 640x640,
  960x544 and 1280x736. The rectangular shapes reduce unused padding while
  preserving the content aspect ratio; there are no guessed HUD crops/strips.
- Static FP32 and FP16 graph copies; no weight training, calibration or quantized
  model activation. CRAFT's unused feature-map output is removed only in the
  temporary deployment copy. Source weights remain intact.
- Four calls per input in initial/profile/control runs; two opposite-order
  **unprofiled** rounds with twelve calls per input for cost. Each round uses
  all eight original game frames plus a uniform blank input. The first three
  calls per game frame are excluded from warm statistics: **144 measured warm
  game-frame samples per model/shape across the two rounds**.
- Same-input outputs are checked bitwise inside each session; replaying the first
  input after all other inputs must restore its exact map. Fresh/profiled versus
  fresh/unprofiled sessions are not claimed bitwise identical: map differences
  were observed, recorded and decoded separately. The text-miss/fragment outcome
  persists in both unprofiled rounds.
- Diagnostic decoding uses axis-aligned DB contours/CRAFT components, at most
  32 ROIs, rejecting empty/padding-only or >half-frame boxes. DB thresholds
  0.2/0.4 and unclip ratio 1.4 follow the downloaded configuration; axis-aligned
  padding approximates its rotated-polygon unclip. CRAFT uses low-text 0.4,
  link 0.4, max-component text 0.7, following the original CRAFT-style admission.
  This is not exact upstream rotated-polygon decoding or a deployed native mask.
- Manual quest/number rectangles measure **box-union reference coverage**, not
  text-pixel recall, recognition accuracy or a general game-HUD accuracy score.
  They are evaluation references only and do not influence model/box decoding.

## Coverage and cost

Times below are standalone FP16 **unprofiled worker wall time**, including
GPU normalization/Run and output readback/wait/copy. They exclude setup,
resize/input preparation/upload, component decoding, the existing icon detector,
native masking, flow and presentation. The application/game may run concurrently;
this is not a controlled gaming-load or end-to-end interpolation benchmark.

| Model | Input | Warm mean, round 0 / 1 (ms) | Largest warm sample (ms) | Largest session-first sample (ms) | Arabic quest outcome |
|---|---|---:|---:|---:|---|
| PP-OCRv6 tiny | 640x640 | 1.836 / 1.732 | 3.109 | 4.652 | No quest-text box |
| PP-OCRv6 small | 640x640 | 2.348 / 2.348 | 3.330 | 6.954 | No quest-text box |
| CRAFT MLT | 640x640 | 4.946 / 5.085 | 12.999 | 17.551 | No quest-text box |
| PP-OCRv6 tiny | 960x544 | 2.133 / 1.976 | 2.678 | 4.740 | No quest-text box |
| PP-OCRv6 small | 960x544 | 2.841 / 2.802 | 3.798 | 7.562 | No quest-text box |
| PP-OCRv6 tiny | 1280x736 | 2.952 / 2.858 | 3.952 | 7.628 | Small, inconsistent fragments only |
| PP-OCRv6 small | 1280x736 | 4.336 / 4.281 | 5.797 | 17.802 | Partial words, not complete lines |
| CRAFT MLT | 1280x736 | 9.800 / 9.769 | 12.919 | 34.411 | No quest-text box |

At 1280x736, PP-OCRv6 tiny covers 0–23.8% of the manual quest-title rectangle
and 1.9–5.7% of the instruction rectangle. Small covers 6.2–8.1% and 26.5–28.6%,
respectively. Annotated copies show fragmented words, not a complete quest ROI.
FP32 controls of small/CRAFT at 1280x736 retain the same basic failure; this is
not repaired simply by retaining FP32 precision.

HUD-number coverage improves with resolution: CRAFT at 1280x736 covers the
health-counter rectangle fully on these eight frames, while still missing the
quest. Some PP-OCRv6 candidates also admit icons and **small scenery false
positives**: e.g. tiny/960 `game_000` has an extra box over the right-hand rocks;
small/1280 `game_000` has an extra box over top-center foliage. These are not
evidence that the current runtime icon model produced those boxes.

All evaluated blank controls decode to zero regions. Python contour/component
decode times are retained separately in evaluation JSON; they must not be
treated as deployed native CPU cost or omitted when considering total cost.
Even a warm mean below 4 ms here does not establish a maximum, and adding a
second detector alongside icons consumes additional GPU time. Lower sampling
frequency or an asynchronous queue does not remove that contention.

## Evidence and preservation

Temporary experiment root:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-text-alternatives-20261101-p_73uq2d\`.

Key files: `sources.json`, each source's `source.json`/`LICENSE.txt`, static
copies' `preparation.json`, `initial-comparison.json`,
`higher-resolution-comparison.json`, `unprofiled-comparison.json`, per-run
`timings.json`/raw maps/placement traces/evaluation JSON and annotated copies,
`review/candidate-contact-sheet.png`, `final-summary.json`,
`protected-before.json` and `preservation-final.json`.

The interrupted first cost comparison intentionally stopped on a bitwise
cross-session difference; its completed first run was reused read-only in the
final comparison. No exact cross-session agreement is asserted. Earlier failed
preparation artifacts/logs remain diagnostic only; authoritative models are the
hash-recorded `static-*` folders. No failed candidate was installed.

All capture originals, runtime weights/licences/metadata, application EXE,
application sources/headers/shaders, protected flow models/checkpoints, dataset
index, trainer settings and installed program remain unchanged. Only offline
diagnostic tooling and validation documentation are added/updated in the
repository. No training, environment/package update, active application rebuild,
process closure, commit, push or installation was performed.

Validation: the standalone project builds from
`tools/ui-detect/offline-probe/CMakeLists.txt`; all **11/11** text-probe tests
pass, including actual GPU BGR normalization, changed-input readback and CPU-EP
rejection. The canonical application's **3/3** icon-only negative regressions
still reject the removed text choice and both former probability-map shapes.
SHA256 checks preserve **153** current protected files and recheck the previous
delivery's protected-original/model/shader/source manifests, including the
31 runtime shader binaries. The canonical EXE remains
`0432109bf38f8c77f92b5537ab825be095e938c1b360bdc46e814ed0fe82ce06`;
`MOTION_ENHANCER_STAGE_AI_FLOW_MODEL` remains `OFF`.

**Recommendation:** do not swap in these candidates or lower runtime thresholds
as a claimed fix. This case still needs a detector validated on small stylized
multilingual game text and representative negative scenes, within the actual
gaming-load budget. No training or further dataset work is authorized here.
