# Temporal UI head: staged training and validation plan

**Status: prototype and opt-in training implementation authorized; no real job authorized, 2026-11-01.** After the original plan, the user said **“OK GO BUT I WANT THIS FEATURE ALSO WHEN I TRAIN THE MODEL FROM SCRTATCH.”** That first approved the isolated prototype, synthetic checks and feasibility diagnostics, not data preparation or real training. Budgets and acceptance thresholds below remain proposals, not approved jobs. See [prototype usage](../../tools/ui-temporal/README.md).

**Later authorization: “now implement it for training.”** This additionally approves implementing opt-in trainer GUI/CLI support for scratch and frozen modes, dense-label ingestion, separate versioned checkpoints and continuation. It does **not** launch/approve a training job, process existing data, activate outputs, export deployment models or authorize runtime integration. See [training implementation and label contract](../../tools/ai-train/UI_TRAINING.md). Flow-only training and its original objectives remain unchanged.

**Labeling-tool authorization:** the user subsequently requested labeling tools with optional AI guidance or manual GUI editing. This approves implementing the Label UI editor, explicitly requested local guidance adapters, reviewed-only label export and generated-fixture tests. It does not approve downloads, processing existing recordings/datasets, or training. See [labeling usage and supported model contract](../../tools/ai-train/UI_LABELING.md). Automatic suggestions are drafts, not independently verified labels.

The [labeling implementation report](../validation/ui-labeling-tools-2026-11-01.md) records 85 passing generated/no-update/regression checks and the remaining real-model/data limitations.

**Later model/adaptor authorization:** the user requested an Internet model or automatic process and chose **“add both”** after being offered EfficientSAM-Ti smart click/box selection and automatic candidate masks, including the pinned model download and GUI adapter. The 48,312,857-byte Apache-2.0 model is stored separately in the user's local labeling cache. Generated-input model/GUI checks are allowed; existing recordings/datasets, training, package installation and runtime integration remain outside this authorization. See [AI labeling usage](../../tools/ai-train/UI_LABELING.md).

The [EfficientSAM implementation report](../validation/ui-labeling-efficientsam-2026-11-01.md) records 112 repeatable checks plus actual pinned-model smart/grid/GUI smoke checks on generated shapes. Those tests establish operational plumbing, not game-HUD labeling accuracy.

## Recommended approach

Train a small, independently designed **UI candidate head on detached MotionEnhancer flow features**. Support both a frozen-backbone experiment and a new flow model trained from scratch with the original flow objective. Preserve the working icon detector and the existing full-resolution native pixel test. The new head would propose additional pixels where that native test is permitted to run; it would never fill rectangles, replace native protection evidence, or directly zero motion vectors.

This design reuses MotionEnhancer's pairwise backbone activations to predict dense screen-space UI candidate probabilities from overlapping frame pairs. It adds no second image encoder and does not directly attenuate flow. Accuracy and latency require independent validation.

### What the source actually supports

| Observed fact | Consequence |
|---|---|
| The active export metadata specifies Medium `pyramid-refine-v1`, 832×480, bundle `dbd389e4f8d7485da0dbfde1714205a0`. | Pin this exact baseline before future experiments; do not change the flow preset or resolution to make UI timing look better. |
| The model processes **pairs** of RGB frames, warped frames, accumulated flow and blend logits. It does not independently encode each captured frame. | Correlation across independent per-frame descriptors cannot be applied directly to these pair-conditioned activations. |
| The finest block has reusable features immediately after `body(stem) + stem`, before its original output/refinement heads. Medium uses 40 channels there, at half network resolution: 416×240 for the active export. | A research wrapper could expose/compress this existing activation without running another image encoder. Its usefulness for UI remains untested. |
| Export/runtime currently support exactly `frames -> flow, mask`. `mask` is a **midpoint blend preference**, not UI probability. | Never repurpose that output. Extra research outputs require a separate versioned graph and loader. |
| AI refinement on selects the integrated graph without DIS; off selects the paired backbone graph plus DIS. | Preserve both behaviors and the independent blending control. Validate both variants before any later integration. |
| AI UI regions currently gate `UiMaskCS` through fresh icon boxes. Native damage/change/history tests choose protected pixels. | A dense candidate map could extend admission, but must not bypass those native tests. |

Source references: [model.py](../../tools/ai-train/mefi/model.py), lines 101–168, 179–250 and 287–298; [export.py](../../tools/ai-train/mefi/export.py), lines 48–78; [AiFlowEstimator.cpp](../../src/AiFlowEstimator.cpp), lines 172–184 and 222–238; [FlowEngine.cpp](../../src/FlowEngine.cpp), lines 1065–1111; [UiMaskCS.hlsl](../../shaders/UiMaskCS.hlsl), lines 47–123. The metadata's checkpoint path alone does **not** prove that a file currently at that path still matches the active exports.

**Read-only follow-up:** [parameter verification](../validation/flow-checkpoint-identity-2026-11-01.md) confirmed that the current `runs/best.pt`, SHA256 `620c208022fd421b3fe08097428747d5017e48a8877e1bfd2d6492103af4f091`, matches all 64 integrated and 58 paired-backbone FP16 parameter tensors. This pins compatible stored weights, not unique historical provenance. The source path remains mutable; no immutable copy or trained-wrapper inference/export was made.

Lossless evidence, collected read-only: UI dispatch `ev_2814b31d8cbd3601e83c942e66a8e902f83833cedd02bdbc1d058ddba563cccf`; frame encoder/ring `ev_40f4e9262b4f41627ff7603ff78f9973e31c7d2292994994b1332b6bc8bd000d`; texture/constants `ev_cc7cb99538fb1437502eb1b78bcf78629294bd0038dfc1f8b2550910ead0022e`. Detailed local static report: `C:\Users\bsher\AppData\Local\Temp\opencode\lossless-ui-analysis-20261101-ih0iuhmw\report.md`. These observations do not supply permission to import proprietary assets or establish behavior on the user's Arabic capture.

## 1. Architecture to test, not activate

Use a causal history of three captured frames, without waiting for a future frame:

```text
Previous pair: (F[t-2], F[t-1]) -> flow backbone -> detached features -> projected state Qprev
Current pair:  (F[t-1], F[t])   -> flow backbone -> detached features -> projected state Qcur
                                                   \-> original flow and blend outputs

UI head: concatenate(Qprev, Qcur) -> small spatial head -> ui_allow probability
Cache: Qcur, associated with this pair's frame IDs and capture generation
```

- Tap the finest block's existing 40-channel body feature. Start with a trainable 1×1 projection to eight channels, shared across the two pairs.
- Start the UI head with 1×1 channel mixing, two small depthwise/pointwise 3×3 blocks, and a one-channel sigmoid output. Hidden width 16 is a pilot choice, not an accuracy or matrix-acceleration claim.
- Do **not** interpret pair features as per-frame descriptors or apply a dot-product cost volume across them without a separate representation study.
- Deployment would run the flow backbone once per new pair and retain only the projected previous state. UI training evaluates the two overlapping pairs without backbone gradients; recompute both projected states so the shared projection receives gradients from both pairs rather than training against stale cached states.
- In frozen mode, keep the backbone in evaluation mode and construct the optimizer from the UI projection/head parameters only. In scratch mode, use separate flow/UI parameter groups: original flow supervision trains the backbone while detached UI supervision trains only the projection/head. No changes to the existing flow targets/losses are permitted in either path.
- First compare a current-pair-only head against the same head with valid previous-pair context. A temporal head must demonstrate a benefit; three-frame input alone is not evidence of one.
- Start with a 416×240 candidate map. If small-glyph recall is inadequate, separately test a higher-resolution head. Fine strokes below the feature sampling resolution may be lost; upsampling cannot restore details absent from the features. Do not silently increase the flow model's resolution or add a second encoder.
- Use FP16 as the initial export candidate, with FP32 input/output contracts as required by the existing runtime. INT8, UNORM feature quantization, and explicit matrix backends are out of this pilot; each needs separate accuracy and placement evidence.

### Research graph contract

The isolated prototype implements the identifiers below; these are **not accepted application model formats**:

```text
format: me-ui-pair-temporal-1
inputs:
  frames          [1, 6, 480, 832]    current pair, existing normalization
  previous_state  [1, 8, 240, 416]    projected prior-pair features
  history_valid   [1, 1, 1, 1]        0 or 1; host verifies frame-chain identity
outputs:
  flow            [1, 4, 480, 832]    original output, unchanged semantics
  mask            [1, 1, 480, 832]    original blend output, unchanged semantics
  ui_allow        [1, 1, 240, 416]    current-frame screen-space candidate probability
  next_state      [1, 8, 240, 416]    current projected features
```

`history_valid=0` must force the additional `ui_allow` contribution to zero; it must not reinterpret zero state as valid evidence. The current projection can still seed the next valid pair. No future-frame buffering is introduced, but startup needs two overlapping pairs before temporal admission becomes available.

The research wrapper reuses the finest block computation; it does not run another encoder or execute that block twice. Adding outputs can change ONNX/DirectML fusion, materialization and timing even when weights are frozen. Check parity and cost **before** spending a training budget.

## 2. Prediction target and safe future integration

Train **UI presence/candidate coverage**, not “always freeze this pixel.” Include HUD text, icons and panels. Animated or changing UI can still be UI; the native change/release rules remain responsible for deciding whether it is safe to protect a pixel now.

Dense labels must distinguish UI from scenery, including text on signs, world-space nameplates, and stationary world detail. Ambiguous pixels are ignored, not automatically background. Coverage/antialiasing and compositing opacity are recorded separately; opacity is not a calibrated protection confidence.

Only after independent approval and successful validation, an opt-in runtime could admit:

```text
allowed(pixel) = existing_fresh_icon_allowlist(pixel)
                 OR fresh_temporal_ui_candidate(pixel)

protected(pixel) = existing_native_pixel_evidence(allowed(pixel))
```

The candidate map extends only admission to the native pixel test; it does not attenuate flow directly. Keep native damage/change/confirmation/neighbour/release calculations and synthesis unchanged. A candidate map or box alone must never create protection. A failed temporal branch contributes nothing; it must not disable the preserved icon path or enable unrestricted full-screen masking.

Future integration requirements:

- Off by default; disabled must use the existing graphs and produce unchanged behavior.
- Initially supported only with the validated AI flow variants. Do not turn on AI flow silently for DIS/FFX users or invent standalone features on those engines.
- Pin checkpoint/export/head hashes, preset, resolution, variant, tensor meanings and coordinate transform in versioned metadata. Reject incompatible bundles.
- Reset/reject temporal state after resize, cuts, capture restart/generation change, missing/out-of-order frames, device failure, or variant switch. Match both overlapping frame IDs, not just recycled capture slots.
- Handle timeline replacement pairs using the matching predecessor, never a state from the replaced future pair. Test this separately from ordinary sequential playback.
- Own state and candidate maps across in-flight submissions; no presenter may read a buffer overwritten by later inference. Include actual queue/fence and memory costs.
- Preserve DirectML-default/Matrix-opt-in icon selection, thresholds, continuity, expiry and 32-box limits. Do not restore the rejected Text/HUD model.

## 3. Data approval and labeling plan

**Do not run the existing dataset builder, scan/process training datasets, or relabel captures under this approval.** A UI task needs its own labels and protocol; the existing flow triplets are not HUD ground truth.

Before preparing data, obtain an explicit allow-list of source videos/sequences, approved fonts/assets and a new destination. Keep originals, existing datasets/checkpoints, and installed files read-only. Record hashes, provenance, source timestamps/frame IDs, label versions and ignored pixels. No uploads or external labeling service.

Proposed bounded pilot, subject to source availability:

- At least 12 independent sequences with genuine consecutive frames; reserve 6 for training, 3 for validation and 3 for locked testing. Prefer at least three games and keep one game's sequences entirely out of training; if that is impossible, report the weaker same-game scope.
- Up to eight manually labeled current-frame targets per sequence (96 total). Each target needs its two actual preceding frames. This is a feasibility pilot, not enough evidence for production generalization.
- Cover Arabic/RTL quest text, Latin text and digits, icons, opaque and translucent HUDs, antialiased edges, fades/animated counters, appearing/disappearing UI and different UI scales. Include CJK and additional scripts only with approved examples; do not claim multilingual coverage from Arabic/Latin alone.
- Include difficult negatives: world-space text/nameplates, walls/foliage, repeated patterns, world detail behind panels, camera pans/stops, flashes, transitions and UI-free frames. The separately reported wrong-area detections require a separate labeled set; the eight-frame Arabic case is not evidence of those errors.
- Split by original source/game **before** cropping or augmentation. Keep related time windows, renamed copies, overlays and derived crops in one split. Add temporal guards at boundaries; never let overlapping three-frame windows cross splits.
- Optional synthetic overlays require separate approval. Use licensed assets and a renderer verified to shape Arabic correctly. Composite on approved HUD-free backgrounds, retaining exact per-pixel coverage, opacity and animation metadata. Synthetic-only success is not a game-HUD accuracy result.
- Do not use unreviewed third-party model predictions, icon proposals, the rejected text detector, flow teachers, or the current native mask as ground-truth labels. The separately authorized labeling editor can display local dense AI predictions as drafts for explicit human correction/review; confidence alone is not review. Boxes can help review but are not dense glyph truth.

Keep `build-project/Release/captures/20261101-124214/` as a **locked text-miss regression**, not training data. Existing reference rectangles measure localization, not glyph coverage. Confirm chronology and spacing before treating the saved originals as temporal input; if insufficient, request a fresh consecutive capture. Repeating still images is only a plumbing test.

## 4. Proposed head-only optimization

These are initial experiment settings, not changes to `mefi.train`:

- Frozen, verified Medium backbone and refiner; only the new projection/head receives gradients.
- Start with ignored-pixel-aware weighted BCE plus 0.5 soft Dice on UI candidate coverage. Normalize over valid pixels and handle empty-positive samples explicitly. Record class weighting and sampling rules in the new run manifest.
- Initially no temporal smoothness loss: it can conceal flicker by retaining disappearing UI. Temporal behavior is measured on held-out sequences. Introduce a consistency term only as a separately reviewed ablation using reliable unchanged-label regions.
- Proposed AdamW: learning rate `3e-4`, weight decay `1e-4`, batch up to 8, aligned crops up to 448×256 in **network coordinates**. Preserve text scale; use identical crop/photometric transforms across each three-frame window. Do not mirror Arabic into invalid script examples.
- Save a new UI-only checkpoint with head architecture, frozen-backbone identity, data/split/label identities, optimizer/scheduler/RNG and next-batch state. No writes to existing runs, no legacy resume or checkpoint conversion.
- Pilot cap: **30 minutes of active GPU training or 2,000 optimizer steps, whichever occurs first**, one seed. Record real elapsed time, device, memory and sample counts. No automatic continuation, sweep or larger run. More data/seeds/training require another budget approval.
- Select checkpoints/thresholds on validation only. Keep the test split and Arabic regression locked; failures do not authorize lowering the icon threshold or tuning repeatedly on test frames.

### From-scratch flow + UI path

The isolated `TemporalUi.from_scratch` API initializes both branches without a checkpoint. The research objective adapter accepts the original flow interpolation batch and a **separate, densely labeled causal UI window**; it calls the existing `mefi.train.losses` unchanged, then adds UI loss on detached features. No UI loss reaches the flow backbone, refiner or blend branch. This preserves the requested future from-scratch capability without reinterpreting flow truth as UI labels.

The subsequent training implementation exposes this path in a separate `mefi.train_ui` runner and opt-in GUI controls, with versioned continuation. UI checkpoint format is deliberately rejected by the current runtime exporter. Before running a scratch job, approve its exact source/label manifest, independent UI sampling schedule, simultaneous updates versus a separate future stable-backbone warmup study, and a **separate full-flow training budget**. The 30-minute/2,000-step head-only proposal above does not authorize or size a from-scratch flow run. Actual scratch training, automatic warmup and additional teacher/label generation have not been performed or authorized.

## 5. Gates before any runtime adoption

All numerical gates below are **proposed acceptance criteria to agree before training**, not measurements already achieved.

### A. No-training feasibility gate

Stage 1 uses generated synthetic tensors only to verify tap shapes, history use, output identity and DirectML placement. No dataset job is needed for this gate. Passing random-weight tests does not close the active-checkpoint parity requirement below.

- Verify the approved checkpoint really matches the active exported model; do not trust the mutable `runs/best.pt` path in metadata alone. The read-only follow-up above establishes exact FP16 parameter identity at the recorded hashes. Before any separately approved trained-wrapper experiment, reverify those hashes and use a checksum-pinned immutable copy supplied/approved for the experiment.
- Show mathematically unchanged flow/blend branches and unchanged frozen tensors. Test FP32 wrapper parity and FP16 ONNX/DirectML parity against the canonical graph on both refinement variants. Record max/percentile errors in network and source pixels; predeclare tolerances before evaluating data, and stop on unexplained changes.
- Compare original graph, feature-tap-only graph and tap-plus-untrained-head graph. An untrained head's output is discarded: this measures plumbing/overhead, not detection accuracy.
- Confirm every experimental model node executes on DirectML or fail the experiment; no silent CPU execution. GPU placement alone is not matrix-ISA proof.
- Stop before labeling/training if materializing features already exceeds the available latency or memory budget.

### B. Detection and native-mask safety gate

Report both raw candidate maps and final protected pixels, at original capture resolution:

- Proposed candidate pixel precision at least 99% and Arabic/Latin glyph recall at least 90% on each relevant held-out category, with explicit boundary/ignore rules. Report per-sequence counts, worst cases and confidence calibration; pooled averages must not hide a failed script or UI style.
- Proposed protected-pixel false-positive rate outside annotated UI at most 0.01% on the separate negative suite, with no persistent coherent scenery region wrongly held. Zero visible persistent scenery sticking remains a required qualitative check.
- Demonstrate text-miss improvement over the current icon-only admission without reducing icon coverage or weakening native changing-UI release. If the capture does not contain enough valid history, report that limit instead of inventing it.
- Measure acquisition delay, disappearing/changing-UI release, frame-to-frame mask flicker, glyph-edge preservation and background sticking. Confirm low candidate probability does not leave stale native evidence.
- Evaluate static, slowly moving, translucent and animated HUD separately; report unresolvable cases instead of promising universal detection.
- Regression tests: icons-only unchanged when disabled; invalid/empty candidate contributes zero; no rectangle/map filling; gap/resize/cut/replacement invalidation; both refinement modes; blending independent; no reused in-flight state; unchanged flow/synthesis.

### C. Gaming-load latency gate

Retain the **4 ms maximum AI-latency target**. Report incremental UI cost and the combined AI work, including the existing icon workload, without redefining the target as network-only time.

- Measure preprocessing, feature-tap materialization, temporal state transfer, head inference, candidate-map conversion, synchronization and readback if any. Also report full flow/native-mask/synthesis/presentation cost and frame pacing separately; disclose which work each timer covers.
- Use the active 832×480 model, both correction modes, and real game/capture resolutions. Compare baseline and candidate in opposite orders with identical game scenes/settings. Include cold startup separately, steady state, clock variation and actual gaming GPU contention.
- Proposed validation budget after training: four 10-minute loaded sessions, at least two games, two opposite baseline/candidate orders. Measure every submitted pair and retain spikes/missed deadlines rather than discarding them as outliers.
- Any observed >4 ms eligible AI sample fails the tested maximum target. Report counts, p50/p95/p99/max and measurement uncertainty. Queue/submit time, warm averages, low-rate execution and asynchronous overlap do not establish a deadline guarantee.
- Even zero observed violations does not prove an absolute maximum under every OS/driver/game workload. If a hard universal guarantee is required, say it is unproven rather than certifying it from a finite benchmark.

## 6. Files, destinations and staged authorization

Research diagnostics remain in `tools/ui-temporal/`. The shared head and new opt-in training modules live under `tools/ai-train/mefi/`; the separately authorized GUI selector routes to `mefi.train_ui`, never silently replacing `mefi.train`. Existing flow-only model/train/loss/export code, defaults and checkpoint schema remain unchanged. New UI runs use their own non-deployable checkpoint format, not an in-place conversion of existing runs.

If later approved, use a **new** experiment directory under `C:\Users\bsher\AppData\Local\Temp\opencode\` for generated data, checkpoints, experimental ONNX files and diagnostic build outputs. Research graph exports are not application/trainer deployment exports; no automatic copying or registration as an alternate active export destination is allowed. Metadata must clearly mark them research-only and incompatible with the current application loader.

`build-project` remains the sole active application build and trainer deployment export destination. Flow-model staging remains **OFF**. No writes to its runtime models/executable, `C:\MotionEnhancer`, SDKs, branding, backups, supplied capture originals or `Lossless.dll`. No environment/package upgrades, commits or pushes are part of this plan.

| Stage | Proposed scope | Authorization |
|---|---|---|
| 0 | This source-grounded design document | Approved and completed. |
| 1 | Isolated wrapper/head/synthetic tests and GPU feasibility benchmark; include scratch/frozen research APIs, no training/data preparation | Approved; synthetic scope completed. Follow-up stored-parameter identity verified; immutable experiment copy and trained-wrapper behavioral parity still pending. |
| Training implementation | Opt-in GUI/CLI, independent dense-label loader and versioned scratch/frozen training continuation; no real job/data preparation | Approved separately by “now implement it for training”; implemented, 50 no-update tests passed. |
| Labeling implementation | Manual pixel GUI, optional local AI drafts, reviewed-only immutable label export and generated-fixture tests | Separately requested and implemented; no existing recording processing, model downloads or training. |
| 2 | Prepare labels/data in a new directory from an exact approved source list and split | Not approved. |
| 3 | One capped head-only pilot described above; any full-flow scratch job needs its own budget | Not approved. |
| 4 | Locked accuracy/native-mask/load evaluation, including game execution/capture | Not approved. |
| 5 | Optional application integration, canonical build and deployment | Separate approval only after evidence passes; not approved. |

### Decisions needed to proceed

1. Stage 1 synthetic work and the separate read-only parameter check are complete. Before a trained-wrapper experiment, approve an immutable copy of the checksum-pinned compatible checkpoint and the exact inference/export/benchmark scope. Parameter identity alone is not trained-model behavioral parity or latency feasibility.
2. Before Stage 2, name approved training/validation/test sequences and labeling budget, plus any permitted synthetic assets. Confirm the Arabic capture remains test-only.
3. Before Stage 3, approve the immutable backbone checkpoint and the proposed 30-minute/2,000-step pilot cap; agree on label semantics and quality/latency acceptance criteria.
4. Approve specific game/load captures before Stage 4. Passing a pilot does not authorize runtime activation or additional training.

## Work performed

The original planning stage inspected source and export metadata only. The separately approved Stage 1 added the isolated prototype and synthetic CPU/DirectML diagnostics. Generated random research weights were saved/reloaded and exported to temporary research graphs; no existing trained checkpoint was loaded. No dataset/capture images were processed, optimizer steps run, application built, active model exported/staged or runtime integrated. Existing application/trainer/model source files remain unchanged. This does not establish HUD/text accuracy, active-checkpoint parity or a 4 ms gaming-load guarantee.

See [Stage 1 results and limitations](../validation/temporal-ui-prototype-2026-11-01.md): 27 tests passed; both synthetic refinement variants ran on DirectML. The paired baseline retained a >4 ms warm spike, and first-run costs were higher still. Do not advance to training or claim the maximum-latency target from these isolated random-weight measurements.

The subsequently selected **Verify checkpoint** step loaded the existing checkpoint safely as CPU tensor data and compared it with the two existing ONNX files. All stored-parameter checks passed without model execution, re-export or source/runtime model changes. The [identity report](../validation/flow-checkpoint-identity-2026-11-01.md) records the exact hashes and remaining limits.

The separately requested training implementation adds opt-in GUI/CLI code and synthetic/fully mocked no-update tests. No real training, existing dataset/capture processing, trained-checkpoint inference, export or application activation is part of that implementation task. Full UI learning and real GPU/AMP continuation remain to be evaluated under an explicitly approved job.

See [training implementation results](../validation/temporal-ui-training-implementation-2026-11-01.md) for the changed files, 50 checks, preservation results and remaining limits.
