# Temporal UI export and opt-in runtime

User request: “it's finished but I can't export it; also add an option to the
programme to run it.” The user subsequently closed Motion Enhancer and approved
updating the normal executable. No training, label preparation, dataset reads,
game capture or automatic activation was performed by this implementation.

## Installed result

- Rebuilt `build-project/Release/motion_enhancer.exe` and UiMaskCS. The first
  link attempt was blocked by the open application. A separate
  `motion_enhancer_ui.exe` was used for initial tests; the normal executable
  was rebuilt successfully after the user closed the application.
- Exported `tools/ai-train/runs/ui/best_ui.pt` at **832×480**, Medium,
  eight state channels and seven pyramid levels, into the four separate
  `build-project/Release/models/ai_flow_ui.*` files.
- Runtime format `mefi-pyramid-ui-1`, bundle
  `feb90c1c26234a6b8f5df3384c358d74`; primary ONNX SHA256
  `2638e7bb396b08f3cd61c9f2c0606cbcd3c5817587468baecd8243ade815c75f`.
- New **Trained temporal UI (exp.)** checkbox, gated on AI pyramid, static UI
  masking and available UI export. Default remains off. CLI `--ai-temporal-ui`;
  optional `MOTION_ENHANCER_AI_UI_MODEL` override. No implicit model staging.
- Frozen/joint checkpoints are converted explicitly, not passed to the old
  flow-only loader. Existing legacy exports and their default paths remain.

The exported best checkpoint is **step 1 / one UI update**, frozen flow. Its
stored validation has **0 recall, 0 IoU**, 0 TP, 0 FP and 1,751 FN. “Finished”
does not mean this head has learned useful detection; later `ui_validation.json`
also reports zero recall. Export GUI now displays this warning. Nothing here
establishes detection accuracy, good labels or a sufficient training budget.

## Safeguards

- UI exports are separate from `ai_flow.onnx`; the canonical flow name is
  redirected to `ai_flow_ui.onnx`. Cross-format overwrites are refused. GUI
  copying uses the corresponding canonical runtime filename.
- Static FP32 input/output tensors are checked by exporter and native loader;
  internal learned computation remains FP16. The runtime allow map is expanded
  to network resolution to preserve D3D12 row alignment. Research output shapes
  are unchanged.
- Flow and UI share one backbone execution. Native DirectML binding owns two
  distinct state buffers on the serialized flow queue. Immutable upload data
  supplies resets and valid flags; there is no asynchronous CPU overwrite.
- Startup, nonoverlap, skipped IDs, replay/replacement pairs and resize reset
  history. Capture skips scene-cut pairs, breaking overlap. Only a successfully
  submitted pair commits the next state index. Model/refiner switches require
  restart. Explicit missing/incompatible UI models fail instead of fallback.
- Finite candidate probability >=0.5 admits native per-pixel static/damage
  testing; it does not fill a mask, freeze boxes, or write motion. Optional
  icon-region admission is unioned, preserving that detector's own behavior.

## Checks

- **53 CPU/export/GUI tests passed**: checkpoint hashes, UI training safeguards,
  original flow/refiner interfaces, paired export copying and new UI exports.
  Generated fixtures only; training optimizer updates remain forbidden in the
  UI trainer tests.
- Generated 128×64 refined UI graph and headless companion: ONNX checker,
  DirectML binding, numerical output comparison, mixed-bundle rejection and
  Win32 checkbox default/engine/static-mask/Refresh gating passed.
- Synthetic DirectML MAEs (DIS/integrated): flow 0 / 0.000954, blend mask
  0.000061 / 0.000061, candidate probability 0.000000954 / 0.000000954,
  projected state 0.0000515 / 0.0000515.
- **Actual exported 832×480 model:** `--ai-temporal-ui-self-test` passed 48 pairs
  across DIS and integrated variants on an AMD Radeon RX 9070 XT. Covers
  startup/gap/replay/resize/ring reuse, finite/ranged candidates, zero candidates
  with invalid history, and no native protection earned by a flat unchanged
  scene even if the head proposes candidates.
- No CPU fallback is allowed in the temporal DirectML session.
- On generated noise tensors at the actual trained resolution, the UI graph's
  flow and blend-mask outputs are **bit-identical** to separate flow-only
  DirectML exports of the same saved backbone, for both refinement variants.
  PyTorch CPU also computes identical flow/mask with and without the feature
  tap. This tests preservation of the backbone, not its accuracy on games.
- Cross-backend FP16 output is not bit-identical: DirectML versus PyTorch CPU
  MAEs were about 0.040–0.041 flow pixels, 0.00604 blend preference, 0.0000651
  candidates and 0.00964 state. An initial GPU-reference diagnostic exceeded
  a provisional 0.01 state-MAE check (0.01126); it was retained, not treated as
  a strict numerical-parity pass. Subsequent same-backend comparison above
  showed no flow/mask differences introduced by the UI graph. No stringent
  cross-backend numerical-parity/quality guarantee is claimed.
- SHA256 checks confirmed `runs/ui/best_ui.pt`, `runs/ui/last_ui.pt` and all
  four installed `ai_flow.*` files unchanged after export. User settings and
  unrelated existing source changes were not reset or overwritten.

Detailed local logs/artifacts:

- `%LOCALAPPDATA%/Temp/opencode/ui-runtime-20261101-6c8a`
- `%LOCALAPPDATA%/Temp/opencode/ui-deployment-20261101-c73d`

The pipeline self-test is bounded synthetic input, not real game capture or an
in-flight stress test. Real-world candidate quality, capture scheduling under
load and combined-AI gaming latency remain unverified; no 4 ms guarantee.
