# Temporal UI research prototype

**Current runtime update:** the separately authorized trainer/runtime workflow
now converts actual `best_ui.pt` / `last_ui.pt` checkpoints into an experimental,
default-off `ai_flow_ui.onnx` bundle. See [export and runtime instructions](../ai-train/UI_TRAINING.md#export-and-run-the-trained-ui-model).
The research fixture tools and restrictions documented below remain separate;
their untrained snapshots are not accepted by the runtime exporter.

**Untrained and not active in MotionEnhancer.** Independent UI
candidate head using a shared projection of detached features from overlapping
frame pairs.
The shared model now lives in `tools/ai-train/mefi/ui_model.py`; this directory
retains the isolated research/synthetic diagnostics. Separately approved
[opt-in trainer GUI/CLI support](../ai-train/UI_TRAINING.md) is implemented.
The flow-only workflow, flow objectives, exports and application are unchanged.

## Both requested training paths

- **Frozen:** `TemporalUi(backbone, mode='frozen')` copies and freezes the supplied
  backbone; only projection/head parameters appear in its optimizer groups.
  A real experiment still needs a verified immutable backbone and approved labels.
- **From scratch:** `TemporalUi.from_scratch('medium', refiner=True)` creates new
  flow and UI weights without any checkpoint. Its parameter groups separate flow
  and UI. `training.losses` passes the original interpolation batch verbatim to
  `mefi.train.losses`, and adds the independently labeled UI objective.
- **Both:** UI features are detached from flow. UI supervision cannot change
  flow/refinement/blend gradients. Three causal RGB frames for UI supervision are
  separate from the original flow interpolation triplet/target. UI training
  recomputes both overlapping pairs; deployment would cache the projected state.

The same model/objective contract is now available through the separate
`mefi.train_ui` runner and opt-in GUI controls, including versioned continuation
and UI-validation checkpoint selection. **No training has been launched.** Data
preparation, actual training and application activation still need approval.
The proposed head-only pilot budget does not authorize training the flow
backbone from scratch. Research `snapshot.py` below remains weights-only; it is
not the trainer's new resumable checkpoint format.

## Contract and safeguards

At 832×480, format `me-ui-pair-temporal-1` takes FP32 external inputs:

| Input | Shape |
|---|---|
| `frames` | `[1,6,480,832]`, RGB pair in 0..1 |
| `previous_state` | `[1,8,240,416]` |
| `history_valid` | `[1,1,1,1]`, exactly 1 only after host identity checks |

Outputs are original `flow`/`mask`, `ui_allow` `[1,1,240,416]`, and `next_state`
`[1,8,240,416]`. Internal export computation is FP16. `mask` remains blend
preference, never UI probability. Invalid/nonfinite history contributes zero.
`History` is an offline ownership reference for generation, size, bundle/variant,
overlapping frame IDs and replacement pairs; it is **not** a GPU fence manager or
scene-cut detector. The caller must signal cuts/gaps and supply reliable IDs.

No candidate is used by the app. Future candidates may extend admission to the
existing native per-pixel test, never fill boxes, replace native evidence or zero
flow directly. The working icon detector and all its rules are preserved.

`snapshot.py` exclusively creates checksum-pinned, weights-only research files
with `deployable: false`; it refuses legacy flow checkpoints. It does not claim
optimizer/RNG continuation or training-source provenance. Research exports and
probe builds stay in **new** directories under
`C:\Users\bsher\AppData\Local\Temp\opencode`. Never use a runtime export path.
`build-project` remains the only active application build/deployment destination;
flow-model staging stays OFF.

## Tests and explicit synthetic diagnostics

From the repository root, using the existing environment (no installation):

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = "$env:LOCALAPPDATA\Temp\opencode"
$env:TMP = $env:TEMP
& tools/ai-train/.venv/Scripts/python.exe tools/ui-temporal/test_temporal.py -v
```

Optional research exports below are generated random tensors/weights, not a
trained model or dataset. All named output directories must be new. Exporting
does not run a GPU benchmark or activate anything.

```powershell
$root = "$env:LOCALAPPDATA\Temp\opencode\<new-experiment>"
& tools/ai-train/.venv/Scripts/python.exe tools/ui-temporal/export_probe.py `
  --from-scratch --preset medium --refiner --width 832 --height 480 `
  --untrained-fixture --out "$root\integrated"

# Preserve the same non-refiner and UI weights for the paired bypass:
$sha = (Get-Content "$root\integrated\preparation.json" -Raw | ConvertFrom-Json).weights_sha256
& tools/ai-train/.venv/Scripts/python.exe tools/ui-temporal/export_probe.py `
  --snapshot "$root\integrated\research-weights.pt" --snapshot-sha256 $sha `
  --without-refiner --untrained-fixture --out "$root\paired-backbone"

cmake -S tools/ui-temporal/offline-probe -B "$root\native-build" -A x64
cmake --build "$root\native-build" --config Release
$probe = "$root\native-build\Release\offline_temporal_probe.exe"
$runtime = (Resolve-Path build-project/Release).Path
$env:MOTION_ENHANCER_TEST_TEMPORAL_PROBE = $probe
$env:MOTION_ENHANCER_TEST_RUNTIME = $runtime
& tools/ai-train/.venv/Scripts/python.exe tools/ui-temporal/test_native.py -v

& tools/ai-train/.venv/Scripts/python.exe tools/ui-temporal/benchmark.py `
  --prepared "$root\integrated" --probe $probe --runtime $runtime `
  --runs 24 --out "$root\benchmark-integrated"
```

The standalone probe binds GPU buffers for all inputs/outputs, disables CPU
fallback, verifies DirectML profile events separately from timing, checks repeat
and A→B→A restore behavior, and compares baseline/tap/head in opposite orders.
It does **not** feed `next_state` back through a live capture queue. Fixed valid
history is varied to test consumption; the offline `History` class is tested
separately. Raw feature-tap output is diagnostic-only and materializes all 40
channels; the temporal graph exports only the compressed eight-channel state.

Wall timing covers model Run through queue completion. GPU timestamp intervals
also include submission gaps, not just kernel time. Neither covers input
preparation/upload, live-state ownership/copy, output validation readback, icon
inference, native masking, DIS, synthesis or presentation. No real game is run.
Random zero-initialized flow heads cannot establish trained-checkpoint fusion,
flow parity or representative trained timing. Any isolated numbers are not a
4 ms combined-AI gaming-load guarantee or evidence of HUD detection accuracy.

See the [plan](../../docs/plans/temporal-ui-head-2026-11-01.md) for the separate
data, training, safety, gaming-load and integration approval gates.
Recorded [Stage 1 results](../../docs/validation/temporal-ui-prototype-2026-11-01.md)
include the retained >4 ms baseline spike and first-run costs; no latency or
accuracy guarantee was established.
