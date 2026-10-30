# AI refinement restored to the main build

## Cause and correction

The character-branding revision on `main` did not include the earlier refinement
implementation. That implementation was recovered from saved revision `8dd0f79`
(`Ai refinment`). Its runtime, trainer/export support and regressions were
restored without importing its old personal settings, models or build outputs.
The GUI matches a clean three-way merge of refinement and character branding.

There is still only one active workspace build:
`build-project/Release/motion_enhancer.exe`. The 24 FPS character, splash and
orbital icon are retained. No installation, training job, commit or push occurred.

## Active model and control

The user first approved activating the completed refined Medium checkpoint, then
supplied the four corresponding files from Documents. Those supplied files are
now copied byte-for-byte into `build-project/Release/models/`:

- `ai_flow.onnx` / `ai_flow.json`: integrated `pyramid-refine-v1`.
- `ai_flow.dis.onnx` / `ai_flow.dis.json`: head-free `pyramid-v2` backbone.
- Both metadata files identify bundle `dbd389e4f8d7485da0dbfde1714205a0`.
- Medium, 832x480, saved zero-based epoch 5 (six epochs), saved validation PSNR
  29.74400069978502 dB. This is not an AI-versus-DIS playback comparison.
- Integrated/bypass graphs have 383/369 nodes; bypass has no refiner weights.

The supplied ONNX graphs were byte-identical to the independently exported
completed checkpoint used for the numerical tests below. All four Documents
originals remain unchanged.

On **Live Capture**, select **AI pyramid (trained model)**. The **AI flow refinement
(off: DIS)** checkbox is enabled for this pair:

- Checked: integrated AI correction, zero DIS patch passes.
- Unchecked: head-free AI backbone followed by DIS refinement.
- Blending-mask selection is independent. Changes require stop/restart capture.
- Standalone DIS remains the initial engine, so the checkbox initially appears
  disabled until the AI engine is selected.

## Verification

- All 70 restored CPU/UI/reporting regressions passed.
- Native routing passed 40 ring-reuse pairs, including mask toggles, finite
  fields and both exclusive paths, both before and after activating the supplied
  bundle in the canonical build.
- Real Win32 checkbox tests passed engine gating, checked/unchecked selection,
  invalid-pair disablement and refresh recovery. The test now waits for the main
  window to be shown rather than inspecting a partially created control.
- A separate read-only check against the active Documents bundle verified the
  actual main EXE's initial DIS engine, AI selection, enabled checkbox and toggles.
- Native character/splash checks passed all three pages, 24 FPS cadence and loop
  timing. Explorer/window icon checks passed as well.

DirectML versus PyTorch on the actual completed model:

| Graph | Mean flow error (px) | Flow p99 (px) | Mean mask error |
|---|---:|---:|---:|
| Backbone | 0.005922 | 0.032959 | 0.000685 |
| Integrated | 0.004486 | 0.031250 | 0.000685 |

All passed the existing 0.1 px mean, 0.5 px p99 and 0.005 mean-mask tolerances.
No new quality-superiority or 4 ms gaming-load guarantee is claimed.

## Preservation

`backup/2026-10-30_single-build-archive-mzb1slcg/refinement-restoration/` holds the
previous main runtime/models, overwritten source copies, saved refinement source
archive and deployment manifest. The intermediate fresh export is also backed
up before copying the Documents pair. Earlier build archives remain intact.

All 12 other protected hashes matched, including checkpoints, dataset index,
installed EXE and character atlas. Trainer settings changed during the work;
they were preserved rather than reverted, and their export destination still
points to the canonical build. Checkpoints and source models were not modified.

Active executable SHA256:
`9d84317170ea6cd6ded364a4a5eb4c87cd4ee79337a8ff1a07e8a62ef0c85612`.
