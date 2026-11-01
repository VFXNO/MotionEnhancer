# Icon-only UI detection — 2026-11-01

> Subsequent change: after the user reported live icon dropouts, current-proposal
> continuity was added. This report records the removal-only build; see
> [continuity validation](ai-ui-continuity-2026-11-01.md) for current behavior.

## Change requested

The user reported random Text/HUD detections and asked to remove that option,
retaining the icon detector that works for them. No new detector, training,
calibration, flow change or speculative temporal filter was introduced.

The canonical `build-project/Release/motion_enhancer.exe` now offers only:

- **DirectML: icons (default)**.
- **Matrix: icons (WMMA)**.

These are execution backends for the existing icon detector, not two different
detector types. UI detection still defaults off and requires **Keep static UI
sharp**. Refinement, blending, depth and motion-engine controls are independent.

Removed the PP-OCR/text-map decoder, model metadata dispatch, GUI/CLI text
choice, text model preparation/comparison tools and associated text-only tests.
`--ui-detector text` now fails explicitly rather than silently selecting icons.
Former 640/320 text-map graphs also fail the restored fixed float32 icon I/O
contract `[1,3,640,640] -> [1,5,8400]`, including through an ONNX path override.

The three runtime `ui_text.*` files were moved intact to:
`backup/2026-11-01_ui-icons-only-before-vgkfer40/removed-runtime-text/`.
That archive also preserves the before-change executable, source, runtime
models, licences/provenance and shaders. Historical experiment reports remain
available, with the earlier text report marked as superseded.

## Consistency scope

The working icon decoder (`include/UiRegions.h`), icon weights, matrix graph and
matrix backend source are unchanged. Confidence >=0.10, NMS 0.5, two-pixel edge
padding, 32-region cap and bounded-geometry rejection remain the same.

Fresh detections are already retained between at-most-5-Hz updates; pair-owned
debug boxes match the actual native mask pair. Missing/empty/failed/expired
detections authorize no masking. The 600 ms expiry is not extended; no stale
box hold, guessed HUD region or filled rectangular protection mask is added.
Native per-pixel evidence/confirmation/release remains unchanged.

Same-input repetition checks are deliberately limited: they can prove repeated
execution is stable on the supplied image, **not** that icons cannot flicker,
be missed or be misclassified as the game scene changes. No broad game-HUD
accuracy or strict 4 ms gaming-load deadline is claimed.

## Artifacts and preservation

Diagnostics:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-icons-only-20261101-urhbgym1\`.

`preservation-final.json` verifies all **20 protected original hashes**, all
**31 existing shader binaries**, and the optional matrix model quartet remain
unchanged. It also verifies unchanged icon-decoder/matrix-backend source and
absence of the removed text files from the active runtime. Flow models,
training data/checkpoints/settings, branding and `C:\MotionEnhancer` were not
changed. Build configuration still has flow-model staging OFF and matrix ON.

Release executable SHA256:
`ca377441426a7e78686c28750fb1ba3ebf54fb758e4e1504c99eba6d249c216f`.

No commit, push or installation was performed.

## Validation results

- Canonical Release build passed with no text decoder/source dependency.
- **28/28** safety/native unit tests passed, including removed-text CLI
  rejection, both former text-map shapes rejected despite old metadata,
  CPU fallback rejection and existing matrix ISA/changed-input replay checks.
- **96 same-image calls** (two fresh instances per backend, 24 calls each,
  reversed backend order) produced stable raw proposals within/across instances.
  DirectML checksum: `15210798726307519102`; matrix:
  `3162389816323055489`. Each backend's boxes exactly match its corresponding
  pre-removal output on the same GUI fixture. Different backends are not
  claimed to be numerically identical to each other.
- The actual archived PP-OCR graph was also rejected through the generic ONNX
  override; no output file was produced and no icon fallback occurred.
- Existing eight native/integration checks, independent AI/DIS refinement and
  blend-mask behavior, native UI mask gating, debug/status regressions and
  repeatable normal rendering passed.
- Owned GUI checks passed exactly two icon-only choices, correct default,
  checkbox dependencies, AI/DIS independence and non-overlapping layout.
- Source whitespace checks passed. No runtime `ui_text.*` files remain.

Logs, `consistency-final/summary.json`, GUI preview and preservation manifests
are retained in the diagnostic directory above. Same-input repetition is not a
live temporal test; changing-game-scene stability remains unmeasured.
