# AI UI debug output — 2026-10-31

Added **view 7** to the existing **Ctrl+Alt+F3** cycle, after AI blend mask.
One further press returns to normal output. The Live Capture page includes
a hotkey hint; **Ctrl+Alt+F2** saves the displayed diagnostic through the
existing replay buffer. `MOTION_ENHANCER_DEBUG_VIEW=7` also works offline.

## Display

- Cyan outlines: the detector allow-list used for this flow pair's mask.
- Yellow: native evidence that is not providing supported protection yet.
- Green: actual mask coverage after the existing 3x3 support requirement.
  An isolated confirmed pixel is yellow, not incorrectly labelled protected.
- Dim source-frame backdrop, not an interpolated-output quality comparison.
- Top-left status: MASK OFF, AI OFF (native-only masking), NO PAIR, NO RESULT
  (waiting/expired/failed), or REGIONS: 00–32 (valid result, including empty).

The allow-list/status is stored by value with the flow-ring entry and copied
with its mask into the render job. A newer worker detection cannot change the
boxes shown for an older mask. Missing/invalid results suppress stale bounds.
Pass-through frames show no boxes or pair mask. Hotkey changes participate in
the presentation skip key, so unchanged/held frames can redraw the new view.

`UiDebugViewCS` is a separate debug-only kernel. It does not alter masks or
run inference. Normal synthesis/projection and all old debug modes retain
their shader code. No UI-model replacement, changed threshold or enabled-by-
default setting; the previously documented game-HUD detector limitation remains.

## Verification

- Canonical Release build: PASS; existing SDK/DXC/MSVC warnings remain.
- `tools/ui-detect/test_debug_view.py`: PASS, including real GPU pixel checks,
  pair-snapshot value ownership, cyan outlines, supported green patch, yellow
  evidence/isolated confirmed pixel, unfilled box interior, count cap, stale
  box suppression, five distinct rendered status headers, no-pair output,
  offline view 7 and identical normal PNG hashes before/after the debug run.
- Existing native D3D12, projection, AI blend-mask, AI refinement and UI-region
  self-tests: PASS. Both Medium flow routes and ring-reuse tests remain intact.
- Missing-model fail-closed, parent-mask-off and real detector/native DIS
  integration tests: PASS.
- Owned Win32 GUI: default-off checkbox, parent dependency, AI/DIS independence,
  unchanged refinement state, toggles/layout: PASS. Screenshot reviewed.
- D3D12 debug-layer run of the complete native UI-region/debug self-test:
  PASS, with no validation messages returned.
- `git diff --check`: PASS.

All **20** pre-recorded protected hashes match (runtime/root model files,
checkpoints, settings, dataset index and protected installed files). All **30**
previous compiled shader binaries are byte-identical to the before-debug
archive, including native masking, interpolation, projection and flow shaders.
The new debug shader is additional. No production training or live game-load
benchmark was run; these checks do not establish latency or HUD accuracy.

Canonical EXE: `build-project/Release/motion_enhancer.exe`.
SHA256: `536e6887b270b4f6123a6d864e6542b44b410b405dc27a61d0b8f6cee802c059`.
Previous runtime/shaders: `backup/2026-10-31_ui-debug-before-af086_s3/`.
Diagnostic images/logs: `C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-debug-20261031\`.
No commit, push or installed-programme update.
