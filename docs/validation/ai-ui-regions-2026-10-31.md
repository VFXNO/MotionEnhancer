# Experimental AI-region-gated native UI masking — 2026-10-31

## Outcome

Implemented the requested split: **AI detects candidate UI areas; native
`UiMaskCS` protects qualifying pixels only inside those areas**. A detector
box is never a filled protection mask. Outside regions, old evidence is
cleared; inside, all previous static/damage/change/confirmation tests remain.

The canonical `build-project/Release/motion_enhancer.exe` now has **AI: mask
only detected UI areas (experimental)**. It defaults **off**, depends on
**Keep static UI sharp**, and is independent of motion engine, AI refinement,
AI blending mask and depth. Restart capture to change it. CLI:
`--ai-ui-regions`; `--no-ui-mask` disables detection as well as native masking.

**Quality limitation:** the downloaded OmniParser V1 detector produced **zero
regions in both tested 2560x1440 game frames**, despite correctly locating
many controls in an owned Motion Enhancer GUI screenshot. Thus the pipeline
works, but this model is **not validated as a usable HUD detector for that
game**. Do not enable this mode expecting those HUD elements to be protected.

## Model/runtime

- New, separate `build-project/Release/models/ui_detect.onnx`, 6,156,263 bytes.
- Pinned ONNX Community revision `799bd041b5d053ed44651c2237ced04d8fdb2777`.
- SHA256 `cbc393231545eee6155291c7e2527db3eca86ed54935f4155556113344e24d00`.
- Detector-specific **AGPL-3.0** licence and provenance saved alongside it as
  `ui_detect.LICENSE.txt` and `ui_detect.source.json`. Not relabelled MIT or
  automatically added to install/package rules.
- GPU-centred letterbox RGB NCHW input, separate DirectML queue/worker, at
  most 5 requests/second, one in flight. Small raw box output returns to CPU
  for decoding/NMS. No inference/readback wait on capture or presentation.
- Confidence 0.10; IoU-NMS 0.5; at most 32 boxes; two detector-pixel padding;
  invalid/degenerate/padding-only/whole-scene boxes rejected.
- Empty until ready, empty on failure, expires 600ms after its input was
  recorded. No full-screen fallback. Frame-ID gaps invalidate cached boxes;
  resize recreates the worker. Missing/incompatible explicitly requested
  models fail session setup rather than silently changing mask behaviour.

See `tools/ui-detect/README.md` for setup and reproducible diagnostics.

## Verification

- Release CMake build: PASS (existing SDK/DXC/MSVC warnings remain).
- All **70** existing trainer/export/quality/refinement/latency regressions:
  PASS, 96.556s. An initial 120s attempt timed out; the completed rerun with
  a larger timeout is the result counted here. No production training run.
- **7** downloader tests: licence acknowledgement, verified content,
  provenance preservation, existing-file protection, bad hash/licence rejection.
- Actual native GPU tests: legacy native masking unchanged; only in-region
  evidence survives; empty/lost regions clear prior evidence; boxes alone
  create no mask; changing UI still releases. Decoder tests include landscape
  and portrait letterbox inversion, NMS, cap, invalid values and expiry.
- Native D3D12, projection, AI blend-mask and AI refinement self-tests: PASS.
  Both real Medium flow graphs still route correctly: integrated has zero DIS
  patch passes; unchecked runs backbone + DIS. Mask/ring-reuse tests pass.
- Missing detector fails closed; `--no-ui-mask` skips even a missing detector;
  actual pretrained detector + native DIS flow integration: PASS.
- Owned Win32 GUI: default-off, parent dependency, independent AI/DIS engine
  selection, refinement unaffected, checkbox toggling and non-overlap: PASS.
  Full screenshot confirms retained orbital/character branding and controls.
- Actual GPU preprocessing/DirectML worker/readback/buffer reuse/invalidation
  exercised for 24 runs per screenshot (three warmups, 21 recorded samples).
- Optional D3D12 debug-layer execution completed. It logged DirectML
  unsupported-meta-command probing messages; the returned messages contained
  no resource-state/queue-ownership diagnostics. Do not call this a completely
  clean debug-layer log.

### Limited measurements, not a latency guarantee

Initial two game-frame worker wall-time samples (21 each, after warmup):

| Frame | Regions | Mean ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| `game_000.png` | 0 | 3.216 | 8.142 | 9.790 |
| `game_007.png` | 0 | 3.622 | 5.678 | 5.719 |

The full GUI positive control returned 32 regions (the cap), mean worker
1.856ms, p95 2.650ms, max 2.927ms. This is a different input/run, not a claim
that all UI detection is below 4ms. Worker timing excludes session creation
and flow-queue input preprocessing; it is not total interpolation, capture,
presentation, or controlled gaming-load timing. The debug-layer run is not
included in the normal timing comparison. No live pacing guarantee or game
HUD-quality improvement is established.

## Preservation and build-hook correction

The rebuild exposed an existing `cmake/Install.cmake` post-build hook that
restaged the old root packaging model over the activated runtime primary.
Its two overwritten files were restored **only after verifying the original
Git blobs exactly matched the pre-build protected SHA256 manifest**. The
automatically staged copies were retained in the preservation archive.
The activated quartet matches its original pre-build hashes at completion.

Ordinary rebuild model staging is now explicitly opt-in through
`MOTION_ENHANCER_STAGE_AI_FLOW_MODEL` (default **OFF**). The subsequent actual
rebuild confirmed no flow-model copy commands and unchanged protected hashes.
No installer/package was run and `C:\MotionEnhancer` was not updated.

Archive: `backup/2026-10-31_ui-regions-before-2rt43nz1/`, containing the
previous executable/shaders/build notes, protected-file manifest and copies
of the old automatically staged primary model. All **12** pre-recorded
protected files match: runtime flow quartet/MiDaS, completed checkpoints,
trainer settings, original dataset index, installed EXE and installed Large
checkpoints. Production trainer/model/loss/target source is unchanged.

Separate changes/copies also appeared under the root `models/` folder during
the session; their origin was not established. They were investigated but
not deleted/reverted as potentially independent user work. They are not the
canonical runtime's model selection and were not used to replace its quartet.

Final canonical EXE SHA256:
`74b87b3b6d8db70c06d709de04f30cb883888e0e40c53b929864e216fabbbbcb`.

Local diagnostic images, JSON and raw logs:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-regions-20261031\`.
No commit, push, smoothing, flow-model activation or installed-programme update.
