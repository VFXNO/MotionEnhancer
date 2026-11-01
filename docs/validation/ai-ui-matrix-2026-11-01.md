# Explicit UI matrix backend — 2026-11-01

## Outcome

**An explicit, ISA-verified matrix backend now runs the UI detector. It remains
opt-in, not the default, because it is slower and misses the 4 ms limit.**
The canonical build contains the backend; the candidate model remains separate.
DirectML FP16 stays selected for the existing, default-off AI UI-region option.
No flow model, native mask, synthesis, training target or refinement routing was
changed. No training, calibration, dataset processing, installation update,
commit or push occurred during this matrix investigation.

## What is verified

- RX 9070 XT, observed AMD driver `32.0.32015.7029`. The Windows Vulkan driver
  exposes `VK_KHR_cooperative_matrix` and `VK_AMD_shader_info`. Vulkan adapter
  LUID must match the application's D3D12 adapter; no other adapter is chosen.
- New `UiMatrixDetector` uses pinned ncnn 20261020 with cooperative matrices,
  FP16 weights/storage and FP32 matrix accumulation. INT8 is not required.
- Runtime hooks inspect actual shader modules for SPIR-V opcode 4459,
  `OpCooperativeMatrixMulAddKHR`, then inspect their compiled AMD pipelines
  through `vkGetShaderInfoAMD`. For example:

  ```text
  v_wmma_f32_16x16x16_f16 v[9:16], v[17:20], v[19:22], v[5:12]
  ```

- The final graph contains **60 matrix dispatches per inference**. All dispatched
  cooperative-matrix pipelines have verified `v_wmma_` ISA. A diagnostic run
  retained 26 pipeline disassemblies containing 830 static WMMA instruction
  occurrences; that count is not a dynamic instruction count or utilization %.
- The first call records the verified command graph without the one-time-submit
  flag. Every replay resubmits that same retained command buffer with freshly
  copied/flushed input and waits for its completion fence. Input/output barriers,
  retained descriptors/intermediates and output invalidation cover reuse.
- Model layers that need CPU execution or feature masking are rejected. A
  compatible-size GPU-only graph without matrix instructions is also rejected.
  No missing-ISA, failed-model or failed-inference fallback to DirectML/CPU or
  full-screen native masking. Ordinary GPU shaders still handle non-matrix ops;
  native CPU proposal decoding/NMS is not neural-network fallback.

This establishes explicit matrix instructions in dispatched, completed detector
work. It is not a hardware utilization/counter measurement, proof that DirectML
does not already use matrix units, or proof of a speedup.

## Integration and timing scope

- Existing asynchronous worker, at most 5 Hz, one job in flight, 600 ms expiry.
  Capture does not wait for inference or readback. All flow controls stay independent.
- GPU letterboxing/input layout and output contract remain
  float32 `[1,3,640,640] -> [1,5,8400]` at the native boundaries.
- The worker bridges the 4.9 MB input through D3D12 readback into a mapped Vulkan
  buffer; GPU shaders convert it to FP16. **Not zero-copy.** Native layers never
  execute the network on CPU. The cached graph retains intermediate GPU buffers,
  adding VRAM use; memory/throughput under gaming load has not been characterized.
- Boxes remain ROI gates only. Native per-pixel evidence creates the actual
  protection mask. Debug view 7 and pair ownership are unchanged.

## Counterbalanced comparison

Same three saved, previously held-out screenshots; two rounds in reverse backend
order; 127 calls/group with the first three excluded. **248 warm samples per
image/backend, 1,488 total** in the final comparison. Raw-proposal/ISA diagnostics
were separate from timed runs. No deliberately controlled gaming workload.

| Input | DirectML mean ms | Matrix mean ms | Matrix p95 ms | Matrix max ms | Matrix >4 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Owned GUI | 1.591 | 3.632 | 4.264 | 5.103 | 45/248 |
| game_001 | 1.540 | 3.741 | 4.562 | 5.779 | 59/248 |
| game_007 | 1.576 | 3.756 | 4.545 | 4.948 | 67/248 |

Matrix exceeded 4 ms in **171/744** warm samples. DirectML had 2/744 over 4 ms
(observed maximum 9.340 ms); neither is a strict deadline guarantee. First-worker
matrix times were **21.315–27.370 ms**, versus DirectML **2.908–3.651 ms**.
These cold calls are retained as `all_worker_ms_including_cold`, not hidden as
warm performance. Session/model setup is outside both measurements.

Worker times include queue waits, the matrix input bridge/upload, GPU inference,
proposal readback and CPU decoding. They exclude capture, flow-queue input
preprocessing, session creation and presentation. They are not total frame time.

The first noncached implementation measured 7.90–8.36 ms means over another
1,488 counterbalanced warm samples. Caching reduced overhead but did not make
the path competitive. An initial cache prototype incorrectly recreated the
output in an unmapped GPU allocator; it failed in diagnostics, was fixed to
use the mapped staging allocator, and is not the final tested implementation.
**No slower candidate was activated over the existing detector.**

### Numerical agreement, not accuracy

GUI: both return the 32-region cap; **31/32** boxes match at IoU >=0.5,
matched median IoU 1.0. Mean absolute score delta `0.00005546`, maximum
`0.0057373`; relevant proposal-coordinate median delta 0 and maximum 1.0
detector pixel. Both backends return **zero regions in both game frames**.
Game score max deltas are `0.0003967` / `0.0002441`. This is numerical agreement,
not evidence of general-game HUD detection accuracy. The existing domain mismatch
remains unresolved.

## Reproduction and opt-in use

Canonical executable: `build-project/Release/motion_enhancer.exe`.
Build with `MOTION_ENHANCER_ENABLE_UI_MATRIX=ON` after running
`tools/ui-detect/setup_matrix.py`. Keep flow-model staging OFF. The dependency
is new, hash-pinned source under `third_party/ui-matrix/`; no original SDK source
or installed driver was changed. ncnn/glslang notices are retained in runtime
`licenses/`; the detector conversion retains its AGPL licence/provenance.

Tools:
- `convert_matrix.py`: source and Windows x64 pnnx executable hashes pinned;
  creates a new graph folder, never replaces flow/runtime models.
- `benchmark_matrix.py`: native raw outputs, boxes, ISA dumps and reversed-order
  worker comparisons, without activation.
- `test_matrix.py`: archive/conversion safety, negative GPU/CPU fixtures and
  changing-input graph replay against separately initialized graphs.

Current candidate (not staged or auto-selected):
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-matrix-20261101-811mk7wu\converted-model\ui_detect.matrix.param`.

To test it explicitly in a new PowerShell process environment:

```powershell
$env:MOTION_ENHANCER_UI_BACKEND = 'matrix'
$env:MOTION_ENHANCER_UI_MATRIX_MODEL = 'C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-matrix-20261101-811mk7wu\converted-model\ui_detect.matrix.param'
.\build-project\Release\motion_enhancer.exe --gui
```

Enable **Keep static UI sharp**, then **AI: mask only detected UI areas**.
This chooses the experimental detector backend only, not the motion estimator.
Remove those process-local environment variables or set `UI_BACKEND=directml`
to return to the unchanged ONNX backend. No persistent environment is modified.

| Artifact | SHA256 |
| --- | --- |
| Pinned ncnn full-source ZIP | `699d9acc67489eee0993a551ca1ebc247bbf0f4e33babe1dec810e6a9a52b988` |
| pnnx 20261020 Windows x64 executable | `b8c56a0b0319e6ce4d229e9a339e4ee377fb0c42f7f3e66eec235ed00cd71b18` |
| Source FP32 ONNX | `0432face5e01e085eb3fc1b2b64ca706cc7916cd18a48baa5fdc55a4bfe88ef4` |
| Matrix param | `f34941eb6a82d67a84b1aa469ee76e1faea97cf4f17de0447f5d84fd6d8760e6` |
| Matrix weights | `c4245eb498c431aa18bb4e26e41926df3f3305eddc22829251cf790514536822` |

Evidence/logs/raw comparisons:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-matrix-20261101-811mk7wu\`,
especially `comparison-cached/report.json` and `raw-matrix-0-isa/` inside that
comparison. Pre-matrix runtime preserved at
`backup/2026-11-01_ui-matrix-before-ahxmwg8g/`.

## Final validation and preservation

- Release build with optional matrix backend enabled: PASS.
- **25** downloader, conversion, quantization-safety, numeric and native negative/
  replay tests: PASS. The CPU-only ONNX fixture still fails setup. A masked CPU
  ncnn layer and a valid-output-size GPU graph without WMMA both fail closed.
  Fresh A/B graphs and 12 alternating A/B calls produce identical per-input raw
  fingerprints, including restoration of A after B; 36 replay-fixture calls total.
- Eight existing native/integration checks: PASS (D3D12, projection, blend mask,
  refinement, UI regions, missing detector, parent off, real DirectML integration).
- Four matrix integration/negative checks: PASS, with D3D12 debug layer enabled:
  native DIS integration, missing graph, parent-mask-off override, invalid backend.
- Native/debug-view-7 pixel/status/expiry assertions, repeatable ordinary output,
  and owned Win32 GUI default/toggle/engine/refinement/layout checks: PASS.
- All **20** protected original file hashes match the earlier manifests,
  including FP16 UI, flow quartet, root models, trainer settings/checkpoints,
  dataset index and installed programme/checkpoints. All **31** pre-existing
  shader binaries match the pre-matrix archive. No existing shader source was
  modified by this matrix-backend work.
- Existing SDKs, character/orbital branding, native 24 FPS animation and the
  installed `C:\MotionEnhancer` remain unchanged. No extra application build,
  active-model replacement, training, teacher pass, commit or push.

Final canonical EXE SHA256:
`096df58ec618fab32d7362b9ce6499731122f597db2e5848c86fe169e554502e`.
Full preservation results: diagnostic `preservation-final.json`.
