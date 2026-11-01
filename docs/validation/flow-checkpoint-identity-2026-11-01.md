# Flow checkpoint: read-only export identity check

**Verified:** the current `tools/ai-train/runs/best.pt`, converted to FP16 as the
existing exporter does, matches **every learned weight/bias bit** in both active
flow ONNX files. This resolves the stored-parameter identity question, not the
trained UI-wrapper parity, accuracy or gaming-load latency gates.

## Pinned files

| Artifact | SHA256 |
|---|---|
| `tools/ai-train/runs/best.pt` | `620c208022fd421b3fe08097428747d5017e48a8877e1bfd2d6492103af4f091` |
| `build-project/Release/models/ai_flow.onnx` | `f21c9d9317869b0996ec26db1a8c7abff2d117fab8a250c4970e027406194473` |
| `build-project/Release/models/ai_flow.dis.onnx` | `6339da402e05f29c975497133cfc269b1b673591c77ce9479b08856f16116595` |

Checkpoint: Medium `pyramid-refine-v1`, epoch 5, training height 480, stored PSNR
29.74400069978502. Both export metadata files agree on bundle
`dbd389e4f8d7485da0dbfde1714205a0`, 832×480, seven levels and validation selection
`39bc3f42ff7d2f03ab5fee4717fa6996c49c313d0b59636e6d17bcc457b42291`.
These are stored metadata, not newly measured quality scores.

## Method and results

1. Fingerprinted the exact metadata-referenced checkpoint and active export,
   metadata, model/export source, canonical executable and CMake-cache files.
2. Loaded checkpoint **data only** with explicit `torch.load(...,
   map_location='cpu', weights_only=True)`. No unsafe loading fallback, model
   instantiation, forward pass, ONNX Runtime session or GPU work.
3. Checked all checkpoint parameters are finite FP32 tensors. Parsed the existing
   embedded ONNX initializers read-only; external tensor files were refused.
4. Compared each named checkpoint tensor after FP16 conversion against its
   `net.<key>` initializer, including exact dtype, shape and byte identity. No
   tolerance, matching by approximate statistics or re-export was used.
5. Checked parameter-name sets, Conv weight/bias consumers, graph validity, static
   FP32 I/O, variant metadata and shared bundle identity. A small comparator
   control rejected changed bits, dtype, shape and missing values.
6. Rechecked all fingerprinted input hashes, sizes and modification timestamps:
   unchanged.

| Export | Expected/matched tensors | Matched parameters | Result |
|---|---:|---:|---|
| Integrated `ai_flow.onnx` | **64 / 64** | 716,688 | Exact FP16 bits |
| Paired `ai_flow.dis.onnx` | **58 / 58** | 715,932 | Exact FP16 bits |

All 58 retained tensors also match bit-for-bit between the two exports. The only
six tensors absent from the paired graph are `fine.3.refiner.*`; no learned
backbone or blend tensor is missing or extra. Both graphs retain the current
`frames -> flow, mask` contract; `mask` remains blend preference, not UI evidence.
DIS itself was not executed.

## Limits and next gate

- **The source path is still mutable.** No immutable copy was created, checkpoint
  overwritten, run resumed or checkpoint converted. Every later experiment must
  verify the pinned checkpoint/export hashes before use. An independently
  approved new immutable experiment copy can be made later, never in an existing
  run or active deployment folder.
- FP16 conversion is lossy. Exact exported-bit agreement plus matching metadata
  establishes compatibility of this checkpoint's weights with the active
  exports; it does not uniquely prove which historical FP32 file produced them.
- This comparison does not execute the current source model or experimental
  feature tap/head. **Trained-weight behavioral parity and timing remain untested.**
  No inference, re-export, training, dataset/capture processing, teacher work,
  downloads, environment changes, application build or model activation occurred.
- The [training plan](../plans/temporal-ui-head-2026-11-01.md) still requires
  separate approval for data/labels, actual training and its budget, trainer GUI
  support, loaded-game evaluation and application integration. This verification
  does not authorize those stages or establish the 4 ms maximum target.

Evidence directory:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-checkpoint-verify-20261101-i9o1blcm\`.
`verification.json` records every tensor's source/export hashes and all checks;
`verify.py` is the read-only diagnostic; `verification.log`, `inventory.json`
and before/after manifests retain the supporting results. No commit or push.
