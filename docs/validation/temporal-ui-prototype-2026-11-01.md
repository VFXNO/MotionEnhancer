# Temporal UI prototype: Stage 1 synthetic results

**Outcome:** independent research prototype supports frozen-backbone and
from-scratch model/objective APIs. Synthetic correctness and DirectML plumbing
passed. **No trained UI detector, trainer GUI option, runtime integration, HUD
accuracy result or 4 ms gaming-load guarantee was produced.**

Sources: [prototype and commands](../../tools/ui-temporal/README.md),
[staged plan](../plans/temporal-ui-head-2026-11-01.md).

## Scope and implementation

- New `tools/ui-temporal/` only: finest pair-feature tap, shared eight-channel
  projection, 16-channel temporal head, offline history reference, objective
  adapters, weights-only research snapshots, exports, standalone GPU probe/tests.
- `TemporalUi.from_scratch` creates flow and UI weights without a checkpoint.
  Separate optimizer groups are exposed but no optimizer was constructed/run.
  The original flow interpolation loss/target contract is unchanged. UI labels
  supervise detached features and cannot change flow/refinement/blend gradients.
- Frozen mode copies/freezes the supplied backbone and exposes only UI parameters
  to the optimizer. Neither mode supplies labels, creates a dataset or launches
  a resumable training job. That runner and GUI integration are not implemented.
- The Medium integrated backbone has 716,688 parameters; projection/head add
  1,481. This is not evidence that the head can recognize any script or HUD.
- The app never receives a candidate map. Any future adoption must use additional
  candidates only as admission to native pixel evidence, retaining icons and
  native masking/synthesis behavior. No flow attenuation or solid-box masking.

## Correctness checks

**25 CPU synthetic tests and 2 opt-in native DirectML tests passed.** Checks cover:

- Exact FP32 tap parity for Small/Medium/Large, refiner on/off, even/odd sizes and
  synthetic nonzero flow/refiner heads; no duplicate finest-body execution.
- Exact original flow-loss values and backbone gradients in scratch mode while
  UI head gradients are nonzero; UI-only gradients never reach the backbone in
  either mode. Backward checks do not update weights and are not training steps.
- Invalid/nonfinite history and logits contribute zero, including per-sample
  validity; valid changed history is consumed; fixed inputs restore identically.
- Generation/size/variant changes, signaled cuts/gaps, replay/out-of-order state,
  missing predecessors, ownership copies and repeated replacement-pair handling.
  This is a CPU history reference, not a live GPU fence/scene-cut implementation.
- Exclusive/checksum-pinned research snapshot roundtrip, rejection of legacy or
  nonfinite weights, and refusal of existing/active export destinations.
- GPU-bound multiple inputs/outputs and an intentionally CPU-only ONNX negative
  control: fallback-disabled execution rejects the latter instead of using CPU.

## DirectML experiment

Device: **AMD Radeon RX 9070 XT**. Existing ONNX Runtime/DirectML DLLs were used
read-only. Medium 832×480 FP16-internal graphs, FP32 external I/O:

1. Original one-input/two-output graph.
2. Feature-tap graph exporting the full 40-channel half-resolution activation.
3. Temporal graph with prior state/validity inputs and candidate/next-state outputs.

Both integrated-refiner and paired-backbone variants were exported. The latter
was derived from the exact generated integrated snapshot: all retained flow/UI
tensors matched, and only six refiner tensors were removed. **DIS itself was not
run.** No existing trained checkpoint or canonical model weights were loaded.

Separate profiling passes reported only `DmlExecutionProvider` events for all
six graphs, with CPU fallback disabled. This establishes placement for these
graphs, not matrix-ISA use. Timing used two opposite graph orders per variant,
four synthetic cases, three warmups and 24 measured repetitions per case: **192
warm samples per graph**. Profiles were not included in the timing table.

Within each variant, baseline versus tap/temporal flow and blend outputs were
exact on DirectML for every case. Repeat/first-input-restore drift was zero.
Startup invalid history authorized zero pixels; changing valid history changed
the temporal map. Eager FP16 graph outputs also had exact flow/blend parity.

**Important limitation:** these exports contain random untrained weights and
zero-initialized flow output heads. Nonzero-head FP32 tests exercise the adapter,
but DirectML parity/timing on these fixtures does not prove parity, fusion or
performance for the active trained checkpoint. Active-backbone identity remains
unverified **by this synthetic experiment**; the mutable path in export metadata
is insufficient. A subsequent separately selected [read-only parameter check](flow-checkpoint-identity-2026-11-01.md)
verified matching stored weights at pinned hashes, without trained-wrapper execution.

### Isolated timing, milliseconds

GPU intervals include submission gaps. Wall timing covers Run through queue
completion. Neither includes setup, input preparation/upload, output validation
readback, live state ownership/copy, icons, native masking, DIS, synthesis or
presentation. There was no game/capture workload. Do not compare these to the
combined-AI target as if they were full-pipeline measurements.

| Variant / graph | GPU p50 | GPU p99 | GPU max | Wall max | Warm GPU / wall >4 ms |
|---|---:|---:|---:|---:|---:|
| Integrated / baseline | 2.065 | 2.780 | 3.014 | 3.127 | 0 / 0 |
| Integrated / raw tap | 2.329 | 2.821 | 2.853 | 3.113 | 0 / 0 |
| Integrated / temporal | 2.456 | 3.201 | 3.323 | 3.416 | 0 / 0 |
| Paired backbone / baseline | 1.943 | 2.542 | **4.534** | **5.282** | **1 / 1** |
| Paired backbone / raw tap | 2.124 | 2.471 | 2.564 | 2.647 | 0 / 0 |
| Paired backbone / temporal | 2.250 | 2.735 | 2.819 | 2.916 | 0 / 0 |

The >4 ms baseline sample is retained, not discarded as an outlier. First-case
Run maxima, reported separately from warm samples, were approximately **25–32 ms**
across these graphs; session creation/setup is additional and was not timed.
The strict maximum target is therefore not established even in this isolated
experiment, much less under gaming contention with the icon workload included.

The diagnostic raw FP32 tap output is 15,974,400 bytes. The temporal graph instead
exports a 3,194,880-byte FP32 state and 399,360-byte candidate map, and consumes a
state input of the same size. These are tensor sizes, **not measured peak memory
or a live state-copy/fence budget**. No GPU-memory residency claim is made.

## Preserved state and remaining gates

- No dataset/capture processing, teacher use, downloads/package changes, optimizer
  steps, trained-checkpoint conversion, application build or runtime activation.
- Protected-file SHA256 verification compares the 125-file pre-experiment
  manifest, including existing trainer/model source, application source/shaders,
  runtime model files/executable, CMake cache and original `Lossless.dll`.
  Existing unrelated working-tree changes are retained. No commit or push.
- `build-project` remains the sole active application build/trainer deployment
  destination; model staging remains OFF. Temporary diagnostic builds/exports
  are research artifacts, not an alternative deployment location.
- Before advancing, use an approved immutable experiment copy of the now
  checksum-pinned compatible backbone and resolve trained-wrapper behavioral
  parity. Data/labels, actual head training, any full-flow scratch budget,
  loaded-game evaluation, GUI support and application integration still require
  separate approval. The Arabic capture remains locked and unused in this stage.

Local evidence root:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-temporal-stage1-20261101-dg_2ccvl\`.
It contains test logs, export manifests/graphs, raw outputs, placement traces,
per-sample timing reports, paired-weight checks and preservation manifests.
The first export attempt failed on Windows console encoding; its partial folder
was retained. A fresh UTF-8 export succeeded; no environment/package change was
needed. Source fixes also resolved the native header's `Gpu` name collision.
