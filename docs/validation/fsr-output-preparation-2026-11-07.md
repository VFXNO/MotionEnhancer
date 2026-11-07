# FSR output preparation (2026-11-07)

## Change

The previous loop slept until the output deadline, then generated FSR and
recorded/submitted the presenter. At 2x it generated the midpoint before
the real frame's Present on the same queue, unnecessarily delaying that
real frame behind flow and FSR. CPU clock tests alone could not expose this.

- Split `Presenter::prepare` (GPU recording/submission and capacity waits)
  from `Presenter::present` (the timed DXGI call).
- Present each real frame before queuing its pair's FSR generation. The
  midpoint is then prepared during the next output slot, ahead of its
  deadline. Merely submitting the real copy first is insufficient: its
  **Present** must precede FSR on the ordered queue too.
- Real outputs bind only their own source texture, with no future-frame or
  flow dependency, including in replay copying.
- Publish a later present-queue fence after FSR submission and mark both
  capture slots and the flow entry as readers through it. The earlier real
  frame fence cannot protect later FSR reads if a midpoint is skipped, the
  overlay is hidden, or input pauses.
- Live FSR skips a busy allocator instead of blocking the output thread;
  native interpolation supplies that output. Offline FSR retains its
  existing blocking behavior. Failed/cut pairs retain native fallback.
- Skip FSR entirely for odd fixed multipliers (3x has no t=0.5 output).
- Feed regularized source time deltas to FSR, not jittery capture arrivals.
- Preserve exact source FPS x N deadlines/positions and fixed buffering.
  No refresh resampling, adaptive latency, PLL, extra queue or async SDK
  flag was introduced. Capture duplicate detection is unchanged.

## Diagnostics

The status line now separates:

- `capacity ... avg/max`: CPU swap-chain capacity wait during preparation.
- `alloc ... max`: CPU presenter allocator wait.
- `Present call ... avg/max`: time inside DXGI Present, not scanout.
- `gpu pending`: prepared GPU lists incomplete when Present was called.
- `late ... max`: CPU Present-call start lateness vs the output deadline.
- `fsr ... busy`: nonblocking allocator fallback count.
- `fsr ... gpu ... avg/max (N samples)`: conversions + FSR + optional camera,
  measured only from fence-complete timestamp queries, excluding queue waits.
  Zero samples does not establish zero GPU cost.
- `submit cpu ... avg/max`: CPU time for calls that submitted FSR work.

FSR timestamp readback does not wait on incomplete queries. Flow starvation
is counted at the output deadline, not at early preparation (when unfinished
flow may still be on time). `MOTION_ENHANCER_PACING=1` retains detailed
`[Prepare]` logs, including remaining lead in milliseconds.

## Verification and limits

`--timeline-self-test` passes in the rebuilt Release executable:

- Existing 63 monitor-independent source/multiplier/rate combinations,
  streaming auto-cadence, Max, stalls, ring retirement and pause/resume.
- Shared `FrameOutputPlan` policy: real-before-FSR ordering, exact midpoint,
  3x bypass, Max windows, invalid/cut pairs and source-time delta selection.
- Ten CPU queue models: 60->120 on 60/120/144/165/240 Hz, with and without
  synthetic allocator-busy fallback. Each models 600 output slots and
  varying 2-5 ms generation cost. Normal runs generate 300 middles; busy
  runs generate 266 with 34 native fallbacks. No missed deadlines in these
  models. This is NOT a GPU benchmark or a live scanout result.

No live capture, GPU workload or FSR DLL execution was run for verification.
Actual gameplay improvement, driver behavior, GPU timing instrumentation
and image quality remain owner-run checks. A saturated GPU, blocking DWM
delivery or fixed 144 Hz scanout can still stutter at 120 FPS. This fix
does not promise even 120 FPS scanout on incompatible fixed-Hz displays.

The original `build-project/Release/motion_enhancer.exe` was locked by running
instances; they were not terminated. The tested replacement is staged beside
its existing DLLs/models as `build-project/Release/motion_enhancer-fsr-fixed.exe`.
Its GUI launches its own executable path, so capture uses the fixed build too.
