# Source-rate output clock (2026-11-07)

At the owner's request, the earlier refresh-resampling fix was superseded.
Fixed 2x must mean 60 -> 120 FPS, not 60 -> 144 FPS on a 144 Hz monitor.

## Removed output pipeline

- Compositor-clock waits and synthetic refresh ticks.
- Presenter scanout prediction / PLL and wake-cadence correction.
- Refresh-count holds, k/M phase lock, trim/deadband/slide state.
- Resampling fixed multipliers to monitor Hz.
- Gradual media-clock lead adjustment and starvation-driven budget growth/review.
- Independent-flip/composed-mode clock switching and forced/skipped tick workarounds.

## Replacement

`FrameTimeline::nextOutput(now)` reserves the source pair, exact k/N alpha,
and an absolute output deadline. GPU work is now prepared/submitted before
the timer wait, with only `Present(0)` at the deadline. See the follow-up
[FSR preparation fix](fsr-output-preparation-2026-11-07.md). The reservation is not recomputed after capture events
or timer wakes. Fractional 100 ns ticks are retained rather than accumulated
as rounded sleeps: 60 FPS 2x schedules 120 FPS / 8.333333 ms on every monitor.

The presenter always uses `Present(0)`. Queue capacity is back-pressure,
not a display tick. Actual Present intervals and FPS are measured and
reported separately from the requested rate. The GUI calls the old vsync
control **Tearing**; the legacy CLI `--vsync on|off` controls tearing only.

Source buffering defaults to two source periods + 10 ms and respects the
existing latency limits. Starvation is diagnostic only. Missing source data
waits without synthesizing repeated output. Late work discards expired
slots instead of producing a catch-up burst. Paused sources retain the
last real frame and resume without warping across the pause. Capture-side
cadence classification, timestamp regularisation, GPU fences, replay and
hidden-overlay slot retirement remain.

Only explicit `Max` chooses display-rate FPS. Fixed 2x/3x/4x never use
display Hz to choose their output interval or interpolation positions.
FSR is restricted to exact t=0.5 on fixed output, even with a wide window;
real frames and other k/N positions cannot be replaced by its cached middle.

## Offline checks

`--timeline-self-test` is CPU-only; no live capture, overlay or GPU workloads
were launched. It verifies:

- 63 clock combinations: source 24/30/60 FPS, multipliers 2/3/4, and monitors
  at 60/75/120/144/165/240/360 Hz. Exact ordered k/N positions and deadlines
  agree with the source-rate clock; no missed outputs under sub-millisecond
  simulated wake jitter. Starvation reports do not change the budget/clock.
- 60 FPS 2x schedules 120 FPS on all seven monitor rates, with at most
  0.034 microseconds of **deadline arithmetic rounding error** (not a claim
  of actual Windows timer or GPU timing accuracy).
- Streaming auto-cadence with refresh-quantized WGC timestamps and 1.5-3.5 ms
  delivery jitter: approximately 120 FPS at 120/144/165/240 Hz, with no wrong
  output positions and bounded retained source frames.
- Explicit Max uses the FPS timer at its chosen rate.
- Late work skips expired slots without catch-up bursts.
- Exhausting the entire queued interval still retires stale capture slots,
  so a full producer ring cannot deadlock while waiting for fresh frames.
- A paused source displays its final real frame once and resumes without a
  stale interpolated pair.

## Physical limit

This controls generated/submitted FPS, not the monitor's physical scanout.
A fixed 144 Hz panel cannot show 120 distinct full frames with perfectly
even holds; a 60 Hz panel cannot display all 120 outputs. DWM composition,
GPU throughput and swap-chain back-pressure can also reduce actual delivery.
Even 120 FPS scanout needs working VRR with independent flip, or a compatible
120/240 Hz display mode. The application never changes monitor settings.
Live scanout and FSR image quality were not tested by this change.

Release build and `build-project/Release/motion_enhancer.exe --timeline-self-test`
pass. The running executable was not terminated by the agent; a temporary
output directory was used while it was locked, then the normal release was
rebuilt once it became available.
