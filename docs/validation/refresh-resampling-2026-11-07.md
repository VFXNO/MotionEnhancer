# Fixed multipliers on incompatible refresh rates (2026-11-07)

**Superseded at the owner's request.** Refresh resampling and the whole
display-driven output pacing pipeline were removed later the same day.
Current behavior and checks: [source FPS clock](source-fps-clock-2026-11-07.md).
The results below describe the earlier policy, not the current build.

## Cause and policy

60 FPS at 2x gives 120 unique frames per second. A fixed 144 Hz display
cannot show those evenly: 24 refreshes per second repeat a frame. The
timeline explicitly caused this by rounding each refresh's blend position
to k/N (`roundToMultiple`). This is temporal judder; it does not establish
the cause of spatial ghosting in a single screenshot.

Removed that rounding. On composed/vsync presentation, fixed multipliers
are now targets: preserve evenly held output when the display rate is a
whole multiple of the target; otherwise interpolate at each refresh's
actual media time. Thus 60 FPS 2x on 144 Hz becomes effective 2.4x / 144 FPS.
True fixed 2x remains available on a compatible 120 Hz mode, or with own-time
independent-flip presentation (working VRR is needed for even scanout on a
144 Hz panel). No display settings are changed by the application.

For the new resampling path, `Selection::canUseMidpoint` only permits FSR's
cached t=0.5 image at exactly t=0.5, even with a wide `--fsr-window`. Other
positions stay on native interpolation. FSR is no longer rejected at
startup based only on the nominal multiplier: resampling can change the
positions it will be asked to display.

## Verification

Release build succeeded with MSVC. Only CPU timeline simulations were run;
no live capture, overlay, or GPU benchmarks were launched.

```powershell
cmake --build build-project --config Release --target motion_enhancer --parallel 2
.\build-project\Release\motion_enhancer.exe --timeline-self-test
```

The strengthened tests were first run against the old rounding policy and
failed. In the steady-state 13-second measurement interval:

| Scene | Repeated refreshes before | After | Worst step-time error after |
|---|---:|---:|---:|
| 60 FPS 2x / 144 Hz, auto cadence | 312 (24/s) | 0 | 0.29 ms |
| 60 FPS 2x / 144 Hz, explicit 60 FPS | 312 | 0 | 0.34 ms |
| 59.94 FPS 2x / 144 Hz | 313 | 0 | 0.30 ms |
| 60 FPS 2x / 165 Hz | 585 | 0 | 0.27 ms |
| 60 FPS 2x / 180 Hz | 780 | 0 | 0.43 ms |
| 30 FPS 2x / 144 Hz | 1092 | 0 | 0.27 ms |
| 24 FPS 2x / 60 Hz | 156 | 0 | 0.60 ms |
| 72 FPS 2x / 165 Hz, 12 ms flow | 272 | 0 | 0.29 ms |

All these scenes also have zero uneven holds and zero step-time errors
above 1 ms after the fix. Additional checks cover:

- FSR substitution with the widest allowed window on the new resampling path;
- compatible 60 FPS 2x / 120 Hz: 50% real frames and 50% exact midpoints,
  with and without simulated FSR substitution;
- compatible even two-refresh holds at 30 FPS 2x / 120 Hz and
  60 FPS 2x / 240 Hz, preserved intentionally;
- 3x/4x output capped by 144 Hz;
- 120 consecutive ordered k/N positions under free pacing at 60 FPS 2x /
  144 Hz (no repeated or skipped position).

`--timeline-self-test`: PASS. FSR checks model the cached image's timestamp;
they do not execute or assess the FSR DLL's image quality. Live game motion
and spatial ghosting remain unverified.
