# Capture reservation lifetime race (2026-11-08)

The user reports forward/backward stutter with FSR both enabled and disabled.
Removing recovery timing did not resolve that observation. This follow-up
targets shared capture lifetime, not display pacing or recovery.

After the session restart, the workspace contained newer pacing commits
(HEAD `4878cfe3`). This change is layered on those commits; their output-clock,
missing-frame, and startup behavior is retained.

## Reproduced interleaving

1. Capture calls `onFrameArrived`; an unreserved newest frame may be replaced
   by a repaint (`replaceLast = true`). The timeline mutex is released.
2. Rendering calls `nextOutput` and reserves that frame/pair. Rendering may
   still be waiting for DXGI capacity and has not published a read fence yet.
3. Capture calls `popLast`. Previously it removed that frame unconditionally,
   releasing its slot for reuse despite the reservation.
4. Rendering could then read newer pixels through an older selection's slot.
   Timestamp-only ordering assertions cannot detect that content mismatch.

The earlier focused regression removed slot 1 in all four output modes
(Max, 2x, 3x, 4x), violating the reservation invariant. This is evidence of a
real lifetime bug, not proof it is the only cause of the reported live stutter.

## Fix

- `popLast` rechecks the reservation under the same mutex as `nextOutput`.
  When rendering won, it returns -1, keeps the slot/history, and capture
  appends the new repaint instead. Normal unreserved replacement still works.
- Both output selection paths keep a monotonic reservation high-water mark.
- `replaced` counts actual removals, not proposals later vetoed by a reservation.
- Slot acquisition excludes the producer-owned diff reference. A defensive
  self-comparison check cannot report an incoming image as its own duplicate.
  This is a safety guard; the CPU tests do not reproduce a live D3D11 alias.

No recovery epoch, queue drain, pacing mode or additional CPU fence wait is
introduced. Existing GPU reader fences and fixed-multiplier deadlines remain.

## Verification

```powershell
cmake --build build-project --config Release --target motion_enhancer --parallel 2
.\build-project\Release\motion_enhancer.exe --timeline-self-test
```

Both succeeded. The normal Release executable was rebuilt, with no process
termination or alternate executable required.

- Eight deterministic interleaving checks pass: capture wins / rendering wins
  in Max, 2x, 3x and 4x. These simulate texture reuse, check reserved pixels,
  and check that retire/next-output calls retain protection. The capture-first
  cases also check that ordinary replacement and subsequent reservation work.
- All 63 monitor-independent fixed clock cases and the existing Max, streaming,
  missing-frame, pause, late-slot and FSR preparation models pass.
- The streaming suite retains its existing bounded thresholds: some
  near-refresh/repaint cases permit content jumps or empty-input waits. A suite
  PASS does not mean every source/display combination has zero stutter.

No live capture, GPU texture-race reproduction, synthesized-image quality,
input-to-photon timing or physical scanout verification is claimed. Test
`build-project/Release/motion_enhancer.exe` with the same capture and multiplier,
first with FSR disabled, then enabled.
