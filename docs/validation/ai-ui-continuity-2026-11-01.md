# Current-proposal icon continuity — 2026-11-01

## Scope

After the Text/HUD removal, the user confirmed that stationary icons' cyan
boxes appear/disappear. Repeated identical-frame tests do not test that symptom.
No live sequence of their particular dropout has been supplied, so its exact
cause is not established. This change addresses two concrete discontinuities
in the original decoder: the single confidence cutoff and score-only selection
at the 32-region cap.

Both existing icon backends now share the same continuity policy:

- A new icon still requires confidence **>=0.10**.
- An icon from the previous fresh result may continue with a **current**
  proposal >=0.075 and padded-box IoU >=0.60. Weak proposals elsewhere cannot
  acquire new regions; one old icon cannot spawn multiple weak continuations.
- Current-supported existing icons receive capacity priority before unrelated
  newcomers. Within each priority group, confidence ranking remains stable.
  NMS remains IoU >0.5, padding remains two detector pixels and the cap is 32.
- Output coordinates/confidence are always from this inference. No old-box
  hold, interpolated rectangle, motion guess, geometry union or expanded HUD
  strip is used. Missing/below-floor proposals release their regions immediately.
- Only a fresh earlier published result in the same generation can authorize
  continuation. The original 600 ms snapshot expiry, gap invalidation,
  rejected in-flight generations and fail-closed behavior remain unchanged.

This deliberately changes the *continuation* confidence rule, not the model or
new-icon threshold. It may reduce confidence/rank flicker, but cannot recover
an icon for which the model produces no usable proposal. False positives remain
possible, including supported low-confidence continuations. It is not a trained
HUD classifier or a live temporal-accuracy/4 ms guarantee.

## Unchanged behavior

Native per-pixel UI masking, synthesis, flow/refinement/blend/depth behavior,
GPU preprocessing, detector weights and matrix execution are unchanged.
Rectangles remain allow-lists, never filled protection masks. Detection remains
off by default, at most 5 Hz, asynchronous and one job in flight. Capture does
not wait for inference; continuity decoding stays on the existing worker.
No Text/HUD option or fallback is reintroduced.

## Reproducible checks and preservation

`--ui-regions-self-test` now also exercises changing proposal confidence,
current coordinates, absent/unrelated/invalid/relocated proposals, weak-match
one-to-one admission, expiry/order checks, capacity score changes and immediate
slot release. `test_temporal.py` uses tiny generated ONNX proposal fixtures
through actual DirectML GPU preprocessing and the asynchronous worker to test
those transitions. These are regression fixtures, not game-quality evidence.
`--ui-detect-test` now emits per-inference `region_history` in its offline JSON.
Normal capture does not record a history trace.

Diagnostics:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-continuity-20261101-m6vhw6fi\`.

Before-change runtime/source/model/shader archive:
`backup/2026-11-01_ui-continuity-before-um10ip5i/`.

No training, dataset processing, download, environment/package update,
installation, commit or push is performed. Only `build-project` is built;
flow-model staging remains OFF and optional matrix support remains ON.

## Validation results

- Canonical Release build passed.
- **33/33** safety/numeric/native unit tests passed, including five new real
  GPU-worker changing-input tests (eight inferences each):
  - 0.11/0.085 matched score alternation stays detected on all eight updates;
    the reported coordinates change with the current proposal.
  - Starting at 0.085 does not acquire an icon; a subsequent 0.11 acquires it.
  - Missing proposals alternate detected/empty immediately, without lingering.
  - 0.074 cannot continue, even with recent matching history.
  - An unrelated score changing from 0.01 to 0.95 does not evict the 32
    current-supported established icons.
- The CPU/native region self-test passed original bounds/NMS/expiry plus
  continuation, one-to-one weak matching, invalid/unrelated/relocated evidence,
  ordered fresh-history rules, stable capacity and immediate free-slot reuse.
- **96 real-model same-image calls** across two fresh instances per backend
  remained stable. Both backend-specific box lists and raw checksums exactly
  match the prior icon-only build on the saved GUI fixture; the detector graph
  itself has not changed. Matrix changed-input replay and WMMA checks passed.
- Eight native/integration checks passed: native D3D12/projection, independent
  AI/DIS refinement and blending-mask selection, UI gating/debug safety,
  missing-model failure, parent-mask-off bypass and real-detector integration.
- Separate debug regression and owned GUI tests passed; exactly two icon
  choices remain. No extra inference is introduced by the debug view.
- Scoped source/document whitespace checks passed.

`preservation-final.json` verifies all **20 protected original hashes**, all
**12 active runtime model/licence/metadata files**, all **31 shader binaries**
and **96 unrelated source/header/shader files** match the before-change state.
The installed application, datasets, checkpoints, trainer settings, branding,
flow implementation and native pixel-mask shaders remain untouched.

Release executable SHA256:
`0432109bf38f8c77f92b5537ab825be095e938c1b360bdc46e814ed0fe82ce06`.

Logs: `unit-native.log`, `regions-self-test.log`, `consistency/`, `native/`,
`debug/` and `gui.log` in the diagnostics directory above. These tests establish
the intended continuity and fail-closed transitions, not the accuracy or exact
cause of the user's live-game dropouts. Full model misses still require better
current-frame evidence; no stale-result workaround is hidden here.
