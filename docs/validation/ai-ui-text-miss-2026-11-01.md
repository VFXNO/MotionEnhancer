# Saved HUD text miss — 2026-11-01

## User-supplied case

`build-project/Release/captures/20261101-124214/` contains eight original
2560x1440 game frames and twenty presented debug frames. The user clarified
that **this capture demonstrates missing text, not wrong-area detections**.
It must not be used as evidence of scene false positives.

The visible Arabic objective text at the top-left is not covered by the saved
cyan allow-list. The debug output shows one region around the bottom-left dial.
The user-supplied originals were read only and were not annotated/overwritten.

## Reproduction and raw proposals

The existing canonical EXE was run with `--ui-detect-test`, explicit `icons`
and `matrix` selections, four inferences per image and separate opt-in raw
output files. This is **64 inferences across 16 fresh detector instances**.
Both backends missed the quest text on all eight source frames.

At the current 0.10 acquisition threshold, each fresh instance returned zero
regions. This does not mean the saved live overlay was identical: it can use a
fresh result from an earlier asynchronous detection outside these eight frames.

Raw proposal inspection used a manually identified source-coordinate reference
around the quest text `(50,30)-(350,132)`. A proposal was considered geometrically
relevant for this diagnostic if it covered at least half that reference and its
area was at most four times the reference area. This is not a general semantic
accuracy metric or a new runtime filter.

| Backend | Best relevant quest proposal score across the eight frames |
|---|---:|
| DirectML icons | 0.05377–0.05823 |
| Verified matrix icons | 0.05301–0.05759 |

For example, DirectML `game_000` proposal 6804 has score `0.05673218` and
unpadded source bounds `(17,39)-(367,161)`, covering the objective text.
It is below **both** the 0.10 new-region threshold and the 0.075 continuation
floor. It therefore never reaches the native pixel mask. NMS/the 32-region
cap, empty native evidence and rendering are not what discards this proposal.
The two backends show the same issue; switching execution backend is not a fix.

## Resolution/context controls

The normal 2560x1440 input is reduced to 640x360 content inside the fixed
640x640 letterbox. Separate, temporary 2x/4x top-left crops were tested on
`game_000` and `game_007`, using the same model and unchanged thresholds.
These controls returned only the small objective **icon**, at scores around
0.199–0.217, not the adjacent text. Enlarging the text alone did not restore
text coverage in these controls. No crop/tiling logic was installed at runtime.

The model does produce weak geometrically relevant text proposals, so lowering
the global threshold could admit this particular example. It would also admit
weaker proposals generally and is not validated against the separately reported
wrong-area issue. No threshold was lowered or advertised as an accuracy fix.
The current icon model is not a reliable game-HUD text localizer; continuity
does not turn its low-confidence proposals into reliable text recognition.

## Preservation and limitations

Diagnostics, raw outputs, JSON, annotated copies and control crops are under:
`C:\Users\bsher\AppData\Local\Temp\opencode\me-ui-capture-20261101-y2pf83m2\`.

Key evidence: `icons/`, `matrix/`, `raw-target-analysis.json`,
`debug-contact-sheet.png`, `resolution-controls-native/summary.json` and
`preservation-after-analysis.json`. The native-resolution control folder is the
authoritative crop test; earlier inspection-only previews are not inference
inputs or runtime changes.

SHA256 checks confirm that all 28 supplied capture files, all 12 runtime
model/licence/metadata files and the canonical executable remain unchanged.
No application source, active model, native mask, flow/refinement/blending,
training data/checkpoint/settings or installed program was changed. No training,
download, environment update, build, commit or push was performed.

This is an eight-frame text-miss diagnosis, not general HUD accuracy, a live
temporal test, or a latency benchmark. It does not resolve the separately
reported unwanted detections. The rejected Text/HUD path remains removed.
