# Single active build folder

The user selected `build-project` as the sole active build folder.

## Current executable

`build-project/Release/motion_enhancer.exe` now contains the verified native
24 FPS celestial-character animation. The other native features, runtime DLLs
and shaders are retained. The existing `Release/models/` contents are unchanged.

The trainer's saved `export` path now points to
`build-project/Release/models/ai_flow.onnx`; all other settings are unchanged.
No export or training job was run and no model was replaced.

## Preserved archive

`backup/2026-10-30_single-build-archive-mzb1slcg/` contains:

- Original `build-character-preview/`, `build-refinement-preview/` and
  `build-ffx-local/` trees, including their different model exports and captures.
- `previous-build-project-release/`, a hash-verified copy of the former main
  runtime, including its existing models.
- `trainer-settings-before.json` and before/after preservation manifests.

The archive is local and Git-ignored. No archived files, training runs,
datasets, SDKs, existing backups or installed files were deleted.

## Verification

Passed against the canonical executable:

- Three asset tests and native off-screen checks for all three pages, 24 FPS
  frame selection, the nine-second loop, transparency and splash fade/handoff.
- Existing Explorer and GUI/splash icon checks.
- All 175 archived files and 75 former-main runtime files matched their original
  names and SHA256 hashes; all 20 protected files matched too, including retained
  models/checkpoints, the dataset index and the installed executable.
- Only the trainer's `export` setting changed; build discovery returns only
  `build-project/Release`.

Run GUI checks serially: concurrent off-screen GUI processes can compete for
Windows foreground ownership and advance the ambient clock during snapshots.
An initial parallel cadence check saw that interference; the serial rerun passed.

Executable SHA256 at consolidation, before the subsequent
[AI-refinement restoration](ai-refinement-restored-2026-10-30.md):
`69f70916a488bc566b938a9ea66d61d87f83a71e74fd88ede65ac5ed79e54a5d`.
