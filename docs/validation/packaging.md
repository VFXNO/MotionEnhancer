# Windows packaging and cleanup validation

Validated 2026-10-29/30 on Windows x64 with an AMD Radeon RX 9070 XT.

## Distributables

- `dist/MotionEnhancer-1.0.0-Setup-x64.exe` (approximately 47 MiB with theme/music).
- `dist/MotionEnhancer-1.0.0-win64.zip` (approximately 48 MiB).
- `dist/SHA256SUMS.txt`; both final archive checksums verified.

The clean CMake install payload has 68 files plus its SHA256 manifest. It
contains all 29 compiled shaders, FidelityFX statically linked into the app,
ONNX Runtime/DirectML, release x64 MSVC/OpenMP DLLs, AI flow/depth/neural models,
trainer source/setup launchers, and license notices. Build intermediates,
debug logs, raw HLSL, SDK development trees and local training data are excluded.

## Checks passed

- Release build with native FidelityFX Optical Flow enabled.
- Silent per-user installation into a path containing spaces; all 68 installed
  payload hashes, Start menu shortcuts, and install/uninstall registry entries.
- Installed `--d3d12-self-test`, `--ai-blend-mask-self-test` (including the
  installed ONNX model and ring lifetime checks), and nine-case projection test.
- Installed 1080p offline FFX, AI with blending mask, and native neural replay
  (the neural model selected explicitly); depth inference with MiDaS/DirectML.
- Installed GUI startup and off-screen GUI snapshot.
- Portable ZIP extraction, D3D12 and model-backed AI-mask self-tests.
- Runtime tests used a separate working directory and a PATH limited to
  Windows System32, rather than depending on the checkout or development PATH.
- Reinstall preserves a modified model; uninstall removes the application,
  shortcuts and registry entries while retaining models, captures, datasets
  and training runs.
- Final installer refuses update and uninstall while its application is running;
  these operations work after closing the test application.
- Installed trainer detects the installed app as an ONNX export destination.
- Existing trainer environment passes setup's `-CheckOnly`: all imports,
  ROCm tensor execution on RX 9070 XT, FFmpeg and FFprobe. Setup scripts parse
  successfully in Windows PowerShell 5.1; Python/FFmpeg download endpoints work.

Full fresh trainer dependency download/installation on a clean computer was
**not** performed. Trainer setup is Internet-dependent and downloads several
GB; compatible drivers are prerequisites. Runtime tests were performed on the
development PC, not a clean Windows VM. Setup is unsigned. No new long-running
WGC/overlay or frame-pacing validation was performed as part of packaging.

## Cleanup

Removed seven obsolete build trees (`build`, `build-d3d12`, `build-ffx`,
`build-ffx-local`, `build-ffx-test`, `build-local`, `build-vendored-ffx`),
one-off audit runners, raw audit images/logs, package-verification scratch
installs, root diagnostics, generated flow-test outputs and trainer bytecode
caches. Approximately 2 GiB of generated files were removed.

Retained `build-project` as the only development build, its active FidelityFX
static libraries in `bin/ffx_sdk`, SDK source trees, all application source,
backups, test fixtures, models, the working trainer environment, datasets and
training runs. Short earlier validation reports are archived beside this file;
their referenced bulky raw artifacts have been deleted. No commits or pushes
were made. This cleanup does not remove objects from Git history.

## Installer theme and music update (2026-10-30)

- Installer/uninstaller now use the app's monochrome palette, animated emblem,
  sidebar, rounded cards/buttons and dark native input/list/scroll controls.
  [Preview](../assets/installer.png).
- Original, generated 32-second stereo ambient soundtrack. Checked its WAV
  format/duration and approximately 16% PCM peak limit. No external recordings,
  music downloads or system-volume changes.
- Default playback, `/NOMUSIC`, persistent per-session mute/unmute across pages,
  Finish/Cancel exit and uninstaller-without-music behavior passed UI checks.
- All welcome/license/components/location/install/finish and removal pages were
  exercised. Back/forward navigation, visible destination/Browse controls,
  all 68 installed payload hashes and preserved model/user data passed.
- Installed D3D12 self-test and model-backed AI-mask self-test passed. Silent
  install/removal passed without initializing the skin/audio. The skin DLL has
  only Windows system DLL dependencies; skin/audio are not installed as app
  runtime dependencies.
- Final release installer preview/music/Cancel checks preserved the existing
  `C:\MotionEnhancer` registration. Final full roundtrips used an isolated copy
  of the same installer source/DLL/payload with only registry, Start menu and
  default-directory namespaces changed, so the existing installation was not
  overwritten. Its registration and executable hash remained unchanged.
- `tools/test_installer_theme.py` provides the regression test; use
  `--preview-only` when an application is already installed. Full tests refuse
  to alter a pre-existing registration. Temporary test installations/artifacts
  are cleaned, and the short preview image is retained as documentation.

Checks ran on the current Windows 11 machine. Different Windows versions,
monitor DPI settings and audio devices have not all been visually/audibly
evaluated. Audio start/mute was checked through Windows playback success and
UI state, not a subjective listening study. No application rendering/capture
code changed in this update.
