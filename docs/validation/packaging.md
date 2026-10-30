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

## Application icon update (2026-10-30)

- Embedded the existing app/setup emblem in `motion_enhancer.exe`, with seven
  resolutions (16 through 256 pixels). GUI, splash and overlay classes load
  large/small embedded icons; the main shortcut and Installed apps registration
  explicitly select executable icon index 0.
- Rebuilt the FidelityFX-enabled application, installer and portable ZIP.
  `tools/test_app_icon.py` passed for development, portable and isolated-installed
  executables: exact embedded image bytes, Windows shell extraction, and actual
  off-screen GUI/splash large/small icons all matched the shared emblem.
- Isolated silent install/remove verified 68 payload hashes, shortcut target,
  shortcut icon and quoted `DisplayIcon` path. D3D12 and model-backed AI-mask
  self-tests passed; final archive checksums matched.
- Existing `C:\MotionEnhancer` registration and executable hash were unchanged.
  No live capture session was started to test the overlay icon. Reinstalling
  the new package updates an existing copy; pinned-icon cache refresh was not
  forced on this machine.

## Logo refinement and transparent icon (2026-10-30)

- Replaced the busy orbit/core/circuit logo with a bold, forward-leaning **M**
  inside a restrained circular badge. `packaging/assets/logo.svg` is the single
  master for the ICO and PNG; app, splash, installer and documentation now use
  the same artwork and a subtle rim highlight instead of the old effects.
- Fixed the actual square-icon cause: the old generator wrote fully opaque
  pixels across the entire icon. Every size now has zero-alpha corners,
  supersampled alpha edges, no hidden matte color, and a matching legacy AND
  transparency mask. Preview: [light/dark/transparent](../assets/logo-preview.png).
- The updated icon regression checks embedded PNG/ICO bytes, all seven sizes,
  transparency and AND masks, shell extraction and real `DrawIconEx` compositing
  on light/dark surfaces, plus actual off-screen GUI/splash window icons.
- Logo PNG data is embedded in the executable and installer skin DLL; no loose
  image is required on target machines. The shared renderer keeps its image
  stream alive and releases cached images before GDI+ shutdown.
- Rebuilt installer/portable ZIP passed their checksums and all 68 portable
  payload hashes. Actual release setup passed preview/music/cancel checks.
  An isolated namespace build of the same source, plugin and payload passed
  interactive/silent install/remove, data preservation, D3D12 and AI-mask checks.
  Existing application registration and executable hash were left unchanged.
- Refreshed app/setup previews, README banner and sidebar/splash animations;
  no inference or capture behavior was changed. Windows pinned-icon caches
  were not purged; re-pin an old shortcut if Windows still displays its old icon.

## Original logo and animation restored (2026-10-30)

- Supersedes the rejected M redesign above. Restored the original orbital
  identity: two rings, six nodes, a bright moving-frame core and three motion
  trails. Refined spacing, rounded strokes and small-icon optical weights;
  retained transparent alpha/AND masks instead of the old opaque square.
- Restored the full original animation in the shared app/setup renderer:
  circling spark with a fading tail, orbiting nodes, breathing glow and
  flickering trails. The splash again draws rings progressively, extends nodes
  in sequence, sweeps in trails and flashes the core. Windows icons stay static.
- Actual off-screen frame checks passed for each independently masked logo
  region: nodes (292 changed pixels), spark (203), trails (14), glow (6313),
  and staged splash assembly. `docs/tools/check_logo_animation.py` retains the
  regression and updates actual-app GIFs/screenshots; it never reads the screen.
- Rebuilt the application and themed installer/portable ZIP. Embedded icon,
  transparency/compositing and GUI/splash icon tests passed, as did setup
  preview/navigation/music/cancel checks without changing existing registration.
- Final portable executable repeated the animation checks successfully. All
  68 payload hashes and archive checksums matched; D3D12 and model-backed
  AI-mask self-tests passed. Refreshed sidebar/splash GIFs show the actual
  restored application animation, not a separate mock-up.

## Orbital polish and smoother animation (2026-10-30)

- Kept the original identity, with finer ring/stem weights, slightly smaller
  nodes and a better-balanced core/trail layout. Small-icon optical minimums
  retain legibility; all seven sizes still have transparent corners and AND masks.
- Replaced stacked glow discs with a soft radial fade and the segmented spark
  tail with a continuous gradient. Slower, constant-speed orbits and restrained
  trail flow replace flicker, changing geometry and springy node overshoot.
- Every ambient component shares a seamless 12-second period. Startup uses
  smooth easing and a soft core bloom, then carries its animation clock into
  the sidebar instead of resetting the logo's position or speed.
- Native off-screen exports passed exact loop-endpoint equality, seam-motion
  continuity and intro/ambient equality. Masked animation checks passed for
  nodes (221 changed pixels), spark (140), trails (78) and glow (2020), plus
  the staged splash. The final portable executable repeated these checks.
- Updated real-app screenshots, static artwork and 30-fps previews: 360 frames
  for each 12-second logo/sidebar loop and 78 frames for the 2.6-second splash.
  GIF checks verify frame count, duration and retained bright artwork; palettes
  are sampled across the sequence so the splash's blank first frame cannot
  erase later artwork during quantization.
- Rebuilt the FidelityFX-enabled application, installer and portable ZIP. All
  68 portable payload hashes, executable equality and archive checksums passed,
  along with embedded image/alpha/AND-mask, Windows shell/compositing,
  GUI/splash icon, D3D12 and model-backed AI-mask tests. Final setup preview,
  navigation, music controls and cancel passed without changing registration
  or the existing `C:\MotionEnhancer` executable hash. No installation was
  replaced, no icon caches were purged, and no commits or pushes were made.
