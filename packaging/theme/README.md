# Installer skin and soundtrack

`MotionTheme.dll` is an installer-only, x86 Unicode NSIS plug-in built by
`tools/Build-Package.ps1`. It uses the application's palette from
`src/GuiApp.cpp`: near-black surfaces, the original orbital motion emblem, flowing sidebar
lines, a step indicator, Segoe UI text, rounded cards and monochrome buttons.
The uninstaller uses the same skin. Native NSIS controls, license acceptance,
folder selection, component selection and real extraction progress remain in
charge of installation. Keyboard tab navigation, native control accessibility
and visible focus outlines are retained.

The DLL statically links its C++ runtime and depends only on Windows system
DLLs; it does not require the application or its runtimes to be installed first.
It and the audio are extracted to NSIS's temporary plug-in directory, not into
the application's runtime payload. They do not affect capture or inference.

## Application icon

`../assets/installer.ico` is also the application's emblem, embedded by `app.rc`
at 16, 24, 32, 48, 64, 128 and 256 pixels. Explorer, Start menu shortcuts and
Windows Installed apps use this embedded icon; no loose ICO is required at
runtime. The GUI, splash and overlay window classes load its large/small forms
instead of the generic Windows application icon. The icon has an orbital outline
with fully transparent corners, antialiased alpha and a matching legacy
AND mask at every resolution; there is no opaque square background.

`../assets/logo.svg` is the single master: the original two orbits, six balanced
outer nodes, and a bright core with three motion trails. Stroke weights and
spacing are refined, not replaced by a lettermark. The smallest ICO frames use
optically thicker lines and omit the tiny core halo/highlight to stay readable.
`tools/make_installer_assets.py --icons-only` regenerates the
ICO and transparent PNG without changing the soundtrack. The PNG is embedded
in both the app and theme DLL. `include/BrandLogo.h` supplies the shared vector
animation using the same orbital geometry: a constant-speed spark with a smooth
gradient tail, slowly orbiting nodes, a soft radial glow and gently flowing
motion trails. All parts share a seamless 12-second period; no stepped glow,
flicker or bouncing geometry. Startup draws the rings progressively, extends
nodes in sequence and sweeps in trails with smooth easing and a soft core bloom.
The app carries the same motion phase from splash to sidebar rather than
restarting it. The static Windows icon stays transparent.

Run `python tools\test_app_icon.py` to verify every embedded image, alpha/AND
masks, actual Windows compositing on light/dark backgrounds, shell extraction
and the GUI/splash icons using the off-screen startup test.
An optional executable path tests a portable or installed copy instead.
`docs/tools/check_logo_animation.py` (Pillow, as used by the README art tools)
checks real off-screen frames for moving nodes, spark, trails and glow, plus
the full loop seam and splash/ambient hand-off. Pass `--output docs/assets` to
refresh 30-fps animated previews and screenshots. A shared palette sampled
across each sequence avoids color shimmer without losing the splash artwork.
Re-run the updated installer to update an existing installation and its
shortcut. Windows may retain an old pinned icon until it is unpinned/re-pinned.

## Music

`../assets/setup-music.wav` is an original, procedurally composed 32-second
ambient instrumental loop: soft sine pads, a sparse bell motif and stereo delay.
It uses no third-party recordings or samples and is covered by this project's
MIT license. Its PCM peak is limited to approximately 16% of full scale; setup
does not modify system volume or fetch music over the network.

Music plays only during interactive setup. **Music** (or **Alt+M**) toggles it
on/off without restarting setup. `/NOMUSIC` starts muted, `/S` silent installs
never initialize the UI/audio, and uninstall starts without music. Music stops
on Finish, Cancel and window destruction; unavailable audio does not prevent
installation. The emblem animates only while setup is foreground/not minimized.

Regenerate the committed original WAV and multi-resolution icon using Python's
standard library (Python is not needed for ordinary package builds):

```powershell
python tools\make_installer_assets.py
```

Validate the actual UI, mute/default audio behavior, all page navigation,
payload integrity, uninstall, cancellation and preserved data:

```powershell
python tools\test_installer_theme.py
```

The test refuses to run if an application installation is already registered
for the current user. Use `--preview-only` to check pages/music/cancellation
without installing or modifying an existing installation. Optional
`--screenshots <directory>` captures only its own installer window; otherwise
test screenshots and scratch files are removed.
