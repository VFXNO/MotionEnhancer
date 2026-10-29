# Installer skin and soundtrack

`MotionTheme.dll` is an installer-only, x86 Unicode NSIS plug-in built by
`tools/Build-Package.ps1`. It uses the application's palette from
`src/GuiApp.cpp`: near-black surfaces, an animated orbit emblem, flowing sidebar
lines, a step indicator, Segoe UI text, rounded cards and monochrome buttons.
The uninstaller uses the same skin. Native NSIS controls, license acceptance,
folder selection, component selection and real extraction progress remain in
charge of installation. Keyboard tab navigation, native control accessibility
and visible focus outlines are retained.

The DLL statically links its C++ runtime and depends only on Windows system
DLLs; it does not require the application or its runtimes to be installed first.
It and the audio are extracted to NSIS's temporary plug-in directory, not into
the application's runtime payload. They do not affect capture or inference.

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
