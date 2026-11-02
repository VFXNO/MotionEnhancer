# Motion Enhancer for Windows x64

Run the Setup executable to install for the current user. No administrator
rights or Visual Studio/Windows/FidelityFX SDK installation is needed.
The default destination is `%LOCALAPPDATA%\Programs\MotionEnhancer`.
Use a writable folder: captures, exported models and trainer data live beside
the application. Windows 10/11 x64, a Direct3D 12 / Shader Model 6 GPU and a
current graphics driver are required. Windows N editions also need Microsoft's
Media Feature Pack for video decoding. Graphics drivers and Windows features
are not replaced by this installer.

## What's new in 1.1.0

- **HUD persistence:** the UI mask learns, with no model, where the HUD sits on
  screen (structure that stays put while the camera moves) and keeps it sharp;
  nothing outside it is protected, so plain sky and ground stay smooth.
- **AI: UI regions** now adds to that protection instead of limiting it.
- **AI trainer:** a new Label UI page pre-paints the HUD on frames from your
  videos for you to correct; UI model exports are readable when the trainer
  ran as administrator; closing the trainer during a job no longer raises.
- Every AI mode now ships with its model: the AI flow model with its DIS
  companion (`models\ai_flow.dis.onnx`, which the default AI engine needs),
  the trained temporal UI model (`models\ai_flow_ui*.onnx`, experimental), and
  the OmniParser icon detector for **AI: UI regions** (`models\ui_detect.onnx`,
  plus its Vulkan matrix conversion for **Matrix: icons**).

## Included offline

- `motion_enhancer.exe`, compiled GPU shaders, native FidelityFX Optical Flow.
- ONNX Runtime, DirectML and Microsoft C++ / OpenMP runtime DLLs.
- AI flow model and its DIS companion with their JSON metadata, the trained
  temporal UI model, the OmniParser UI detector (DirectML and Vulkan matrix
  forms; AGPL-3.0, licence in `licenses\` and `models\`), MiDaS depth model,
  and the native neural model (`models\neural_model.bin`, selected with Load in the GUI
  or `MOTION_ENHANCER_NEURAL_MODEL` for CLI use).
- AI trainer source, dependency setup launcher, and third-party notices.

Launch **Motion Enhancer** from the Start menu. Keep the shaders, models and
DLLs with the executable; moving only the EXE will break GPU/AI operation.
The portable ZIP contains the same runtime files; extract the entire folder.
The unsigned installer may trigger Windows SmartScreen. Verify the supplied
SHA256 checksums and the source of the download before running it.

## Installer appearance and music

Setup matches the app's monochrome dark interface, animated emblem and rounded
controls. A quiet, original ambient soundtrack loops during interactive setup.
Use **Music** (or **Alt+M**) to mute/unmute it at any step. Add `/NOMUSIC` to the
Setup command line to start muted; `/S` silent installation never plays music.
Music stops when setup closes and does not change your system volume. The
uninstaller uses the dark theme but does not start music. No track is downloaded,
and no music is added to the running Motion Enhancer application.

## AI trainer (one-time Internet setup)

Run **Set up AI Trainer** from the Start menu or `Setup-AITrainer.cmd`.
The installer also offers this as an optional component. It downloads several
GB, rather than redistributing a machine-specific Python virtual environment.
Setup installs a private virtual environment, Qt UI, ONNX export tools, FFmpeg
and FFprobe. It uses an existing Python 3.14 x64 when available; otherwise it
downloads a signed Python 3.14 installer for the current user.

GPU selection is automatic for Radeon RX 9070 (gfx1201) and NVIDIA.
Other hardware defaults to CPU, with a warning; do not install gfx1201 packages
on a different AMD architecture. For a supported different AMD target use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\ai-train\Setup-AITrainer.ps1 -Backend amd -AmdTarget gfx1100
```

AMD support depends on AMD's ROCm wheel availability and driver support.
The NVIDIA option uses official CUDA 13.0 PyTorch wheels and needs a compatible
driver. `-Backend cpu` works without GPU training support. Setup never installs
or changes graphics drivers. Downloads come from python.org, PyPI, AMD/PyTorch
package indexes and Gyan's FFmpeg builds. Their licenses apply separately.
An interrupted setup can be run again; it does not delete datasets or runs.

Once setup finishes, use **OPEN AI TRAINER** on the AI Model page. The installed
application is offered as an export destination. Training is not required to
use the bundled models or the AI blending-mask option.

## Updating and removing

Close the application and trainer before updating or uninstalling.
Reinstalling preserves existing model files, so a model exported by you is not
silently replaced. To restore a bundled model, rename your copy before repair.
Uninstall removes the distributed application files and shortcuts. Captures,
models, datasets, training runs, and the downloaded trainer environment are
preserved; remove those folders yourself only when you no longer need them.
