# Motion Enhancer - Enhanced 8-Level Pyramid ZNCC Frame Interpolator

A C++17 frame interpolation application implementing an **8-level Gaussian pyramid** with **Zero-mean Normalized Cross-Correlation (ZNCC)** block matching motion estimation, multi-candidate spatial predictors (EPZS), 2D Gaussian block weighting, direct quarter-pel bilinear refinement, and **advanced artifact refinement** (photometric-aware occlusion gating, bilateral hole inpainting, Laplacian detail restoration, and color bounding box clamping).

## Features & Improvements

- **8-Level Coarse-to-Fine Pyramid**: Builds an 8-level Gaussian image pyramid ($L_0$ to $L_7$) using a 5-tap separable binomial filter $\frac{1}{16}[1, 4, 6, 4, 1]$ and dyadic downsampling.
- **Multi-Candidate Spatial Predictors (EPZS)**: Evaluates hierarchical coarse predictor, spatial neighborhood vectors, and zero vector before searching, eliminating traps in repetitive patterns and local minima.
- **2D Gaussian Block Weighting**: Replaces flat box blocks with 2D Gaussian window weights, concentrating matching metric on block centers and eliminating motion halo / edge-bleeding artifacts.
- **Direct Quarter-Pel & Half-Pel Subpixel Refinement**: Directly maximizes ZNCC in the continuous image domain using bilinear sampling at $0.5\text{ px}$ and $0.25\text{ px}$ candidate steps.
- **Level-Adaptive Block Sizing**: Dynamically scales block sizes across the 8 levels ($14\times 14$ on coarse levels down to $6\times 6$ on fine levels).
- **Color-Guided Edge-Preserving Vector Median Filter**: Prevents foreground motion from bleeding into background by weighting neighbors by RGB color similarity.
- **Artifact Refinement Suite**:
  - **Photometric-Aware Occlusion Gating**: Measures color discrepancy $\|I_0(\mathbf{s}_0) - I_1(\mathbf{s}_1)\|$ and soft-suppresses occluded sources to eliminate ghosting contours.
  - **Bilateral Motion Hole Inpainting**: Propagates surrounding background motion into disocclusion gaps to prevent smearing.
  - **High-Frequency Detail Restoration (Laplacian Detail Transfer)**: Injects warped high-frequency detail layers back into synthesized pixels to compensate for bilinear resampling blur.
  - **Local Color Bounding Box Clamping**: Constrains synthesized pixels within the local color bounds of the source samples, preventing ringing spikes and color fringing.
- **Multi-Core Acceleration**: Multi-threaded block matching and synthesis with OpenMP.
- **Zero External Dependencies**: Bundles `stb_image` and `stb_image_write`.

---

## Directory Structure

```
Motion enhancer/
â”œâ”€â”€ CMakeLists.txt              # CMake build script (C++17, OpenMP, MSVC /O2)
â”œâ”€â”€ include/
â”‚   â”œâ”€â”€ Image.h                # Multi-channel float image container & bilinear sampler
â”‚   â”œâ”€â”€ Pyramid.h              # 8-level Gaussian pyramid builder
â”‚   â”œâ”€â”€ MotionVector.h         # Flow field, color-guided median filter, flow visualizer
â”‚   â”œâ”€â”€ ZNCCMatcher.h          # Coarse-to-fine ZNCC solver with EPZS & sub-pel search
â”‚   â””â”€â”€ FrameInterpolator.h   # Bidirectional flow, occlusion detection & artifact refinement
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ Image.cpp
â”‚   â”œâ”€â”€ Pyramid.cpp
â”‚   â”œâ”€â”€ ZNCCMatcher.cpp
â”‚   â”œâ”€â”€ FrameInterpolator.cpp
â”‚   â””â”€â”€ main.cpp               # CLI entry point and synthetic benchmark
â””â”€â”€ third_party/
    â”œâ”€â”€ stb_image.h            # Single-header image loader
    â””â”€â”€ stb_image_write.h      # Single-header image saver
```

---

## Build Instructions

### Prerequisites
- CMake 3.16 or newer
- Visual Studio 2022 (MSVC) or GCC 9+ / Clang 10+ with OpenMP support

### Building with CMake
```powershell
# Configure build with Visual Studio 2022 x64
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build Release binary
cmake --build build --config Release
```

The executable will be generated at `build/Release/motion_enhancer.exe`.

### Optional AMD FidelityFX Optical Flow

The repository includes AMD FidelityFX SDK v1.1.4 and its `FidelityFX_SC.exe`
shader compiler under `third_party/FidelityFX-SDK-v1.1.4`. The default build
enables native D3D12 Optical Flow without a separate AMD SDK download:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The integration invokes AMD's supplied `tools/binary_store/FidelityFX_SC.exe`
through the SDK CMake targets. It retains one temporal context and falls back
to the native MSAD graph if AMD initialization or dispatch fails. SDK v1.1.4
suppresses vectors through temporal frame index 5 after a reset, so a stream
is primed once and every sequential pair thereafter submits only its new
current frame. A changed source texture resets and re-primes the history.
Disable it with `-DMOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW=OFF`.

---

## Usage

### Frame Interpolation
```powershell
.\build\Release\motion_enhancer.exe <frame0.png> <frame1.png> <output.png> [options]
```

### Command-Line Options
| Option | Type | Default | Description |
|---|---|---|---|
| `<frame0.png>` | Positional | - | Input image at $t = 0$ |
| `<frame1.png>` | Positional | - | Input image at $t = 1$ |
| `<output.png>` | Positional | - | Output path for interpolated image |
| `--time <float>` | Float | `0.5` | Target intermediate timestamp $t \in (0.0, 1.0)$ |
| `--levels <int>` | Integer | `8` | Number of pyramid levels (default 8) |
| `--block-size <int>` | Integer | `8` | Nominal block matching window size |
| `--search-radius <int>` | Integer | `4` | Refinement search radius per level |
| `--coarse-radius <int>` | Integer | `8` | Search radius at coarsest level ($L_7$) |
| `--grid-step <int>` | Integer | `2` | Grid step: `1` for dense, `2` for $2 \times 2$ grid |
| `--subpel <mode>` | String | `quarter` | Subpixel mode: `none`, `parabolic`, `half`, `quarter` |
| `--no-spatial-pred` | Flag | Off | Disable spatial predictor evaluation (EPZS) |
| `--no-gaussian-weight` | Flag | Off | Disable Gaussian window block weighting |
| `--no-adaptive-block` | Flag | Off | Disable level-adaptive block sizing |
| `--no-bidirectional` | Flag | Off | Disable backward flow and occlusion checking |
| `--no-photo-gate` | Flag | Off | Disable photometric-aware occlusion gating |
| `--no-hole-inpaint` | Flag | Off | Disable disocclusion motion hole inpainting |
| `--no-detail-restore` | Flag | Off | Disable high-frequency detail restoration |
| `--no-color-clamp` | Flag | Off | Disable local color bounding box clamping |
| `--detail-strength <flt>` | Float | `0.65` | Detail restoration strength in $[0.0, 1.0]$ |
| `--save-flow <path>` | String | None | Export forward flow visualization to PNG |
| `--save-occ <path>` | String | None | Export occlusion confidence mask to PNG |
| `--test` | Flag | Off | Run synthetic benchmark with ground truth |
| `--list-windows` | Flag | Off | List active desktop windows available for WGC capture |
| `--capture-window <title>`| String | None | Real-time GPU frame interpolation overlay on target window |

---

## Real-Time GPU Shader Overlay (WGC + D3D12)

### Graphical Interface

Launch the executable without arguments (or pass `--gui`) to open the native Windows interface. It uses a custom dark gaming-styled theme with neon accent tabs, buttons, and panels:

```powershell
.\build\Release\motion_enhancer.exe
```

- **Live Capture** lists capturable desktop windows and starts the GPU overlay for the selected window.
- **Offline Interpolation** provides file pickers, interpolation timing, and direct controls for every matcher and synthesis option.

For live capture, **Source FPS** defaults to `Auto`, which estimates WGC frame cadence and stabilizes it to `24`, `30`, or `60` FPS. You can select a fixed rate when the content cadence is known. Motion Enhancer accepts source frames at that cadence, then generates timestamp-correct intermediate positions at the display refresh rate.

**Output multiplier** selects `2x`, `3x`, `4x`, or `Max`. The presenter targets source FPS multiplied by this value and caps the result at the display refresh rate. For example, a 30 FPS source produces 60, 90, or 120 FPS at `2x`, `3x`, or `4x`. `Max` retains display-refresh pacing.

Quality presets have been replaced by explicit advanced settings. Live capture exposes pyramid levels, the finest searched level, coarse and refinement radii, and predictor smoothness. Offline interpolation additionally exposes block sizing, grid spacing, subpixel mode, consistency and photometric thresholds, detail strength, and all algorithm feature toggles. Values are validated before processing starts.

The corresponding live command-line options are `--gpu-levels`, `--gpu-min-refine`, `--gpu-coarse-radius`, `--gpu-refine-radius`, and `--gpu-smoothness`. The optical flow engine is selected at runtime with `--flow-engine <ffx|msad>`: `ffx` uses the bundled AMD FidelityFX Optical Flow (default, with automatic MSAD fallback on dispatch failure) and `msad` forces the project's own coarse-to-fine block-matching shader. The active engine is shown at startup and reported per interval in the overlay status line. Offline controls use the algorithm options shown by `--help`.
Live source cadence is available with `--source-fps <auto|24|30|60>`.
Output multiplication is available with `--multiplier <2|3|4|max>`.

The command-line modes remain available for scripting and automation.

Motion Enhancer can run directly as a real-time GPU-accelerated video/game frame interpolator using **Windows Graphics Capture (WGC)** and a D3D12-first compute/presentation context:

```
[Target App Window (e.g. YouTube, Player, Game)]
              â”‚
              â–¼ (Zero-Copy GPU Capture via WGC)
    [Direct3D 11 VRAM Texture from WGC]
               â”‚
               â–¼
    [Shared capture texture + fence -> D3D12 queue]
               â”‚
               â–¼
    [DXC HLSL 7-Level Luminance Pyramid] (PyramidCS.hlsl)
              â”‚
              â–¼
   [Fixed 32x32 Block / 64x64 Support Candidate Search] (MotionSearchCS.hlsl)
              â”‚
              â–¼
   [Coarse-to-Fine Block Search + Spatial Filtering] (MotionSearchCS/FilterFlowCS.hlsl)
              â”‚
              â–¼
   [Timestamp-Driven Single-Source Motion Warp] (InterpolateCS.hlsl)
              â”‚
              â–¼
   [Paced Click-Through DXGI/DWM Presenter] (50-120+ FPS)
```

The real-time GPU path uses two native devices on the same adapter: a D3D12 device and direct command queue own all compute and presentation, while an independent D3D11 device serves Windows Graphics Capture''s `ID3D11Texture2D` contract. `D3D12Context` owns native D3D12 pyramid, flow, and output resources; SRV/UAV descriptors; a shared root signature; per-pass compute PSOs; command recording; explicit state barriers; and fence synchronization. Luminance, pyramid reduction, bidirectional MotionSearch/MSAD, FilterFlow, InterpolateCS, and PresentFrameCS are dispatched natively, and presentation targets a native D3D12 flip-model swap chain. Captured frames cross into D3D12 through NT-handle shared textures created on the D3D11 capture device (`MISC_SHARED | MISC_SHARED_NTHANDLE`, no keyed mutex), with two shared timeline fences ordering access: D3D11 signals a capture-ready fence after each copy and the D3D12 queue waits on it before consuming; a second shared fence orders any D3D11 consumption of native results (offline readback). Release configuration compiles the HLSL with DXC (`cs_5_0`) into `.cso` artifacts; the DXC artifact is consumed directly by the native D3D12 pipeline, while the D3D11 compatibility path logs a warning and recompiles DXBC with the Windows compiler when the artifact is DXIL. Forward and backward flow are estimated independently with the same coarse-to-fine hierarchy, using swapped reference and candidate frames. Native resources are retained for synthesis and flow is cached per source pair.

This is not AMD AFMF 2, which remains proprietary driver software. The default
build uses the bundled AMD FidelityFX Optical Flow implementation and falls
back to the project's native MSAD shader pipeline if FFX cannot initialize or
dispatch. The FFX path supplies a current-to-previous 8x8 `R16G16_SINT` vector field.
It is used as the backward field; the forward field is an explicit negated
approximation produced by the conversion shader. Normal capture has no FFX
flow readback or per-frame FFX logging. On the supplied 320x240 test pair with
+16,+8 px motion, the persistent-context offline run measured 32.95 dB PSNR
versus 25.36 dB for a naive blend.

### Queue and Synchronization

The renderer uses three application-owned source slots and a two-buffer flip-discard swapchain. WGC copies end with GPU event queries; only completed slots enter the ordered queue. The queue never grows beyond three frames and never overwrites a source texture referenced by cached flow.

Presentation requires both a high-resolution refresh deadline and an available swapchain frame-latency slot. Exactly one frame is submitted per combined signal. There are no sleeps, `DwmFlush` calls, immediate midpoint/source bursts, or retained back-buffer references.

For presentation timestamp $T$ bracketed by source timestamps $t_0$ and $t_1$, interpolation uses:

$$\alpha = \operatorname{clamp}\left(\frac{T-t_0}{t_1-t_0}, 0, 1\right)$$

The synthesis shader motion-warps both source frames according to $\alpha$ and blends them with forward/backward consistency weighting. It does not run a separate occlusion pass.

### Hotkeys in Overlay Mode
- `[Ctrl+Alt+F1]`: Toggle overlay visibility (Hide / Show)
- `[Ctrl+Alt+Esc]`: Exit overlay and stop capture

### Quick Start for Real-Time Capture

1. **List all open desktop windows:**
   ```powershell
   .\build\Release\motion_enhancer.exe --list-windows
   ```

2. **Launch real-time overlay on your application:**
   ```powershell
   # Capture browser / YouTube window
   .\build\Release\motion_enhancer.exe --capture-window "Brave"

   # Capture media player or game window
   .\build\Release\motion_enhancer.exe --capture-window "VLC"
   ```

---

## Example Commands

### 1. Real-Time GPU Window Capture & Interpolation
```powershell
.\build\Release\motion_enhancer.exe --capture-window "YouTube"
```

### 2. Offline Interpolation with Visualizations
```powershell
.\build\Release\motion_enhancer.exe frame0.png frame1.png interpolated_0.5.png `
    --time 0.5 `
    --levels 8 `
    --save-flow flow.png `
    --save-occ occlusion.png
```

### 3. Built-in Ground Truth Verification Benchmark
```powershell
.\build\Release\motion_enhancer.exe --test
```
Estimates motion across 8 pyramid levels, validates subpixel precision against ground truth ($\Delta x = +16\text{ px}, \Delta y = +8\text{ px}$), and outputs PSNR and execution time.
