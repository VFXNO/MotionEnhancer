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
├── CMakeLists.txt              # CMake build script (C++17, OpenMP, MSVC /O2)
├── include/
│   ├── Image.h                # Multi-channel float image container & bilinear sampler
│   ├── Pyramid.h              # 8-level Gaussian pyramid builder
│   ├── MotionVector.h         # Flow field, color-guided median filter, flow visualizer
│   ├── ZNCCMatcher.h          # Coarse-to-fine ZNCC solver with EPZS & sub-pel search
│   └── FrameInterpolator.h   # Bidirectional flow, occlusion detection & artifact refinement
├── src/
│   ├── Image.cpp
│   ├── Pyramid.cpp
│   ├── ZNCCMatcher.cpp
│   ├── FrameInterpolator.cpp
│   └── main.cpp               # CLI entry point and synthetic benchmark
└── third_party/
    ├── stb_image.h            # Single-header image loader
    └── stb_image_write.h      # Single-header image saver
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

## Real-Time GPU Shader Overlay (WGC + DirectCompute)

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

The corresponding live command-line options are `--gpu-levels`, `--gpu-min-refine`, `--gpu-coarse-radius`, `--gpu-refine-radius`, and `--gpu-smoothness`. Offline controls use the algorithm options shown by `--help`.
Live source cadence is available with `--source-fps <auto|24|30|60>`.
Output multiplication is available with `--multiplier <2|3|4|max>`.

The command-line modes remain available for scripting and automation.

Motion Enhancer can run directly as a real-time GPU-accelerated video/game frame interpolator using **Windows Graphics Capture (WGC)** and **DirectX 11 HLSL Compute Shaders**:

```
[Target App Window (e.g. YouTube, Player, Game)]
              │
              ▼ (Zero-Copy GPU Capture via WGC)
   [Direct3D 11 VRAM Texture]
              │
              ▼
   [HLSL 7-Level Luminance Pyramid] (PyramidCS.hlsl)
              │
              ▼
   [Bidirectional 16x16 Block / Candidate Search] (MotionSearchCS.hlsl)
              │
              ▼
   [Coarse-to-Fine Block Search + Spatial Filtering] (MotionSearchCS/FilterFlowCS.hlsl)
              │
              ▼
   [Timestamp-Driven Single-Source Motion Warp] (InterpolateCS.hlsl)
              │
              ▼
   [Paced Click-Through DXGI/DWM Presenter] (50-120+ FPS)
```

The real-time GPU path is a Direct3D 11 / Shader Model 5 adaptation of AMD FidelityFX Optical Flow v5. It estimates one `R32G32_FLOAT` vector per non-overlapping 16x16 luminance block, evaluates every block over a fixed 16x16 matching support region using Rec.709 luma ZNCC, performs coarse-to-fine candidate search by consuming the parent block vector directly in `MotionSearchCS.hlsl`, applies spatial flow filtering, and retains the highest-scoring candidate without content-specific rejection gates. Forward and backward flow are estimated independently with the same coarse-to-fine hierarchy, using swapped reference and candidate frames. Flow is cached per source pair and reused for every presentation timestamp. Runtime statistics report asynchronous D3D11 timestamp measurements as `GPU Pipeline`.

This is not AMD AFMF 2, which remains proprietary driver software. It ports the publicly documented FidelityFX optical-flow core while retaining this application's WGC capture, single-source synthesis, and D3D11 presentation architecture. See `THIRD_PARTY_NOTICES.md` for attribution.

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
