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
              |  WGC (D3D11 surface, timestamp)
              v
    [Capture thread: copy -> slot ring, luma diff, cadence classify]
              |  shared NT-handle texture + capture-ready fence
              v
    [Flow queue: FFX Optical Flow  (or MSAD pyramid search + filter)]
              |  flow ring entry + flow fence
              v
    [Frame timeline: media clock = display time - offset - latency budget]
              |  pair (t0, t1), alpha
              v
    [Present queue: InterpolateCS -> flip swap chain, Present(1)]
              |  paced by the DWM compositor clock (one frame per refresh)
              v
    [Click-through overlay window]
```

The real-time path is D3D12-only for compute and presentation; a D3D11 device on the same adapter exists solely to satisfy Windows Graphics Capture's `ID3D11Texture2D` contract. It is organised as five components:

| Component | Thread | Role |
|---|---|---|
| `GraphicsDevice` | – | D3D12 device, a **present queue** and a separate **flow queue**, the D3D11 capture device, and the shared NT-handle textures/fences that order D3D11 → D3D12 access. |
| `CaptureEngine` | capture (producer) | WGC callback only queues frames. The producer thread copies each into a slot ring, runs a sparse GPU luma diff against the newest frame (rejects identical repaints, flags scene cuts), classifies the frame against the source cadence, and submits its flow. |
| `FlowEngine` | capture | AMD FidelityFX Optical Flow (MSAD graph as fallback) into a 16-entry flow ring on the flow queue. Nothing here ever blocks the render thread. |
| `FrameTimeline` | shared | Cadence lock, timestamp regularisation, the media clock, and the adaptive latency budget. |
| `Presenter` | render | 3-buffer flip-model swap chain, `Present(1)`, one command list per refresh, GPU-side fence waits only. |

Release builds compile the HLSL with DXC (`cs_5_0`) into `.cso` artifacts loaded by the D3D12 pipelines; the D3D11 luma-diff kernel is a tiny embedded shader compiled at runtime.

This is not AMD AFMF 2, which remains proprietary driver software. The default
build uses the bundled AMD FidelityFX Optical Flow implementation and falls
back to the project's native MSAD shader pipeline if FFX cannot initialize or
dispatch. The FFX path supplies a current-to-previous 8x8 `R16G16_SINT` vector field.
It is used as the backward field; the forward field is an explicit negated
approximation produced by the conversion shader. Normal capture has no FFX
flow readback or per-frame FFX logging.

### Native MSAD matcher (`--flow-engine msad`)

`MotionSearchCS.hlsl` is a coarse-to-fine block matcher built the way AMD's FidelityFX Optical Flow kernel is: 8×8 blocks on an 8-level luma pyramid, `msad4`, packed luma, ±8 search at every level. One 8×8 thread group per block (SM 6.0, wave intrinsics for every reduction).

- **Pyramids per frame, not per pair**: each captured frame's luma pyramid and its packed 4-bytes-per-texel copy (`PackLumaCS`) are built once and cached in its capture slot, so a running stream costs one pyramid per new frame — FFX keeps the previous frame's pyramid the same way.
- **msad4**: one instruction scores a 4-byte reference word against the four alignments of an 8-byte source pair, i.e. four horizontal candidates at once; a block against four candidates is 16 msad4. The candidate window is staged in shared memory from the packed texture, word-aligned. (msad4 is native on AMD hardware; NVIDIA drivers emulate it with byte ops, so there it is merely not slower.)
- **Edge weighting for free**: msad4 ignores reference bytes that are 0, so the reference is split into an *edge* word (samples above the block's adaptive gradient threshold, full weight) and a *smooth* word (the rest, weight 0.1). A flat-shaded (cel) character is matched by its outline, not by the background behind it, while the low-weight remainder breaks ties along straight edges.
- **Temporal prediction**: the filtered flow of every level is double-buffered, and the previous pair's flow *at the same level* is a candidate for every block (valid while the frame chain is unbroken). With the search disabled entirely, the temporal predictor alone reaches 33.6 dB on the 3-frame sequence test below.
- **Candidates**: parent predictor, temporal predictor and the eight neighbouring parents (EPZS) are scored once each; a ±radius window is searched around the best, plus a window around zero when zero is outside it. Zero only wins when it is *strictly* better by one luma level — ties go to motion, never to a stall. Blocks with too few edge samples pick among the candidates only; `FilterFlowCS` then applies a 3×3 consensus.
- **Radius**: `--gpu-coarse-radius` above the finest level and `--gpu-refine-radius` at it (both default 8). A wide window at every level means a wrong vector from a coarse level is recoverable one level down — the coarse levels only extend the reach.

`tools\make_flow_tests.ps1` regenerates the synthetic verification pairs and runs both engines. Current results (PSNR vs. ground truth, naive blend in brackets); `n*` cases use a noise-like background, the others the sinusoidal `--test` background, which is diagonally translation-invariant and misleads aperture-ambiguous blocks:

| Case | MSAD | FFX |
|---|---|---|
| global shift −12 px | 48.4 dB | 54.6 dB (27.3) |
| global shift −10 px | 51.0 dB | 56.6 dB (27.5) |
| cel ellipse, motion (6,4) | **39.3 dB** | 34.9 dB (27.1) |
| cel ellipse, motion (16,8) | **30.1 dB** | 27.3 dB (24.2) |
| noise bg, cel ellipse (6,4) | **35.0 dB** | 33.6 dB (26.6) |
| noise bg, cel ellipse (16,8) | 28.3 dB | 28.2 dB (23.9) |
| textured circle, motion (16,8) | **34.5 dB** | 33.0 dB (25.4) |
| 3-frame sequence, temporal chain primed | 34.0 dB | – (25.8) |

At 1080p on an RTX 3050 Ti laptop (best of 40 back-to-back submissions, `MOTION_ENHANCER_BENCH_ITERS=40`) the MSAD graph costs 5.7 ms per source pair on a worst-case noise image, FFX 2.5 ms; both run on the flow queue and never touch the presentation deadline. `--gpu-offline-prev <frame>` primes the temporal chain with the pair before the timed one; `MOTION_ENHANCER_MSAD_DEBUG=1|3|5|6` makes the matcher report edge counts, the parent level's flow, or skip its window/centre stages for timing.

### Pacing and Synchronization

**Output clock.** The render loop is woken by the DWM compositor clock (`DCompositionWaitForCompositorClock`), i.e. once per vsync; the swap chain's frame-latency object is used only as non-blocking back-pressure. A phase-locked loop on the wake times predicts the display time of the frame being rendered, so consecutive frames are exactly one refresh apart even when the wake-up itself jitters, and it resyncs only on a genuinely missed refresh.

**Media clock.** Source frames carry WGC timestamps $t_i$. The timeline tracks the minimum capture offset $O$ between wall clock and $t$, and maps a predicted display time $T$ to media time

$$m = T - O - D$$

where $D$ is a latency budget that starts at one source period plus two refreshes and adapts: it grows whenever a needed pair was not ready (starvation) and decays while pairs have been ready with margin. For the pair $t_0 \le m < t_1$ the blend factor is $\alpha = (m - t_0)/(t_1 - t_0)$. Because $m$ advances with the wall clock, a missed refresh skips media time instead of slowing it, and motion always plays at 1x.

**Source cadence.** The period is estimated from every non-duplicate WGC arrival (mean of the middle half of the last 16 intervals) and locks when three quarters of them agree. Once locked, a frame arriving far off the cadence grid is held for 0.7 periods: if a newer frame supersedes it (a UI repaint between video frames) it is dropped, otherwise it is accepted late and the grid resyncs. Two frames closer than 0.3 periods are treated as one (the later replaces the earlier). Accepted timestamps are regularised onto the grid with a slow phase correction, which removes the ±8 ms jitter of a 24 fps video repainted at 60 Hz. Identical repaints (fewer than 0.05 % of sampled pixels changed) never enter the timeline; pairs classified as scene cuts are presented as a hard switch rather than interpolated.

**GPU ordering.** All ordering is on the GPU: the present queue waits on the capture-ready fence for both source frames and on the flow-queue fence for the pair, then interpolates. Shared capture textures are simultaneous-access resources read through implicit COMMON-state promotion, so no transition barriers are recorded on them from either queue. Slots and flow-ring entries are recycled only after the fences of every queue that read them have passed.

The synthesis shader treats flow as one vector per 8×8 block: the default warp for each pixel is the bilinear blend of the four nearest block vectors of both fields. Around a moving object that blend ramps between object and background motion over an ~8 px band, dragging background along the object and smearing edges. So each pixel also scores the raw vectors of its surrounding blocks (both fields, put in frame0→frame1 sense) with a symmetric photometric check — the pixels' trajectory $p - v\alpha$ in frame0 vs. $p + v(1-\alpha)$ in frame1 must match over a 5-tap neighbourhood — and switches off the bilinear default only when a block vector beats it by a clear margin (~4 luma levels per tap). Both source samples are then warped by the same chosen vector and blended by $\alpha$, so the boundary follows the silhouette instead of the block grid; ties and flat/noisy areas stay on the bilinear default (no flicker), and occlusions keep the best available match. There is no separate occlusion pass.

Diagnostics: `MOTION_ENHANCER_PACING=1` logs every presented frame (mode, predicted display time, media time, pair, alpha); `MOTION_ENHANCER_D3D_DEBUG=1` enables the D3D12/D3D11 debug layers and prints validation messages with the 2-second status line; `MOTION_ENHANCER_PIXEL_SELECT=0` disables the per-pixel vector selection and restores the plain bilinear warp blend.

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
