# High-resolution projection fix

## Fixed

The shared two-field projection no longer assumes source indices fit in 21
bits. `ProjectionIndexBits()` in `shaders/Synthesis.hlsli` derives the required
width from the actual flow grid; both `PackProjectionKey()` and the reader
use that layout. DCI 4K and 5K use 22 bits, 8K uses 23. Captures that already
fit keep their previous 21-bit encoding. This also works with coarser flow
grids, whose counts differ from the 2x2 grid.

The priority quantum scales with the remaining priority bits to retain roughly
the previous priority range. The all-ones empty sentinel stays reserved. There
are no extra textures, GPU passes, or 64-bit atomic requirements.

## Regression proof

New permanent command:

```powershell
.\build-project\Release\motion_enhancer.exe --projection-self-test
```

It runs the production projection and interpolation shaders on exact fields,
checks every projected source index, priority and empty cell, and checks the
motion decoded by interpolation at selected rows, including the final row.
The bottom half moves differently from the top so an aliased source cannot
hide behind a uniform field. Before the shader fix, this command passed
1440p/UHD 4K and failed decoded motion on the final row of 4096x2160.

After the fix, all nine cases pass:

- 2560x1440, 3840x2160, 4096x2048 (exact old index capacity), 4096x2160,
  5120x2880, 4097x2049 and 7680x4320, at a 2x2 flow grid.
- 5120x2880 with an 8x8 flow grid.
- 5120x2880 with inconsistent fields that saturate the priority.

Also passed with CPU D3D12 debug validation and GPU-based validation. The
dedicated projection test emitted no validation errors (only the GPU-validation
startup notice). Unlike the original shared-output self-test, it renders to
an ordinary D3D12 texture; this does not resolve the previously reported
shared-output resource-layout diagnostic.

## Replay comparisons

Same rescaled `big48` source pair, settings 10, before versus after:

| Resolution | Engine | Before | After |
| --- | --- | ---: | ---: |
| 2560x1440 | DIS | 26.20 dB | 26.20 dB, identical pixels |
| 2560x1440 | default FFX | 24.37 dB | 24.37 dB, identical pixels |
| 2560x1440 | FFX occlusion | 23.78 dB | 23.78 dB, identical pixels |
| 3840x2160 | DIS | 28.64 dB | 28.64 dB, identical pixels |
| 3840x2160 | default FFX | 21.70 dB | 21.70 dB, identical pixels |
| 3840x2160 | FFX occlusion | 21.66 dB | 21.66 dB, identical pixels |
| 5120x2880 | DIS | 26.87 dB | 27.14 dB |
| 5120x2880 | default FFX | 23.99 dB | 23.99 dB, identical pixels |
| 5120x2880 | FFX occlusion | 23.25 dB | 23.57 dB |

Default single-field FFX bypasses projection, so its unchanged result is
expected. Additional SAD, neural (with the repository's MNFN model explicitly
selected), and AI/ONNX replays completed at 5K, as did `--d3d12-self-test`.

## Build and limits

Rebuilt `build-project/Release/motion_enhancer.exe` with FFX enabled. Verified
the staged projection/interpolation shader binaries match the compiled ones.

This fixes the confirmed overflow, not all possible interpolation artifacts.
UHD 4K already fit the old encoding, so its remaining artifact cause is not
addressed here. No long-running WGC/overlay capture test was performed.
