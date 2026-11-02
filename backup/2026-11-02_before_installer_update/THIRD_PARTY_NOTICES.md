# Third-Party Notices

## AMD FidelityFX SDK

The repository vendors AMD FidelityFX SDK v1.1.4 under
`third_party/FidelityFX-SDK-v1.1.4`, including the standalone Optical Flow
source, DirectX 12 backend, and `FidelityFX_SC.exe` shader compiler. The
project builds the generated Optical Flow shader permutations locally.

Copyright (C) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK

## ONNX Runtime and DirectML

The Windows distribution includes ONNX Runtime 1.24.4 (DirectML execution
provider) and DirectML 1.15.4. Their original Microsoft licenses and dependency
notices are installed under `licenses/ORT_*` and `licenses/DirectML_*`.
Development copies remain under `third_party/ai-depth`.

## MiDaS depth model

`models/midas_v21_small_256_fp16.onnx` is the FP16 deployment conversion of
MiDaS v2.1 small. Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab),
MIT license: `licenses/MiDaS-LICENSE.txt`.
Source: https://github.com/isl-org/MiDaS

## stb image libraries

stb_image and stb_image_write are compiled into the program under their MIT
license. Copyright (c) 2017 Sean Barrett; see `licenses/stb-LICENSE.txt`.
Source: https://github.com/nothings/stb

## Microsoft C++ and OpenMP redistributables

The Windows package includes the release x64 MSVC runtime and OpenMP DLLs
from the Visual Studio redistributable directory. These are Microsoft
Distributable Code, not the Visual Studio development SDK. Microsoft license
terms apply: https://visualstudio.microsoft.com/license-terms/
Windows, Direct3D, Media Foundation, and graphics drivers remain OS/driver
prerequisites and are not copied from System32 or redistributed by this project.

## Installer and optional trainer downloads

The installer is generated with NSIS, which permits redistribution of generated
installers: https://nsis.sourceforge.io/License
The optional trainer setup downloads (rather than bundles) Python, PyTorch,
torchvision, ROCm/CUDA dependencies, PySide6, NumPy, OpenCV, ONNX/ONNX Script,
and FFmpeg/FFprobe. Their upstream licenses and GPU/driver requirements apply.
FFmpeg builds are obtained from https://www.gyan.dev/ffmpeg/builds/ and retain
their upstream license files. Qt/PySide6 and FFmpeg are not statically linked
into the Motion Enhancer executable.

## Optional UI cooperative-matrix runtime

When built with `MOTION_ENHANCER_ENABLE_UI_MATRIX=ON`, the executable links
ncnn 20261020 (Tencent, BSD-3-Clause) and its bundled glslang compiler.
Their original notices are retained as `licenses/UI-Matrix-ncnn.txt` and
`licenses/UI-Matrix-glslang.txt`. Source is obtained by the hash-pinned
`tools/ui-detect/setup_matrix.py`, not by modifying the existing graphics SDK.
Vulkan and the AMD driver remain system prerequisites, not bundled DLLs.
Source: https://github.com/Tencent/ncnn/tree/20261020

The separately downloaded OmniParser V1 icon/UI detector weights (including
their matrix-backend conversion) use **AGPL-3.0**, not this project's MIT or
ncnn's BSD licence. Their licence and provenance remain alongside the optional
model. Model files are not implicitly added to packages; redistribution
obligations still require review before shipping.
