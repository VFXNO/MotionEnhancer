# Optional AI blending mask

Archived validation summary. Generated images, raw logs and comparison JSON
mentioned below were removed during packaging cleanup; permanent self-tests
remain in the executable.

## Use

- Executable: `build-project\Release\motion_enhancer.exe` (Release, FFX enabled).
- Live page: select **AI pyramid (trained model)**, enable **AI: blending mask**, then start capture.
- CLI: add `--flow-engine ai --ai-blend-mask` to capture or GPU offline commands.
- Default remains off. Other engines and AI's DIS fallback do not consume the mask.

## Implementation

The network already produces a sigmoid midpoint preference: 1 prefers frame0, 0 prefers frame1. The optional path copies it entirely on the GPU into a separate network-resolution R32_FLOAT texture for each flow-ring entry. The existing flow/present fences protect its lifetime. No new inference or CPU readback is added to capture.

Synthesis binds it at t6 and weights geometrically valid, non-UI samples. At t=0.5 it samples the mask at the output pixel. At other times, each preference is transported to its predicted midpoint using that sample's motion, multiplied by the normal time weight, and faded toward neutral at the endpoints with 4t(1-t). A zero total weight retains the existing visibility result. Static-UI protection remains last. Both projected and trajectory-search candidates use the preference.

This is experimental: the mask was trained for midpoint synthesis using the network's fields, while runtime refines those fields with DIS. Neither arbitrary-time correctness nor improved occlusion quality is guaranteed.

## Checks

- Final build succeeded with FFX enabled; the GUI snapshot confirms the unchecked control is visible without overlapping other controls (`gui-live.png`).
- `--ai-blend-mask-self-test` passed: constant/extreme and neutral masks, disabled path, t=0/0.25/0.5/0.75/1, spatial-mask transport, UI protection, sole-visible-source fallback, single-field bypass.
- Installed-model checks passed: GPU output copy and mask range, distinct resources for successive pairs, earlier-mask retention after a reversed pair, enabling/disabling and ring reuse.
- Ordinary D3D12 and GPU-based validation runs passed. DirectML emits unsupported MetaCommand-parameter diagnostics while selecting implementations. The synthetic-only GPU-validation run produced only the validation startup notice, with no resource-state or descriptor diagnostics.
- Before/after 1080p AI output with the option off is pixel-identical at t=0.25, 0.5 and 0.75. DIS and FFX defaults are also pixel-identical (`comparison-results.json`).
- Enabling the option changes the AI intermediate frames; t=0 and t=1 remain exactly the original frames. DIS and FFX are unchanged even if the flag is supplied.
- Missing-model fallback with the flag supplied succeeds and is pixel-identical to DIS (`fallback-ai-mask.log`).
- A 5120x2880 AI replay with 10 requested levels and the mask enabled succeeded (`enabled_5k.log`).
- All nine `--projection-self-test` cases passed, as did `--d3d12-self-test`.

No end-to-end WGC/overlay capture session or representative-footage quality study was performed for this option. Changes remain uncommitted.

## Added mask debug view

Ctrl+Alt+F3 now cycles through view 6, **AI blend mask**, after AI depth and before off. It displays the raw pair-owned midpoint map (white = earlier frame, black = later, grey = neutral). A purple checkerboard distinguishes inactive/unavailable masks and pass-through frames from a valid neutral mask. Existing frame-kind markers and six mode dots remain visible. The raw map is deliberately independent of timestamp, geometric visibility and UI protection; it is not the final source-weight visualization.

The extended GPU self-test passes grayscale extremes/neutral, timestamps, spatial maps, UI independence, inactive and single-field bypass cases, pass-through rendering and frame-kind markers. Synthetic GPU-based validation passes with only the startup notice (`debug-view-gpu-validation.log`). Installed-model/ring checks also pass.

Actual 1080p AI debug renders have grayscale values 67..178 on the test pair, retain all six mode dots, and are pixel-identical at t=0.25 and 0.75 as expected for a raw midpoint map. Disabling the mask produces exactly the two purple checkerboard colours. Normal AI output with the option on/off, DIS and FFX remains pixel-identical to the previous build (`debug-view-results.json`). The nine-case projection regression and original D3D12 self-test also pass. The final FFX-enabled Release executable and staged shaders were rebuilt successfully.
