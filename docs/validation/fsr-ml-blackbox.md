# FSR ML frame generation 4.0.2: black-box test (2026-11-23)

AMD's model ships only as a signed DLL whose licence forbids reverse
engineering, so it was probed through its normal API only, offline, with the
inputs changed one at a time. Provider in use: `4.0.2 (id 0xf600000001000002)`
on an RX 9070 XT. Quality only (the GPU is shared with training; no timings).

New test-only option: `--fsr-mv-scale <0-2>` multiplies the motion vectors
given to FSR (`--fsr-fg-offline`, `--fsr-realtime-test`). Live never sets it.

## Motion vectors (PSNR vs the true middle frame, DIS flow)

| Vectors          | big24 (one pair) | scroll (11 pairs) | stop (12 pairs) |
|------------------|------------------|-------------------|-----------------|
| ours (x1)        | 28.87 dB         | 32.95 dB          | 60.60 dB        |
| none (x0)        | 19.79 dB         | 21.37 dB          | 54.58 dB        |
| 20% short (x0.8) | 23.64 dB         | 29.38 dB          | 56.76 dB        |
| 20% long (x1.2)  | 23.47 dB         | -                 | -               |
| naive blend      | 19.47 dB         | 22.15 dB          | 54.44 dB        |

With no vectors the model gives roughly a blend: it does not recover the
motion from the images by itself. A 20% vector error costs 3.5-5 dB. The
middle frame is only as good as our flow.

## Depth and camera (sequences, DIS flow)

| Setting                         | scroll   | stop     |
|---------------------------------|----------|----------|
| flat depth, fixed camera        | 32.95 dB | 60.60 dB |
| AI depth (span 0.25)            | 32.85 dB | 60.37 dB |
| camera from flow (live, lagged) | 32.35 dB | 59.41 dB |
| camera from flow (exact)        | 32.27 dB | 60.46 dB |

Neither AI depth nor a camera estimate helps on these sequences (both
within -1.2..-0.1 dB). They are flat screen content; a 3D game pan may differ.

## Against this program's own interpolation (same pairs)

| Flow engine | FSR, scroll | own synthesis, scroll | own synthesis, stop |
|-------------|-------------|-----------------------|---------------------|
| DIS         | 32.95 dB    | 35.05 dB              | 65.10 dB            |
| AI pyramid  | 34.39 dB    | 40.62 dB              | -                   |

Better flow raises FSR too (+1.4 dB with the AI engine), but our own
synthesis stays ahead by 2-6 dB with the same vectors.
