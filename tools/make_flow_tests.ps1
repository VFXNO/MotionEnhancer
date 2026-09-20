# Generates the synthetic frame pairs used to verify the GPU optical-flow
# engines with --gpu-offline, then runs both engines on each and prints the
# PSNR against ground truth.
#
#   .\tools\make_flow_tests.ps1 [-Exe build\Release\motion_enhancer.exe] [-Out flow_tests]
#
# Cases (all derived from test_frame0.png, produced by `--test`):
#   hx   global horizontal shift (-12 px)      -> must be found exactly, ~50 dB
#   vy   global vertical shift (-10 px)        -> must be found exactly, ~50 dB
#   cs   cel-shaded ellipse, small motion (6,4)
#   cel  cel-shaded ellipse, large motion (16,8): flat interior, textured
#        background; the case that exposed coarse-level lock-on and the need
#        for edge-weighted scoring. NOTE: this background is diagonally
#        translation-invariant (see below) and misleads edge blocks.
#   ncs / ncel  the same two motions over a noise-like background: the
#        representative cel cases
#   circle  the --test pair itself (textured circle, +16,+8; its 21 px texture
#        period makes -5 an aliased match for +16)
param(
    [string]$Exe = "build\Release\motion_enhancer.exe",
    [string]$Out = "flow_tests"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $Out | Out-Null
if (-not (Test-Path test_frame0.png)) { & $Exe --test | Out-Null }
$src = [System.Drawing.Bitmap]::FromFile((Resolve-Path test_frame0.png))

function Save24($bitmap, $path) {
    $copy = New-Object System.Drawing.Bitmap $bitmap.Width, $bitmap.Height, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($copy); $g.DrawImage($bitmap, 0, 0); $g.Dispose()
    $copy.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); $copy.Dispose()
}
function Crop($x, $y, $w, $h, $path) {
    $b = $src.Clone((New-Object System.Drawing.Rectangle $x, $y, $w, $h), $src.PixelFormat)
    Save24 $b $path; $b.Dispose()
}
# Noise-like background (random rectangles): no translation-invariant
# direction, unlike the sinusoidal --test background whose diagonal texture
# matches equally well along (1,-1) and misleads aperture-ambiguous blocks.
$noise = New-Object System.Drawing.Bitmap 320, 240, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$ng = [System.Drawing.Graphics]::FromImage($noise)
$ng.Clear([System.Drawing.Color]::FromArgb(120, 130, 140))
$rng = New-Object System.Random 11
for ($i = 0; $i -lt 900; $i++) {
    $col = [System.Drawing.Color]::FromArgb($rng.Next(60, 220), $rng.Next(60, 220), $rng.Next(60, 220))
    $ng.FillRectangle((New-Object System.Drawing.SolidBrush $col), $rng.Next(320), $rng.Next(240), $rng.Next(3, 24), $rng.Next(3, 24))
}
$ng.Dispose()

function Cel($dx, $dy, $path, $background) {
    $b = New-Object System.Drawing.Bitmap 320, 240, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.DrawImage($background, 0, 0, 320, 240)
    $g.SmoothingMode = 'AntiAlias'
    $g.FillEllipse((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(230, 80, 60))), 100 + $dx, 80 + $dy, 120, 80)
    $g.DrawEllipse((New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(40, 20, 20)), 3), 100 + $dx, 80 + $dy, 120, 80)
    $g.FillEllipse((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)), 130 + $dx, 100 + $dy, 18, 18)
    $g.Dispose()
    $b.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); $b.Dispose()
}

Crop 0 0 288 224 "$Out\hx0.png";  Crop 12 0 288 224 "$Out\hx1.png";  Crop 6 0 288 224 "$Out\hx_gt.png"
Crop 0 0 288 224 "$Out\vy0.png";  Crop 0 10 288 224 "$Out\vy1.png";  Crop 0 5 288 224 "$Out\vy_gt.png"
Cel -3 -2 "$Out\cs0.png" $src;  Cel 3 2 "$Out\cs1.png" $src;  Cel 0 0 "$Out\cs_gt.png" $src
Cel -8 -4 "$Out\cel0.png" $src; Cel 8 4 "$Out\cel1.png" $src; Cel 0 0 "$Out\cel_gt.png" $src
Cel -3 -2 "$Out\ncs0.png" $noise;  Cel 3 2 "$Out\ncs1.png" $noise;  Cel 0 0 "$Out\ncs_gt.png" $noise
Cel -8 -4 "$Out\ncel0.png" $noise; Cel 8 4 "$Out\ncel1.png" $noise; Cel 0 0 "$Out\ncel_gt.png" $noise
# Sequence: frames at (-8,-4), (0,0), (8,4); the timed pair is the last two
# with the temporal chain primed by the first pair (MSAD only).
Cel 4 2 "$Out\nseq_gt.png" $noise
Copy-Item test_frame0.png "$Out\circle0.png"; Copy-Item test_frame1.png "$Out\circle1.png"; Copy-Item test_groundtruth.png "$Out\circle_gt.png"
$src.Dispose(); $noise.Dispose()

foreach ($case in @('hx', 'vy', 'cs', 'cel', 'ncs', 'ncel', 'circle')) {
    foreach ($engine in @('msad', 'ffx')) {
        $line = & $Exe --gpu-offline "$Out\${case}0.png" "$Out\${case}1.png" "$Out\${case}_$engine.png" `
            --ground-truth "$Out\${case}_gt.png" --save-flow "$Out\${case}_flow_$engine.png" --flow-engine $engine 2>&1 |
            Out-String -Stream | Select-String "PSNR|Centre"
        "{0,-7}{1,-5} {2}" -f $case, $engine, (($line | ForEach-Object { $_.Line.Trim() }) -join ' | ')
    }
}
foreach ($primed in @($false, $true)) {
    $extra = if ($primed) { @('--gpu-offline-prev', "$Out\ncel0.png") } else { @() }
    $line = & $Exe --gpu-offline "$Out\ncel_gt.png" "$Out\ncel1.png" "$Out\nseq_msad.png" `
        --ground-truth "$Out\nseq_gt.png" --flow-engine msad @extra 2>&1 |
        Out-String -Stream | Select-String "PSNR|Centre"
    $label = if ($primed) { 'nseq+t' } else { 'nseq' }
    "{0,-7}{1,-5} {2}" -f $label, 'msad', (($line | ForEach-Object { $_.Line.Trim() }) -join ' | ')
}
