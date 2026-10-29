[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build-project',
    [string]$NsisExe = 'makensis.exe',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = Split-Path -Parent $PSScriptRoot
$build = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
$dist = Join-Path $root 'dist'
$work = Join-Path $root '.packaging'
$version = '1.0.0'
$name = "MotionEnhancer-$version-win64"
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt'))) {
    Invoke-Checked cmake @('-S', $root, '-B', $build, '-A', 'x64', '-DMOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW=ON')
} else {
    Invoke-Checked cmake @('-S', $root, '-B', $build)
}
if (-not $SkipBuild) { Invoke-Checked cmake @('--build', $build, '--config', 'Release', '--target', 'motion_enhancer', '--parallel') }
$cache = Get-Content -LiteralPath (Join-Path $build 'CMakeCache.txt') -Raw
if ($cache -notmatch 'MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW:BOOL=ON') { throw 'The complete package requires an FFX-enabled build.' }
$compiler = Get-Command $NsisExe -ErrorAction Stop
New-Item -ItemType Directory -Path $dist, $work -Force | Out-Null
$stage = Join-Path $work ('stage-' + [guid]::NewGuid().ToString('N'))
try {
    Invoke-Checked cmake @('--install', $build, '--config', 'Release', '--prefix', $stage)
    $required = @('motion_enhancer.exe', 'DirectML.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll',
        'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll',
        'models\ai_flow.onnx', 'models\ai_flow.json', 'models\midas_v21_small_256_fp16.onnx', 'models\neural_model.bin',
        'tools\ai-train\Setup-AITrainer.ps1', 'tools\ai-train\app.py', 'Setup-AITrainer.cmd')
    foreach ($file in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $stage $file) -PathType Leaf)) { throw "Missing package dependency: $file" }
    }
    if (@(Get-ChildItem -LiteralPath (Join-Path $stage 'shaders') -Filter '*.cso').Count -ne 29) { throw 'Expected all 29 compiled shaders.' }
    $files = @(Get-ChildItem -LiteralPath $stage -Recurse -File)
    if (@($files | Where-Object { $_.Extension -in @('.obj', '.pdb', '.log', '.lib', '.hlsl', '.hlsli', '.pyc') }).Count) {
        throw 'Build/debug-only files leaked into the package.'
    }
    $hashes = @($files | ForEach-Object {
        [pscustomobject]@{ file = $_.FullName.Substring($stage.Length + 1).Replace('\', '/');
                          sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    $hashes | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $stage 'payload-sha256.json') -Encoding UTF8
    # Exact uninstall list; models and any runtime-created files are preserved.
    $lines = @('; Generated from the clean install payload, not the development tree.')
    $files = @(Get-ChildItem -LiteralPath $stage -Recurse -File)
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($stage.Length + 1)
        if ($relative.StartsWith('models\')) { continue }
        $lines += 'Delete "$INSTDIR\' + $relative + '"'
    }
    foreach ($dir in (Get-ChildItem -LiteralPath $stage -Recurse -Directory | Sort-Object { $_.FullName.Length } -Descending)) {
        $lines += 'RMDir "$INSTDIR\' + $dir.FullName.Substring($stage.Length + 1) + '"'
    }
    $uninstallList = Join-Path $work 'uninstall-files.nsh'
    $lines | Set-Content -LiteralPath $uninstallList -Encoding UTF8
    $setup = Join-Path $dist "MotionEnhancer-$version-Setup-x64.exe"
    $sizeKB = [int][math]::Ceiling(($files | Measure-Object Length -Sum).Sum / 1KB)
    Invoke-Checked $compiler.Source @('/V2', "/DPAYLOAD=$stage", "/DOUTPUT=$setup", "/DVERSION=$version",
        "/DSIZE_KB=$sizeKB", "/DUNINSTALL_LIST=$uninstallList", (Join-Path $root 'packaging\MotionEnhancer.nsi'))
    $zip = Join-Path $dist ($name + '.zip')
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal -Force
    @($setup, $zip) | ForEach-Object {
        ((Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + (Split-Path -Leaf $_))
    } | Set-Content -LiteralPath (Join-Path $dist 'SHA256SUMS.txt') -Encoding ASCII
    Write-Host "Installer and portable ZIP ready in $dist"
} finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
