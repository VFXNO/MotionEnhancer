[CmdletBinding(SupportsShouldProcess = $true)]
param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$summary = @()

# Explicit obsolete build trees only. Never match SDKs, backups or user data.
$folders = @('build', 'build-d3d12', 'build-ffx', 'build-ffx-local', 'build-ffx-test',
    'build-local', 'build-vendored-ffx', 'build-project\blend-mask-tests',
    'build-project\projection-fix', 'build-project\pyramid-audit', '.packaging',
    'tools\ai-train\__pycache__', 'tools\ai-train\mefi\__pycache__')
foreach ($folder in @('blend-mask-tests', 'projection-fix', 'pyramid-audit')) {
    $report = Join-Path $root "build-project\$folder\REPORT.md"
    if (Test-Path -LiteralPath $report) {
        $archive = Join-Path $root "docs\validation\$folder.md"
        if (-not (Test-Path -LiteralPath $archive)) { throw "Preserve $report as $archive before cleaning." }
    }
}
foreach ($relative in $folders) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path)) { continue }
    # Refuse to traverse junctions/symlinks outside the named generated tree.
    $links = @(Get-ChildItem -LiteralPath $path -Recurse -Force | Where-Object {
        $_.Attributes -band [IO.FileAttributes]::ReparsePoint
    })
    if ((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint -or $links.Count) {
        throw "Refusing to recursively remove reparse points in $relative"
    }
    $files = @(Get-ChildItem -LiteralPath $path -File -Recurse -Force)
    $summary += [pscustomobject]@{ Group = $relative; Files = $files.Count; Bytes = ($files | Measure-Object Length -Sum).Sum }
    if ($PSCmdlet.ShouldProcess($path, 'Remove obsolete/generated build directory')) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
$rootOutputs = @(Get-ChildItem -LiteralPath $root -File | Where-Object {
    $_.Name -match '^(gpu_|ffx_|output_|flow_frac_).*\.(png|txt|log)$|^out\.png$|^shared_probe\.obj$|^x_f\.png\.txt$|^test_(flow|interpolated|occlusion)\.png$'
})
$flowOutputs = @(Get-ChildItem -LiteralPath (Join-Path $root 'flow_tests') -File -Recurse | Where-Object {
    $_.Name -match '(sad|ffx|neural|dis|neu|fx)\.png(\.txt)?$|\.png\.txt$|_(out|ui)\d+\.png$|_[wx]\.png$|^dbg\d*\.png$|^(det|dt|d)\d+\.(png|txt)$|^(lv|fl_(sad|neural)|st_[fghi]\d*|tf_sad)\.(png|txt)$|^st_(o|out)\.png$|^(bigf|csf|qf|rsweep_f|L0)\.png$'
})
$buildOutputs = @(foreach ($file in @('ffx-check.png', 'pyramid_audit.py', 'pyramid_audit_extra.py')) {
    $path = Join-Path $root "build-project\$file"
    if (Test-Path -LiteralPath $path) { Get-Item -LiteralPath $path }
})
foreach ($group in @(@{ Name = 'root diagnostics'; Files = $rootOutputs },
    @{ Name = 'flow-test outputs (inputs retained)'; Files = $flowOutputs },
    @{ Name = 'obsolete one-off audit files'; Files = $buildOutputs })) {
    $files = @($group.Files)
    $summary += [pscustomobject]@{ Group = $group.Name; Files = $files.Count; Bytes = ($files | Measure-Object Length -Sum).Sum }
    if ($files.Count -and $PSCmdlet.ShouldProcess("$($files.Count) files: $($group.Name)", 'Remove generated diagnostic files')) {
        $files | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    }
}
$summary | Select-Object Group, Files, @{ Name = 'MB'; Expression = { [math]::Round($_.Bytes / 1MB, 1) } } | Format-Table -AutoSize
Write-Output ('Total selected: {0:N1} MB' -f (($summary | Measure-Object Bytes -Sum).Sum / 1MB))
