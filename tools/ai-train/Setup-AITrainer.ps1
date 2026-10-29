[CmdletBinding()]
param(
    [ValidateSet('auto', 'amd', 'cuda', 'cpu')][string]$Backend = 'auto',
    [ValidatePattern('^gfx[0-9a-f]+$')][string]$AmdTarget = 'gfx1201',
    [string]$PythonExecutable,
    [switch]$CheckOnly,
    [switch]$Pause,
    [switch]$Launch
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$homeFolder = $PSScriptRoot
$venvPython = Join-Path $homeFolder '.venv\Scripts\python.exe'
$cache = Join-Path $env:LOCALAPPDATA 'MotionEnhancer\downloads'

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}
function Get-Download([string]$Url, [string]$Destination) {
    Write-Host "Downloading $Url"
    $oldProgress = $ProgressPreference
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile ($Destination + '.partial') -UseBasicParsing
        Move-Item -LiteralPath ($Destination + '.partial') -Destination $Destination -Force
    } finally { $ProgressPreference = $oldProgress }
}
function Find-Python {
    if ($PythonExecutable) { return (Resolve-Path -LiteralPath $PythonExecutable).Path }
    $privatePython = Join-Path $homeFolder 'python\python.exe'
    if (Test-Path -LiteralPath $privatePython) { return $privatePython }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = & $launcher.Source -3.14 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { return [string]($found | Select-Object -Last 1) }
    }
    return $null
}

$exitCode = 0
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'The trainer requires Windows x64.' }
    if (-not $CheckOnly) {
        New-Item -ItemType Directory -Path $cache -Force | Out-Null
        if ($Backend -eq 'auto') {
            $gpus = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ', '
            if ($gpus -match 'Radeon.*RX\s*9070') { $Backend = 'amd'; $AmdTarget = 'gfx1201' }
            elseif ($gpus -match 'NVIDIA') { $Backend = 'cuda' }
            else {
                $Backend = 'cpu'
                Write-Warning "No auto-supported training GPU detected ($gpus). Using CPU; see README for other AMD targets."
            }
        }
        Write-Host "Trainer backend: $Backend $(if ($Backend -eq 'amd') { $AmdTarget })"
        Write-Host 'This setup downloads several GB. Keep the window open; interruption can be retried.'
        if (-not (Test-Path -LiteralPath $venvPython)) {
            $python = Find-Python
            if (-not $python) {
                $version = '3.14.7'
                $installer = Join-Path $cache "python-$version-amd64.exe"
                if (-not (Test-Path -LiteralPath $installer)) {
                    Get-Download "https://www.python.org/ftp/python/$version/python-$version-amd64.exe" $installer
                }
                $signature = Get-AuthenticodeSignature -LiteralPath $installer
                if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                    throw 'Python installer signature is not valid. Remove the cached download and retry.'
                }
                $privateFolder = Join-Path $homeFolder 'python'
                $p = Start-Process -FilePath $installer -ArgumentList @('/quiet', 'InstallAllUsers=0',
                    ('TargetDir="' + $privateFolder + '"'), 'PrependPath=0', 'Include_launcher=0',
                    'Include_test=0', 'Include_doc=0', 'Include_pip=1', 'Shortcuts=0') -Wait -PassThru
                if ($p.ExitCode -notin @(0, 3010)) { throw "Python installer failed: $($p.ExitCode)" }
                $python = Join-Path $privateFolder 'python.exe'
            }
            Invoke-Checked $python @('-c', "import sys,struct; assert sys.version_info[:2] == (3,14) and struct.calcsize('P')==8, 'Python 3.14 x64 required'")
            Invoke-Checked $python @('-m', 'venv', (Join-Path $homeFolder '.venv'))
        }
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
        switch ($Backend) {
            'amd' {
                Invoke-Checked $venvPython @('-m', 'pip', 'install', '--index-url', 'https://stable.repo.amd.com/rocm/whl-next/',
                    "torch[device-$AmdTarget]==2.14.0+rocm10.1.0", "torchvision[device-$AmdTarget]==0.29.0a0+rocm10.1.0")
            }
            'cuda' { Invoke-Checked $venvPython @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu130') }
            'cpu' { Invoke-Checked $venvPython @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu') }
        }
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', (Join-Path $homeFolder 'requirements.txt'))

        $ffmpegBin = Join-Path $homeFolder 'ffmpeg\bin'
        if (Test-Path -LiteralPath (Join-Path $ffmpegBin 'ffmpeg.exe')) { $env:PATH = "$ffmpegBin;$env:PATH" }
        if (-not (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -or
            -not (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
            $url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
            $archive = Join-Path $cache 'ffmpeg-release-essentials.zip'
            $checksum = Join-Path $cache 'ffmpeg-release-essentials.zip.sha256'
            Get-Download ($url + '.sha256') $checksum
            Get-Download $url $archive
            $expected = [regex]::Match((Get-Content -LiteralPath $checksum -Raw), '(?i)\b[0-9a-f]{64}\b').Value
            if (-not $expected -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
                throw 'FFmpeg download checksum mismatch. Retry setup.'
            }
            $unpack = Join-Path $cache ('ffmpeg-' + [guid]::NewGuid().ToString('N'))
            Expand-Archive -LiteralPath $archive -DestinationPath $unpack
            $root = @(Get-ChildItem -LiteralPath $unpack -Directory)
            if ($root.Count -ne 1) { throw 'Unexpected FFmpeg archive layout.' }
            New-Item -ItemType Directory -Path (Join-Path $homeFolder 'ffmpeg') -Force | Out-Null
            Copy-Item -Path (Join-Path $root[0].FullName '*') -Destination (Join-Path $homeFolder 'ffmpeg') -Recurse -Force
            $env:PATH = "$ffmpegBin;$env:PATH"
            # Only the uniquely named extraction directory created above.
            Remove-Item -LiteralPath $unpack -Recurse -Force
        }
    }
    if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Trainer environment not installed. Run setup without -CheckOnly.' }
    $ffmpegBin = Join-Path $homeFolder 'ffmpeg\bin'
    if (Test-Path -LiteralPath $ffmpegBin) { $env:PATH = "$ffmpegBin;$env:PATH" }
    Invoke-Checked $venvPython @('-c', "import torch,torchvision,cv2,onnx,onnxscript,numpy,PySide6; print('Trainer imports OK; torch', torch.__version__); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('Tensor check:', (torch.ones(4,device='cuda' if torch.cuda.is_available() else 'cpu')+1).sum().item())")
    Invoke-Checked 'ffmpeg.exe' @('-version')
    Invoke-Checked 'ffprobe.exe' @('-version')
    Write-Host 'AI trainer setup/check finished successfully.' -ForegroundColor Green
    if ($Launch) {
        Start-Process -FilePath (Join-Path $homeFolder '.venv\Scripts\pythonw.exe') `
            -ArgumentList ('"' + (Join-Path $homeFolder 'app.py') + '"') -WorkingDirectory $homeFolder
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'Fix the problem and run setup again. Your datasets and runs have not been deleted.'
    $exitCode = 1
} finally {
    if ($Pause) { Read-Host 'Press Enter to close' | Out-Null }
}
exit $exitCode
