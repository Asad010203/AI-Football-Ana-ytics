$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$pythonVersion = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ([version]$pythonVersion -lt [version]"3.10.11") {
    throw "PyInstaller requires Python 3.10.11 or newer for this build. Found Python $pythonVersion. Install Python 3.10.11+, recreate .venv, and install requirements.txt."
}

$models = @(
    "Modals\yolov11\yolo11x.pt",
    "Modals\yolov26\yolo26x.pt"
)
foreach ($model in $models) {
    if (-not (Test-Path $model)) {
        throw "Missing model file: $model"
    }
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name football-worker `
    --add-data "Modals;Modals" `
    --exclude-module tensorboard `
    --exclude-module IPython `
    --exclude-module jupyter `
    worker.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

Copy-Item "start_worker.bat" "dist\football-worker\start_worker.bat" -Force
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name updater `
    updater.py
if ($LASTEXITCODE -ne 0) {
    throw "Updater build failed."
}
Copy-Item "dist\updater.exe" "dist\football-worker\updater.exe" -Force
Write-Output "Worker created at $root\dist\football-worker\football-worker.exe"
