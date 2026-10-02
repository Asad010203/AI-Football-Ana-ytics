$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Create the .venv and install requirements.txt before building."
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
    --exclude-module torch.distributed `
    --exclude-module torch.utils.tensorboard `
    --exclude-module IPython `
    --exclude-module jupyter `
    worker.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

Write-Output "Worker created at $root\dist\football-worker\football-worker.exe"
