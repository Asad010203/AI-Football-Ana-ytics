# Football Player Detection (YOLO11)

Detect football players in videos with a pretrained YOLO11-x model. Class 0
(person) only.

## Install (Windows, Python 3.10, NVIDIA GPU)

Model files are expected at:

- `Modals/yolov11/yolo11x.pt`
- `Modals/yolov26/yolo26x.pt`

Torch/torchvision wheels are expected under `wheels/`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install "numpy<2" "opencv-python==4.10.0.84" "Pillow>=10.0"
.\.venv\Scripts\python.exe -m pip install "wheels\torch-2.6.0+cu124-cp310-cp310-win_amd64.whl" "wheels\torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl"
.\.venv\Scripts\python.exe -m pip install "ultralytics==8.4.142" "tqdm>=4.66" "click>=8.1" "norfair>=2.2"
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run locally

Drop a video into `input\`, then:

```powershell
.\.venv\Scripts\python.exe run.py --video input\match.mp4
```

Options: `--output-dir`, `--conf` (default 0.3), `--imgsz` (default 1280),
`--max-frames`.

Outputs go to `output\<video-stem>\annotated.mp4` and
`output\<video-stem>\results.json`.

## Local worker web interface

The worker runs the existing analytics pipeline on the computer where it is
started, so that computer's NVIDIA GPU and local video files are used:

```powershell
.\.venv\Scripts\python.exe worker.py
```

Open `http://127.0.0.1:8000`, select a video, and start analytics. Results are
written under `output\worker_jobs\` and are available from the browser when the
job completes.

The worker checks GitHub Releases through `/api/update-check`. Packaged
`football-worker.exe` releases can use the same endpoint for automatic update
delivery; development runs continue using the checked-out source.

## Build the client worker

After the model files are present locally, install the packaging dependency and
build a Windows folder distribution:

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\build_worker.ps1
```

Copy the complete `dist\football-worker\` folder to the client PC. Start it
with `start_worker.bat`. The folder must include the packaged model files; the
client does not need the Python source or a Python installation.
