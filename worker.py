"""Local browser worker for client-side football video analytics."""

from __future__ import annotations

import cgi
import cv2
import html
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))
APP_VERSION = "0.1.13"
GITHUB_REPOSITORY = "Asad010203/AI-Football-Ana-ytics"
RELEASE_ASSET_NAME = "football-worker.zip"
HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _gpu_status() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "cuda_available": available,
            "device": torch.cuda.get_device_name(0) if available else "CPU",
            "cuda_version": torch.version.cuda if available else None,
        }
    except Exception as exc:
        return {
            "cuda_available": False,
            "device": "Unavailable",
            "cuda_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _safe_video_name(name: str) -> str:
    path = Path(name)
    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video format")
    return f"{uuid.uuid4().hex}{suffix}"


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _run_job(job_id: str, video_path: Path, output_dir: Path) -> None:
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"
        from analyze import main as analyze_main
        from run import main as run_main

        run_main(
            ["--video", str(video_path), "--output-dir", str(output_dir)],
            standalone_mode=False,
        )
        analyze_main(
            ["--output-dir", str(output_dir), "--video", str(video_path)],
            standalone_mode=False,
        )
        with _jobs_lock:
            _jobs[job_id].update(
                status="complete",
                results_url=f"/results/{job_id}/client_response.json",
                video_url=f"/results/{job_id}/annotated.mp4",
            )
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=str(exc))


def _latest_release() -> dict[str, Any] | None:
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "football-worker",
        },
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _version_tuple(version: str) -> tuple[int, ...]:
    cleaned = version.lstrip("v").split("-")[0]
    try:
        return tuple(int(part) for part in cleaned.split("."))
    except ValueError:
        return (0,)


def _check_for_update() -> dict[str, Any]:
    release = _latest_release()
    if release is None:
        return {"current_version": APP_VERSION, "update_available": False}
    latest = str(release.get("tag_name", "")).lstrip("v")
    asset_names = {
        str(item.get("name"))
        for item in release.get("assets", [])
        if str(item.get("name", "")).startswith(f"{RELEASE_ASSET_NAME}.")
    }
    return {
        "current_version": APP_VERSION,
        "latest_version": latest,
        "update_available": _version_tuple(latest) > _version_tuple(APP_VERSION),
        "download_available": bool(asset_names),
    }


def _start_update(release: dict[str, Any]) -> None:
    assets = sorted(
        (item for item in release.get("assets", [])
         if str(item.get("name", "")).startswith(f"{RELEASE_ASSET_NAME}.")),
        key=lambda item: str(item.get("name")),
    )
    if not assets or not getattr(sys, "frozen", False):
        raise RuntimeError("No packaged worker release assets are available")
    package_dir = ROOT
    updater_executable = package_dir / "updater.exe"
    archive_urls = json.dumps([str(asset["browser_download_url"]) for asset in assets])
    subprocess.Popen(
        [str(updater_executable), str(package_dir), archive_urls, str(os.getpid())],
        cwd=package_dir,
        close_fds=True,
    )


def _page() -> bytes:
    status = _gpu_status()
    gpu_text = html.escape(str(status["device"]))
    cuda_text = "Available" if status["cuda_available"] else "Unavailable (CPU fallback)"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Football Analytics Worker</title>
<style>
body{{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 20px;color:#17202a}}
button{{padding:10px 16px;cursor:pointer}} .card{{border:1px solid #ddd;border-radius:8px;padding:18px;margin:16px 0}}
#status{{white-space:pre-wrap}} a{{display:block;margin-top:8px}}
canvas{{max-width:100%;border:1px solid #aaa;cursor:crosshair;display:block;margin:12px 0}}
</style></head><body>
<h1>Football Analytics</h1>
<div class="card"><b>Worker:</b> {APP_VERSION}<br>
<b>Device:</b> {gpu_text}<br><b>CUDA:</b> {cuda_text}</div>
<div class="card"><form id="form">
<input id="video" type="file" accept="video/*" required>
<button>Upload video</button></form>
<div id="zone" hidden>
<p id="zoneHelp">Click the goal-area points. Use at least 3 points, then click “Finish goal zone”.</p>
<img id="preview" hidden>
<canvas id="canvas"></canvas>
<button id="finish" type="button">Finish goal zone</button>
<button id="start" type="button" hidden>Start analytics</button>
</div>
<p id="status">Choose a video to begin.</p><div id="links"></div></div>
<script>
const form = document.querySelector("#form");
const status = document.querySelector("#status");
const links = document.querySelector("#links");
const zone = document.querySelector("#zone");
const preview = document.querySelector("#preview");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const finish = document.querySelector("#finish");
const start = document.querySelector("#start");
let jobId = null, points = [], pitchRef = null, phase = "polygon";
function drawZone() {{
  canvas.width = preview.naturalWidth; canvas.height = preview.naturalHeight;
  ctx.drawImage(preview, 0, 0);
  if (points.length) {{
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(point => ctx.lineTo(point[0], point[1]));
    if (phase === "pitch") ctx.closePath();
    ctx.strokeStyle = "#00a000"; ctx.lineWidth = Math.max(3, canvas.width / 500); ctx.stroke();
    points.forEach(point => {{ ctx.fillStyle = "#00a000"; ctx.fillRect(point[0]-5, point[1]-5, 10, 10); }});
  }}
  if (pitchRef) {{ ctx.fillStyle = "#d00000"; ctx.beginPath(); ctx.arc(pitchRef[0], pitchRef[1], 8, 0, 2*Math.PI); ctx.fill(); }}
}}
preview.onload = drawZone;
canvas.addEventListener("click", event => {{
  const rect = canvas.getBoundingClientRect();
  const point = [(event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height];
  if (phase === "polygon") points.push(point);
  else if (!pitchRef) pitchRef = point;
  drawZone();
}});
finish.addEventListener("click", () => {{
  if (points.length < 3) {{ status.textContent = "Add at least 3 goal-zone points."; return; }}
  phase = "pitch"; finish.hidden = true; start.hidden = false;
  document.querySelector("#zoneHelp").textContent = "Click one point clearly on the pitch, outside the goal, then click “Start analytics”.";
}});
start.addEventListener("click", async () => {{
  if (!pitchRef) {{ status.textContent = "Click the pitch reference point first."; return; }}
  start.disabled = true; status.textContent = "Saving goal zone...";
  const response = await fetch("/api/jobs/" + jobId + "/zone", {{
    method:"POST", headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{polygon_xy:points, pitch_reference_xy:pitchRef}})
  }});
  const result = await response.json();
  if (!response.ok) {{ status.textContent = result.error || "Unable to save goal zone"; start.disabled = false; return; }}
  zone.hidden = true; status.textContent = "Processing..."; poll(jobId);
}});
fetch("/api/update-check").then(response => response.json()).then(update => {{
  if (update.status === "update_started") {{
    status.textContent = "Updating worker. Please wait...";
  }}
}}).catch(() => {{}});
form.addEventListener("submit", async (event) => {{
  event.preventDefault(); links.innerHTML = ""; status.textContent = "Uploading video...";
  const data = new FormData(); data.append("video", document.querySelector("#video").files[0]);
  const response = await fetch("/api/jobs", {{method:"POST", body:data}});
  const job = await response.json();
  if (!response.ok) {{ status.textContent = job.error || "Unable to start job"; return; }}
  jobId = job.job_id; phase = "polygon"; points = []; pitchRef = null;
  preview.src = "/previews/" + jobId + "/first_frame.jpg";
  zone.hidden = false; finish.hidden = false; start.hidden = true;
  status.textContent = "Draw the goal zone on the first video frame.";
}});
async function poll(id) {{
  const response = await fetch("/api/jobs/" + id); const job = await response.json();
  status.textContent = job.status === "processing" ? "Processing..." :
    job.status === "failed" ? "Failed: " + job.error : job.status;
  if (job.status === "complete") {{
    links.innerHTML = `<a href="${{job.results_url}}" target="_blank">Open JSON results</a>
      <a href="${{job.video_url}}" target="_blank">Open annotated video</a>`;
    return;
  }}
  if (job.status !== "failed") setTimeout(() => poll(id), 2000);
}}
</script></body></html>""".encode("utf-8")


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "FootballWorker/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = _page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/status":
            self._send_json({"version": APP_VERSION, "gpu": _gpu_status()})
            return
        if self.path == "/api/update-check":
            release = _latest_release()
            result = _check_for_update()
            if release and result["update_available"]:
                _start_update(release)
                result["status"] = "update_started"
            self._send_json(result)
            if result.get("status") == "update_started":
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path.startswith("/api/jobs/"):
            if self.path.endswith("/zone"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            job = _job_snapshot(self.path.rsplit("/", 1)[-1])
            self._send_json(job or {"error": "Job not found"}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
            return
        if self.path.startswith("/previews/"):
            self._serve_preview()
            return
        if self.path.startswith("/results/"):
            self._serve_result()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path.startswith("/api/jobs/") and self.path.endswith("/zone"):
            self._save_zone(self.path.split("/")[3])
            return
        if self.path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if int(self.headers.get("Content-Length", "0")) > MAX_UPLOAD_BYTES:
            self._send_json({"error": "Video exceeds the 50 GB upload limit"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )
        if "video" not in form or not getattr(form["video"], "filename", None):
            self._send_json({"error": "Select a video first"}, HTTPStatus.BAD_REQUEST)
            return
        job_id = uuid.uuid4().hex
        try:
            filename = _safe_video_name(form["video"].filename)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        video_dir = ROOT / "input" / "worker_jobs" / job_id
        output_dir = ROOT / "output" / "worker_jobs" / job_id
        video_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        video_path = video_dir / filename
        with video_path.open("wb") as target:
            shutil.copyfileobj(form["video"].file, target)
        capture = cv2.VideoCapture(str(video_path))
        ok, first_frame = capture.read()
        capture.release()
        if not ok:
            shutil.rmtree(video_dir)
            shutil.rmtree(output_dir)
            self._send_json({"error": "Unable to read the first video frame"}, HTTPStatus.BAD_REQUEST)
            return
        cv2.imwrite(str(output_dir / "first_frame.jpg"), first_frame)
        with _jobs_lock:
            _jobs[job_id] = {"job_id": job_id, "status": "awaiting_zone"}
        self._send_json({"job_id": job_id}, HTTPStatus.ACCEPTED)

    def _serve_preview(self) -> None:
        parts = self.path.split("/")
        if len(parts) != 4 or parts[3] != "first_frame.jpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = ROOT / "output" / "worker_jobs" / parts[2] / "first_frame.jpg"
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)

    def _save_zone(self, job_id: str) -> None:
        job = _job_snapshot(job_id)
        if job is None or job["status"] != "awaiting_zone":
            self._send_json({"error": "Job is not awaiting a goal zone"}, HTTPStatus.BAD_REQUEST)
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        polygon = data.get("polygon_xy")
        pitch_ref = data.get("pitch_reference_xy")
        if not isinstance(polygon, list) or len(polygon) < 3 or not isinstance(pitch_ref, list):
            self._send_json({"error": "Provide at least 3 polygon points and one pitch reference point"}, HTTPStatus.BAD_REQUEST)
            return
        zone_path = ROOT / "output" / "worker_jobs" / job_id / "goal_zone.json"
        zone_path.write_text(json.dumps({
            "polygon_xy": polygon,
            "pitch_reference_xy": pitch_ref,
        }, indent=2))
        video_dir = ROOT / "input" / "worker_jobs" / job_id
        video_path = next(video_dir.iterdir())
        output_dir = ROOT / "output" / "worker_jobs" / job_id
        with _jobs_lock:
            _jobs[job_id]["status"] = "queued"
        threading.Thread(target=_run_job, args=(job_id, video_path, output_dir), daemon=True).start()
        self._send_json({"job_id": job_id, "status": "queued"}, HTTPStatus.ACCEPTED)

    def _serve_result(self) -> None:
        parts = self.path.split("/")
        if len(parts) != 4:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        job_id, name = parts[2], parts[3]
        if name not in {"client_response.json", "annotated.mp4", "ball_heatmap.png"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = ROOT / "output" / "worker_jobs" / job_id / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "application/json" if name.endswith(".json") else "video/mp4" if name.endswith(".mp4") else "image/png"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[worker] {format % args}")


def main() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(RESOURCE_ROOT)
    server = ThreadingHTTPServer((HOST, PORT), WorkerHandler)
    print(f"Football Analytics Worker {APP_VERSION}")
    print(f"Open http://{HOST}:{PORT}")
    print(f"GPU: {_gpu_status()['device']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping worker")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
