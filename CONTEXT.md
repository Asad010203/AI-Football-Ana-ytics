# AI Football Analytics — Project Context

End-to-end player + ball detection and goal-event detection for football video, running locally on Windows / NVIDIA GPU.

## What the project does

Given a video file:

1. Detects and boxes every **player** (green) using YOLO11-x, COCO class 0 (`person`).
2. Detects and tracks the **ball** (yellow) using YOLO26-x + a Norfair Kalman tracker. COCO class 32 (`sports ball`).
3. Detects **goal events** — moments the tracked ball crosses a user-annotated goal frame. On event: a yellow "GOAL!" flashes on the annotated video for ~1s, and a structured record is added to `results.json`.

Output for a video `input/foo.mp4`:
- `output/foo/annotated.mp4` — the source video with detection overlays + goal flashes
- `output/foo/results.json` — per-frame players + ball entries, plus a top-level `goal_events` array and discovered `static_hotspots`
- `output/foo/goal_zone.json` — the annotated goal-frame corners (re-used across runs)

## Repo layout

Flat by convention; two-file rule broken only for `ball_detector.py` (added later, same style):

```
D:\AI football analytics\
├── .venv\                    virtualenv
├── .gitignore
├── requirements.txt          Python deps (torch/torchvision installed from local wheels)
├── README.md                 minimal install + run
├── CONTEXT.md                this file
├── Modals\                   NOT flat — legacy folder holding the .pt files:
│   ├── yolov11\yolo11x.pt    114 MB, players
│   └── yolov26\yolo26x.pt    113 MB, ball (Jan 2026 release)
├── wheels\                   local prebuilt torch/torchvision wheels for py3.10 + CUDA 12.4
├── input\                    place source videos here
├── output\<video-stem>\      annotated.mp4 + results.json + goal_zone.json
├── player_detector.py        wrapper around Ultralytics YOLO for player detection
├── ball_detector.py          wrapper around Ultralytics YOLO + Norfair for ball detection + tracking
└── run.py                    CLI: video in, annotated video + JSON out, incl. zone picker
```

## Environment

- **Windows 11**, **Python 3.10**, NVIDIA GPU (dev machine: Quadro P3200, 6 GB VRAM).
- Torch 2.6.0+cu124 and torchvision 0.21.0+cu124 installed **from local wheels only** — never from PyPI or the pytorch.org index:
  - `wheels\torch-2.6.0+cu124-cp310-cp310-win_amd64.whl`
  - `wheels\torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl`
- Ultralytics: `ultralytics==8.4.142` (supports YOLO26 — added upstream in 8.4.125).
- Tracker: `norfair>=2.2` (installed version 2.3.0).

Set up from scratch:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install "numpy<2" "opencv-python==4.10.0.84" "Pillow>=10.0"
.\.venv\Scripts\python.exe -m pip install "wheels\torch-2.6.0+cu124-cp310-cp310-win_amd64.whl" "wheels\torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl"
.\.venv\Scripts\python.exe -m pip install "ultralytics==8.4.142" "tqdm>=4.66" "click>=8.1" "norfair>=2.2"
```

CUDA smoke test (should print `True`):
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Player detection (`player_detector.py`)

Thin wrapper around `ultralytics.YOLO`:
- Default weights: `Modals/yolov11/yolo11x.pt`
- Only class 0 (`person`) is asked for, so we never post-filter for non-players.
- `imgsz=1280`, `conf=0.30`, `iou=0.50`, `device="cuda"`.
- `infer(frame_bgr)` returns `{boxes_xyxy, confidences, class_ids}` (numpy arrays).

The class is deliberately tiny — no CLI, no tracking, no team ID. Downstream code owns rendering and aggregation.

## Ball detection (`ball_detector.py`)

Higher-effort wrapper because the ball is small, moves fast, and gets confused with static features on the pitch.

**Detection (YOLO26-x):**
- Default weights: `Modals/yolov26/yolo26x.pt`
- Only class 32 (`sports ball`).
- Large inference size: `imgsz=1920`. Small objects benefit disproportionately.
- Moderate confidence: `conf=0.40` (was 0.20 early on; raised after false positives).

**Per-frame false-positive filters:**
- Bounding-box area fraction of the frame must be in `[2e-6, 2e-2]`.
- Aspect ratio `max(w/h, h/w) <= 1.8` (a ball is near-square in bbox terms).
- Confidence `>= conf_threshold`.

**Top-1 per frame:** after filters, only the single highest-confidence detection is kept per frame and fed to the tracker. Football has one ball in play; this is the biggest single lever for killing false-positive tracks on stationary blobs (shoe patches, ad-board dots, penalty spot). If YOLO produces both the real ball and a static FP in the same frame, top-1 usually keeps the real ball (which is confidence 0.6-0.9 median). When the real ball has a low-confidence frame, top-1 may pick the static FP — that failure mode is handled downstream by the static-hotspot check in `run.py`.

**Tracking (Norfair):**
- `Tracker(distance_function="euclidean", distance_threshold=180.0, hit_counter_max=8, initialization_delay=0, pointwise_hit_counter_max=8, filter_factory=OptimizedKalmanFilterFactory(R=4.0, Q=0.1))`
- Low Q, moderate R: assumes smooth ball motion, tolerates small YOLO center jitter.
- Predictions survive up to 8 frames of YOLO misses (motion blur, brief occlusion behind a player).
- Feeds the tracker each detection's **center point**. The bbox and confidence are kept as `data` on the Detection and returned in the tracked output so downstream code can render the actual detection instead of a synthesized prediction bbox.

**Public API — one method:**
```python
infer_and_track(frame_bgr) -> [
    {"track_id": int, "bbox_xyxy": [x1,y1,x2,y2],
     "confidence": float, "predicted": bool},
    ...
]
```
`predicted=True` means the Kalman filter extrapolated this position because YOLO produced no matching detection this frame.

## Goal event detection (`run.py`) — the interesting part

This was the hardest piece. It went through several revisions as we discovered failure modes on the real 5:20, 4K, 29.97 fps test video.

### Rule (current)

A goal fires only when ALL of the following hold:

1. **Trajectory crosses the zone.** The line segment from the previous real ball position to the current real ball position enters the 4-corner goal-mouth quadrilateral. Handled by `_segment_enters_polygon`: prev outside + curr inside → entered; both endpoints outside but segment transits interior → still entered (zip-through), and the entry point on the segment is returned for downstream checks.
2. **Altitude sanity.** At the ball's x, interpolate the ground line (bottom edge) and the crossbar (top edge). The ball's bbox bottom must lie within ±0.5 × goal-height of the goal opening. A ball flying over the goal has its bbox bottom well above the crossbar → rejected.
3. **Recent-outside window.** At least `_MIN_OUTSIDE_IN_WINDOW=5` of the last `_RECENT_WINDOW=30` real detections were outside. Rejects noise-flickers when the "ball" has been lingering near the zone.
4. **Confidence.** Trigger detection confidence `>= _MIN_TRIGGER_CONF=0.50`.
5. **Cooldown.** No fire within `_GOAL_COOLDOWN=60` frames of the previous fire. One crossing produces one goal.
6. **Not at a static hotspot.** None of {prev pos, curr pos, entry point} sits within `_STATIONARY_REJECT_RADIUS=40` px of a discovered static hotspot.

### Static hotspot discovery

A tracker ID that lives ≥ `_STATIONARY_MIN_LIFETIME=15` frames while its position spread stays ≤ `_STATIONARY_MAX_SPREAD=15` px has its centroid appended to a permanent `static_hotspots` list. Once a spot is on the list, it stays — flip-flops of top-1 between the real ball and that spot won't fire goals even if the spot's live track hasn't been detected in a while.

Real-world discoveries on the test video: the **penalty spot** at (459, 652), plus a few other stationary features (posts, corner flags, ad-board dots).

### Zone annotation (`_pick_goal_corners`)

Interactive picker over the start frame. Camera is static so one frame is enough. User clicks the 4 corners of the **goal opening** in a specific order:

1. LEFT POST BASE (where left post meets ground)
2. RIGHT POST BASE (where right post meets ground)
3. RIGHT POST TOP (where right post meets crossbar)
4. LEFT POST TOP (where left post meets crossbar)

This gives each click geometric meaning:
- Bottom edge (click 1 → click 2) = **goal line**
- Top edge (click 4 → click 3) = **crossbar**
- Left edge (click 4 → click 1) = **left post**
- Right edge (click 3 → click 2) = **right post**

Ground and crossbar are interpolated per-x at trigger time for the altitude check (item 2 above).

**Preview + confirm.** After the 4th click, the picker enters a preview mode: filled semi-transparent polygon + bold outline + corner tags (BL/BR/TR/TL). HUD shows "Enter=save, R=redo, Esc=cancel". Enter commits; R wipes and restarts; Esc cancels.

**Controls:**
- Mouse wheel — zoom in/out around cursor (up to 20×)
- W/A/S/D — pan
- R (in click mode) — reset zoom + pan
- Backspace — undo last click
- Esc — cancel

**Save format** — `output/<video-stem>/goal_zone.json`:
```json
{
  "goal_corners_xy": [[bl_x,bl_y],[br_x,br_y],[tr_x,tr_y],[tl_x,tl_y]],
  "polygon_xy": [ ...same 4 points... ],
  "video_width": 3840,
  "video_height": 2160
}
```

**Backward compat.** Old zone files written before the guided-corner picker only contain `polygon_xy` (arbitrary N-vertex polygon). Those still load — the polygon-based crossing check still works — but the altitude check is skipped (recorded as `altitude_check_enabled: false` in `results.json`).

## Running it

Once — annotate the goal zone (opens the picker on frame 0):
```powershell
.\.venv\Scripts\python.exe run.py --video "input\match.mp4" --draw-zone
```

Subsequent runs — zone auto-loads:
```powershell
.\.venv\Scripts\python.exe run.py --video "input\match.mp4"
```

Options:
| Flag | Default | Meaning |
| --- | --- | --- |
| `--video PATH` | (required) | input video |
| `--output-dir PATH` | `output/<video-stem>/` | output folder |
| `--conf FLOAT` | `0.3` | player-detection confidence threshold |
| `--imgsz INT` | `1280` | player-detection inference size |
| `--start-frame INT` | `0` | absolute source frame to start at |
| `--end-frame INT` | end of video | absolute source frame to stop at |
| `--max-frames INT` | none | legacy; prefer `--end-frame` |
| `--draw-zone` | off | open the picker even if a saved zone exists |

`frame_idx` in `results.json` is always the absolute source-video frame number, so ranges from different runs are directly comparable.

## Iteration history — what we tried and why

The goal-detection rule went through several iterations. Each revision was driven by inspecting `results.json` + extracted frames to see what was actually happening, not by guessing.

**Iteration 1 — naive point-in-polygon per track.**
Rule: for each ball track, if `prev_inside=False` and `curr_inside=True`, fire. Result on the 5-min test clip: **29 goals** (real count ≈ 1).
Failure mode: track churn. 286 unique ball track IDs across 9600 frames because Kalman's distance_threshold=180 px couldn't associate detections on hard kicks at 4K. Every new track that spawned inside the zone had `prev_inside` default to False, so its first-inside frame triggered instantly.

**Iteration 2 — top-1 ball detection + per-track age/outside guards.**
Added: only the single highest-confidence ball detection per frame goes into the tracker; a fire requires the track to have existed ≥5 frames AND been outside ≥3 consecutive frames.
Result: swung to the other extreme — **0 goals** because the real ball's track kept getting killed and respawning inside the zone during the actual score, and new tracks didn't have "outside history" to satisfy the guard.

**Iteration 3 — track-id-agnostic rolling window + static-hotspot discovery.**
Dropped per-track state entirely. Rule: fire when the current real detection is inside AND recent detections were outside AND confidence and cooldown OK. Added stationary-hotspot rejection: any track that lives ≥15 frames within a 15 px spread has its centroid saved as a permanent hotspot; fires near a hotspot are rejected.
Result on the test data: **1 fire** at frame 6415 (3:34), 12 rejections — every rejection at ≈(458, 651), the penalty spot. Success on the hotspot problem.

**Iteration 4 — but that "1 fire" turned out to be a false positive.**
Inspecting the actual video: the ball was being **kicked high into the air, flying up and away from the goal** — not scoring. Its 2D pixel trajectory happened to sweep through the user-drawn polygon area, but the ball was 5+ meters above the pitch in 3D. This is the fundamental limit of 2D projection: you can't distinguish "ball in the goal" from "ball flying above the goal at the same 2D pixel column."
Concurrently: the polygon had been drawn as a diagonal band on the grass in front of the goal (not around the goal opening), which made this ambiguity much worse.

**Iteration 5 — trajectory-based crossing.**
Discovered that at 4K/30fps a hard-kicked ball moves 200+ px per frame while the user's thin goal-line polygon was only ~22 px wide. The ball zipped straight through the polygon in a single frame, and point-in-polygon on discrete frames never caught it. Replaced the point check with `_segment_enters_polygon` — samples the segment prev→curr and returns the actual entry point. Together with a stricter hotspot check (reject if prev OR curr OR entry is near a hotspot), this cleanly caught the trajectory-based case.
But false positives from balls flying over the goal remained (the fundamental 2D depth ambiguity from Iteration 4).

**Iteration 6 — guided 4-corner picker + altitude check.**
This is the current state. The picker requires exactly 4 clicks in a specific order (LEFT-POST-BASE, RIGHT-POST-BASE, RIGHT-POST-TOP, LEFT-POST-TOP) with a preview + confirm step. Each click has geometric meaning, so ground line and crossbar are known and the altitude check can reject a ball whose bbox bottom is far above the crossbar. Old free-form-polygon zone files still load (with the altitude check disabled).

## Known limitations

- **2D depth ambiguity.** Even with a correctly annotated goal frame + altitude check, there's a residual case: a ball flying directly at the correct altitude but far behind the goal in world space would project to the same pixels. In practice this doesn't happen on a normal pitch — nothing is behind the goal — but be aware.
- **Fast-camera-cut videos.** All state (recent-inside deque, prev real position, cooldown) assumes a continuous stream from one camera angle. Broadcast footage with mid-play cuts would need a scene-change reset.
- **Multi-ball scenarios.** The pipeline assumes one ball. Warm-up sessions with several balls, or a stray ball into a live game, will confuse the top-1 filter.
- **Ultra-high-speed shots.** Trajectory sampling uses 30 points along the segment prev→curr. If a shot moves more than ~200 × 30 = 6000 pixels between real detections (real-world impossible at typical fps), the sample density becomes marginal.
- **Video writer codec.** Output uses `mp4v` for portability. For a 4K source this produces a very large annotated file; players struggle to scrub through it. Not a correctness issue, just UX.
- **Windows-only setup instructions.** The wheel-based install path is Windows/CUDA-12.4-specific; POSIX users would install torch differently.

## Future improvements — likely order

1. **Velocity-based confirmation.** Compare mean ball speed in the ~10 frames before vs after the "crossing." Real goals decelerate hard (net); flyovers keep flying. Cheap, would eliminate the residual 2D-depth false positives.
2. **Ball-size sanity.** Given the goal-height in pixels from the 4 corners, expected ball radius at goal distance follows. Reject fires where the trigger detection is way outside that size range.
3. **Camera calibration.** With known real-world goal dimensions, `cv2.solvePnP` on the 4 corners gives full camera pose relative to the goal — no extra picker work needed. Then ball position becomes a proper world-space ray, and a ground-plane assumption gives a 3D point. Overkill for the current use case but a clean upgrade.
4. **Scene-change detection.** For broadcast footage: reset per-camera state when the camera cuts.
5. **Ball-visible-in-net cue.** Optional — check if the trigger detection overlaps the goal-mouth polygon AND is followed by frames where a ball is detected inside the net region. Adds latency but very high precision.

## Files, quick reference

- `player_detector.py` — YOLO11 wrapper for players. Small.
- `ball_detector.py` — YOLO26 + Norfair for ball. Small.
- `run.py` — the CLI, the picker, the goal-event logic, JSON serialization.
- `requirements.txt` — Python deps (torch/torchvision installed from `wheels\` separately).
- `README.md` — minimal install + run.
- `CONTEXT.md` — this document.
