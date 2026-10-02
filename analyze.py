"""Post-process detection results into the client-facing v0.1 JSON.

Runs after run.py. Consumes output/<video-stem>/results.json + the source video,
produces:
  - client_response.json       (the exact schema we send to the client)
  - ball_heatmap.png           (ball position density over the full run)
  - goal_clips/goal_NNN.mp4    (5-10s clip around each goal event)

Focused on goal + ball + player analytics. Player identity/team fields are
omitted for now — they'll be added when re-ID + team classifier land.
"""

from __future__ import annotations
import json
import datetime
from pathlib import Path

import click
import cv2
import numpy as np

_PLAYER_MATCH_MAX_PX = 300         # per-frame nearest-neighbor player matching threshold (4K)
_GOAL_LOOKBACK_FRAMES = 30         # ~1 s at 30 fps: window for shot-distance + speed-at-goal


def _fmt_hhmmss(seconds: float) -> str:
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _extract_clip(video_path: Path, out_path: Path, start_s: float, end_s: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(start_s * fps))
    end_frame = int(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()


def _ball_heatmap(frames: list[dict], width: int, height: int, out_path: Path,
                  cell_px: int = 40) -> None:
    gw = max(1, width // cell_px)
    gh = max(1, height // cell_px)
    heat = np.zeros((gh, gw), dtype=np.float32)
    for f in frames:
        for b in f["ball"]:
            if b["predicted"]:
                continue
            x1, y1, x2, y2 = b["bbox_xyxy"]
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            heat[min(gh - 1, cy // cell_px), min(gw - 1, cx // cell_px)] += 1
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)
    heat = cv2.GaussianBlur(heat, (5, 5), 0)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR))


def _ball_stats(frames: list[dict], fps: float) -> dict:
    """Ball total distance, average speed, peak instantaneous speed. All in pixels."""
    pos = []
    for f in frames:
        real = next((b for b in f["ball"] if not b["predicted"]), None)
        if real is None:
            continue
        x1, y1, x2, y2 = real["bbox_xyxy"]
        pos.append((f["frame_idx"], (x1 + x2) * 0.5, (y1 + y2) * 0.5))
    if len(pos) < 2:
        return {"total_distance_px": 0.0, "avg_speed_px_per_s": 0.0, "max_speed_px_per_s": 0.0}
    total = 0.0
    peak = 0.0
    for (i1, x1, y1), (i2, x2, y2) in zip(pos, pos[1:]):
        d = float(np.hypot(x2 - x1, y2 - y1))
        total += d
        dt = (i2 - i1) / fps
        if dt > 0:
            peak = max(peak, d / dt)
    span_s = (pos[-1][0] - pos[0][0]) / fps
    return {
        "total_distance_px": round(total, 2),
        "avg_speed_px_per_s": round(total / span_s, 2) if span_s > 0 else 0.0,
        "max_speed_px_per_s": round(peak, 2),
    }


def _player_stats(frames: list[dict], fps: float) -> dict:
    """Aggregate player movement via per-frame nearest-neighbor bbox matching.

    NOT a real per-player tracker. Only aggregate speed metrics are meaningful;
    named per-player stats need a proper tracker (Norfair on players).
    """
    total_disp = 0.0
    pairs = 0
    peak_speed = 0.0
    total_seen = 0
    prev_centers: list[tuple[float, float]] = []
    for f in frames:
        centers = []
        for p in f["players"]:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        total_seen += len(centers)
        if prev_centers:
            used = set()
            for cx, cy in centers:
                best_d, best_i = _PLAYER_MATCH_MAX_PX + 1, -1
                for i, (px, py) in enumerate(prev_centers):
                    if i in used:
                        continue
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_i = d, i
                if 0 <= best_i and best_d <= _PLAYER_MATCH_MAX_PX:
                    used.add(best_i)
                    total_disp += best_d
                    pairs += 1
                    inst = best_d * fps
                    if inst > peak_speed:
                        peak_speed = inst
        prev_centers = centers
    return {
        "avg_detected_per_frame": round(total_seen / len(frames), 2) if frames else 0.0,
        "avg_speed_px_per_s": round(total_disp * fps / pairs, 2) if pairs else 0.0,
        "max_speed_px_per_s": round(peak_speed, 2),
        "speed_estimation_method": (
            "nearest-neighbor bbox matching between consecutive frames "
            "(approximation; not per-player tracking)"
        ),
    }


def _per_goal_stats(goal: dict, frames_by_idx: dict, fps: float) -> dict:
    """Per-goal analytics: shot distance + ball speeds around the trigger.

    shot_distance_px is the pixel distance from the earliest real ball detection
    within the last _GOAL_LOOKBACK_FRAMES to the goal's entry point — a proxy
    for how far the ball travelled just before scoring.
    """
    trig = goal["frame_idx"]
    entry = goal.get("entry_point") or goal.get("trigger_center")
    positions: list[tuple[int, float, float]] = []
    for i in range(trig - _GOAL_LOOKBACK_FRAMES, trig + 1):
        f = frames_by_idx.get(i)
        if f is None:
            continue
        real = next((b for b in f["ball"] if not b["predicted"]), None)
        if real is None:
            continue
        x1, y1, x2, y2 = real["bbox_xyxy"]
        positions.append((i, (x1 + x2) * 0.5, (y1 + y2) * 0.5))
    if not positions:
        return {"shot_distance_px": None,
                "ball_speed_at_goal_px_per_s": None,
                "ball_avg_speed_before_goal_px_per_s": None}
    _, ex0, ey0 = positions[0]
    shot_distance = ((entry[0] - ex0) ** 2 + (entry[1] - ey0) ** 2) ** 0.5
    speed_at_goal = None
    if len(positions) >= 2:
        (i1, x1, y1), (i2, x2, y2) = positions[-2], positions[-1]
        dt = (i2 - i1) / fps
        if dt > 0:
            speed_at_goal = float(np.hypot(x2 - x1, y2 - y1)) / dt
    total = 0.0
    for (i1, x1, y1), (i2, x2, y2) in zip(positions, positions[1:]):
        total += float(np.hypot(x2 - x1, y2 - y1))
    span_s = (positions[-1][0] - positions[0][0]) / fps
    avg_before = total / span_s if span_s > 0 else None
    return {
        "shot_distance_px": round(shot_distance, 2),
        "ball_speed_at_goal_px_per_s": round(speed_at_goal, 2) if speed_at_goal else None,
        "ball_avg_speed_before_goal_px_per_s": round(avg_before, 2) if avg_before else None,
    }


@click.command()
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help="the output/<video-stem>/ folder produced by run.py")
@click.option("--video", "video_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="source video (defaults to the path recorded in results.json)")
@click.option("--clip-before", type=float, default=5.0,
              help="seconds of context before each goal in the clip")
@click.option("--clip-after", type=float, default=5.0,
              help="seconds of context after each goal in the clip")
@click.option("--base-url", type=str, default="",
              help="URL prefix for asset URLs; blank = relative paths in the JSON")
def main(output_dir: Path, video_path: Path | None, clip_before: float, clip_after: float,
         base_url: str) -> None:
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise click.ClickException(f"{results_path} not found — run run.py first")

    with results_path.open("r", encoding="utf-8") as _fh:
        results = json.load(_fh)
    if video_path is None:
        video_path = Path(results["video"])
    if not video_path.exists():
        raise click.ClickException(f"video not found at {video_path}")

    fps = float(results["fps"])
    width = int(results["width"])
    height = int(results["height"])
    total_frames = int(results["total_frames"])
    frames = results.get("frames", [])
    frames_by_idx = {f["frame_idx"]: f for f in frames}
    duration_s = round(total_frames / fps, 3) if fps > 0 else 0.0

    def as_url(path: str) -> str:
        return f"{base_url.rstrip('/')}/{path}" if base_url else path

    heatmap_path = output_dir / "ball_heatmap.png"
    _ball_heatmap(frames, width, height, heatmap_path)

    clips_dir = output_dir / "goal_clips"
    clips_dir.mkdir(exist_ok=True)
    goal_events_out = []
    for i, g in enumerate(results.get("goal_events", []), 1):
        goal_id = f"goal_{i:03d}"
        clip_start = max(0.0, g["timestamp_s"] - clip_before)
        clip_end = min(duration_s, g["timestamp_s"] + clip_after)
        clip_name = f"{goal_id}.mp4"
        _extract_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
        pg = _per_goal_stats(g, frames_by_idx, fps)
        goal_events_out.append({
            "goal_event_id": goal_id,
            "frame_idx": g["frame_idx"],
            "timestamp_s": round(g["timestamp_s"], 3),
            "timestamp_hhmmss": _fmt_hhmmss(g["timestamp_s"]),
            "entry_point": g.get("entry_point"),
            "trigger_center": g.get("trigger_center"),
            "trigger_confidence": round(g.get("trigger_confidence", 0.0), 3),
            "shot_distance_px": pg["shot_distance_px"],
            "ball_speed_at_goal_px_per_s": pg["ball_speed_at_goal_px_per_s"],
            "ball_avg_speed_before_goal_px_per_s": pg["ball_avg_speed_before_goal_px_per_s"],
            "clip_url": as_url(f"goal_clips/{clip_name}"),
            "clip_start_s": round(clip_start, 3),
            "clip_end_s": round(clip_end, 3),
        })

    ball_stats = _ball_stats(frames, fps)
    ball_stats["heatmap_url"] = as_url("ball_heatmap.png")
    player_stats = _player_stats(frames, fps)

    client = {
        "schema_version": "0.1",
        "match_id": f"match_{video_path.stem}",
        "video_source": str(video_path).replace("\\", "/"),
        "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),

        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": duration_s,
            "total_frames": total_frames,
        },

        "goal_zone": {
            "polygon_xy": results.get("goal_zone_polygon"),
            "pitch_reference_xy": results.get("pitch_reference_xy"),
            "goal_inward_vector": results.get("goal_inward_vector"),
        },

        "match_stats": {
            "goals": len(goal_events_out),
            "ball": ball_stats,
            "players": player_stats,
        },

        "goal_events": goal_events_out,
        "static_hotspots": results.get("static_hotspots", []),

        "highlights": {
            "full_annotated_video_url": as_url("annotated.mp4"),
            "goal_clips": [
                {"goal_event_id": g["goal_event_id"], "clip_url": g["clip_url"],
                 "start_s": g["clip_start_s"], "end_s": g["clip_end_s"]}
                for g in goal_events_out
            ],
        },
    }

    out_path = output_dir / "client_response.json"
    out_path.write_text(json.dumps(client, indent=2))

    click.echo(f"wrote: {out_path}")
    click.echo(f"wrote: {heatmap_path}")
    click.echo(f"wrote: {len(goal_events_out)} goal clips in {clips_dir}/")
    click.echo(f"goals: {len(goal_events_out)}")
    for g in goal_events_out:
        click.echo(f"  {g['goal_event_id']}  t={g['timestamp_hhmmss']}  "
                   f"shot_dist={g['shot_distance_px']}px  "
                   f"speed_at_goal={g['ball_speed_at_goal_px_per_s']}px/s")


if __name__ == "__main__":
    main()
