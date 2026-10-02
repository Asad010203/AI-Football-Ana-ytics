"""CLI: run PlayerDetector + BallDetector over a video, write annotated mp4 + results.json.

Goal-event detection uses a user-drawn polygon zone. The picker is free-form
(any number of points, closed by right-click or Enter) so you can draw a shape
that covers the ball's actual path through the goal area — including some
airspace above the goal for airborne shots (headers, lobs, drives). A tight
goal-mouth-only polygon is fragile for real football where the ball is
frequently in the air when it crosses the line.

Detection is TRAJECTORY-based (not point-based): at 4K/30fps a hard-kicked ball
can move 200+ pixels between frames, so a thin polygon can sit entirely
between two consecutive ball positions — both frames report the ball "outside"
even though it actually zipped through. Instead we check whether the line
segment between the previous real ball position and the current one intersects
the polygon interior.

A fire requires ALL of:
  * the segment from the previous real ball position to the current real ball
    position enters the polygon (prev outside → curr inside, OR the segment
    briefly transits the interior when both endpoints are outside);
  * at least _MIN_OUTSIDE_IN_WINDOW of the last _RECENT_WINDOW real detections
    were outside (rejects lingering-noise fires);
  * trigger confidence >= _MIN_TRIGGER_CONF;
  * no fire within _GOAL_COOLDOWN frames (one crossing = one goal);
  * none of {prev, curr, entry_point} sits at a "static hotspot" — a spot
    where YOLO keeps producing a "ball" detection that never moves (penalty
    spot, corner flag, ad-board dot). Any tracker ID that lives for
    _STATIONARY_MIN_LIFETIME frames within _STATIONARY_MAX_SPREAD pixels has
    its centroid recorded as a permanent hotspot for the rest of the run.

Players are never checked against the zone. The polygon is never drawn on the
output video.
"""

from __future__ import annotations
import json
from collections import defaultdict, deque
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from ball_detector import BallDetector
from player_detector import PlayerDetector

_GOAL_FLASH_FRAMES = 30
_ZONE_FILENAME = "goal_zone.json"

# Goal-trigger tuning
_RECENT_WINDOW = 30
_MIN_OUTSIDE_IN_WINDOW = 5
_GOAL_COOLDOWN = 60
_MIN_TRIGGER_CONF = 0.50
_STATIONARY_MIN_LIFETIME = 15
_STATIONARY_MAX_SPREAD = 15
_STATIONARY_REJECT_RADIUS = 40
_MIN_ZONE_POINTS = 3

_TRACK_HISTORY_MAX = 200
_TRAJECTORY_SAMPLES = 30


def _segment_enters_polygon(prev_pt: tuple[float, float],
                            curr_pt: tuple[float, float],
                            polygon: np.ndarray,
                            samples: int = _TRAJECTORY_SAMPLES
                            ) -> tuple[bool, tuple[float, float] | None]:
    """Test whether the segment prev_pt -> curr_pt enters the polygon.

    Returns (entered, entry_point). If prev is already inside, returns
    (False, None). If curr is inside, entry_point is curr. Otherwise samples
    the segment and returns the first interior sample — handles fast-ball
    zip-through where both endpoints are outside but the trajectory transits
    the polygon in one frame.
    """
    if cv2.pointPolygonTest(polygon, prev_pt, False) >= 0:
        return False, None
    if cv2.pointPolygonTest(polygon, curr_pt, False) >= 0:
        return True, curr_pt
    px, py = prev_pt
    dx, dy = curr_pt[0] - px, curr_pt[1] - py
    for i in range(1, samples):
        t = i / samples
        sx, sy = px + t * dx, py + t * dy
        if cv2.pointPolygonTest(polygon, (float(sx), float(sy)), False) >= 0:
            return True, (sx, sy)
    return False, None


def _pick_polygon(frame: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Two-step picker: polygon + pitch reference point.

    Step 1 (draw): click as many polygon points as you want. Draw a shape that
    covers the ball's actual path through the goal area — include some airspace
    above the goal frame for airborne shots. Right-click or Enter closes the
    polygon (needs >= 3 points).

    Step 2 (pitch_ref): click ONE point clearly on the pitch, outside the goal.
    This defines the "goal-inward" direction (polygon centroid − pitch_ref).
    Real goals travel in that direction; clearances travel opposite to it.

    Step 3 (preview): filled polygon + pitch-reference dot + arrow showing
    "goal-inward". Enter saves, R redoes from scratch, Esc cancels.

    Returns (polygon_points, pitch_reference_point).

    Controls:
      L-click       add polygon point (step 1) or pitch reference (step 2)
      R-click       close polygon (step 1)
      Enter         close polygon (step 1) or save (preview)
      Backspace     undo last polygon point (step 1) or go back to draw (step 2)
      Wheel         zoom around cursor
      w/a/s/d       pan
      r             reset zoom+pan (drawing) or redo from scratch (preview)
      Esc           cancel
    """
    _MIN_POINT_GAP = 5

    win = "draw goal zone"
    img_h, img_w = frame.shape[:2]
    max_canvas_w, max_canvas_h = 1600, 900
    base_scale = min(max_canvas_w / img_w, max_canvas_h / img_h, 1.0)
    canvas_w = max(1, int(img_w * base_scale))
    canvas_h = max(1, int(img_h * base_scale))

    pts: list[tuple[int, int]] = []
    pitch_ref: list[tuple[int, int]] = []  # 0 or 1 element; list so nested fns can mutate
    state = {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0,
             "mx": canvas_w // 2, "my": canvas_h // 2,
             "done": False, "cancel": False,
             "stage": "draw"}  # "draw" -> "pitch_ref" -> "preview"

    def eff_scale() -> float:
        return base_scale * state["zoom"]

    def clamp_pan() -> None:
        vis_w = canvas_w / eff_scale()
        vis_h = canvas_h / eff_scale()
        state["pan_x"] = max(0.0, min(state["pan_x"], max(0.0, img_w - vis_w)))
        state["pan_y"] = max(0.0, min(state["pan_y"], max(0.0, img_h - vis_h)))

    def canvas_to_img(cx: int, cy: int) -> tuple[int, int]:
        s = eff_scale()
        return int(state["pan_x"] + cx / s), int(state["pan_y"] + cy / s)

    def img_to_canvas(ix: int, iy: int) -> tuple[int, int]:
        s = eff_scale()
        return int((ix - state["pan_x"]) * s), int((iy - state["pan_y"]) * s)

    def on_mouse(event: int, x: int, y: int, flags: int, _p) -> None:
        state["mx"], state["my"] = x, y
        stage = state["stage"]
        if stage == "preview":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            ip = canvas_to_img(x, y)
            if stage == "draw":
                if pts and abs(ip[0] - pts[-1][0]) <= _MIN_POINT_GAP \
                        and abs(ip[1] - pts[-1][1]) <= _MIN_POINT_GAP:
                    return
                pts.append(ip)
            elif stage == "pitch_ref":
                pitch_ref[:] = [ip]
                state["stage"] = "preview"
        elif event == cv2.EVENT_RBUTTONDOWN and stage == "draw" and len(pts) >= _MIN_ZONE_POINTS:
            state["stage"] = "pitch_ref"
        elif event == cv2.EVENT_MOUSEWHEEL:
            img_x, img_y = canvas_to_img(x, y)
            factor = 1.25 if flags > 0 else 1.0 / 1.25
            state["zoom"] = max(1.0, min(state["zoom"] * factor, 20.0))
            s_new = eff_scale()
            state["pan_x"] = img_x - x / s_new
            state["pan_y"] = img_y - y / s_new
            clamp_pan()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        s = eff_scale()
        vis_w = max(1, int(canvas_w / s))
        vis_h = max(1, int(canvas_h / s))
        px, py = int(state["pan_x"]), int(state["pan_y"])
        sub = frame[py:py + vis_h, px:px + vis_w]
        canvas = cv2.resize(sub, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
        cpts = [img_to_canvas(ix, iy) for ix, iy in pts]
        cref = [img_to_canvas(ix, iy) for ix, iy in pitch_ref] if pitch_ref else []
        if state["stage"] == "preview" and len(cpts) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(cpts, dtype=np.int32)], (0, 255, 255))
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [np.array(cpts, dtype=np.int32)], True, (0, 255, 255), 3)
            for i, p in enumerate(cpts):
                cv2.circle(canvas, p, 8, (0, 255, 255), -1)
                cv2.circle(canvas, p, 8, (0, 0, 0), 2)
                cv2.putText(canvas, str(i + 1), (p[0] + 12, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if cref:
                pr = cref[0]
                cv2.circle(canvas, pr, 10, (255, 0, 255), -1)
                cv2.circle(canvas, pr, 10, (0, 0, 0), 2)
                cv2.putText(canvas, "PITCH", (pr[0] + 14, pr[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                cx = int(sum(p[0] for p in cpts) / len(cpts))
                cy = int(sum(p[1] for p in cpts) / len(cpts))
                cv2.arrowedLine(canvas, pr, (cx, cy), (255, 0, 255), 3, tipLength=0.05)
        else:
            for i, p in enumerate(cpts):
                cv2.circle(canvas, p, 6, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(canvas, cpts[i - 1], p, (0, 255, 255), 2)
            if state["stage"] == "draw" and len(cpts) >= 2:
                cv2.line(canvas, cpts[-1], cpts[0], (100, 200, 200), 1, cv2.LINE_AA)
            # In pitch_ref stage, close the polygon outline so it stays visible.
            if state["stage"] == "pitch_ref" and len(cpts) >= 3:
                cv2.polylines(canvas, [np.array(cpts, dtype=np.int32)], True, (0, 255, 255), 2)
            mx, my = state["mx"], state["my"]
            ring_color = (255, 0, 255) if state["stage"] == "pitch_ref" else (0, 255, 255)
            cv2.circle(canvas, (mx, my), 7, ring_color, 1)
            if state["stage"] == "draw" and cpts:
                cv2.line(canvas, cpts[-1], (mx, my), (0, 255, 255), 1, cv2.LINE_AA)
        if state["stage"] == "preview":
            label = f"preview ({len(pts)} pts + pitch ref):  Enter=save  |  R=redo  |  Esc=cancel"
        elif state["stage"] == "pitch_ref":
            label = "click a point on the pitch OUTSIDE the goal (this defines 'goal-inward' direction)"
        else:
            label = (f"draw: L-click add  |  R-click/Enter close (>={_MIN_ZONE_POINTS} pts)  "
                     f"|  Backspace undo  |  pts: {len(pts)}")
        cv2.rectangle(canvas, (0, 0), (canvas_w, 62), (0, 0, 0), -1)
        cv2.putText(canvas, label, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"zoom {state['zoom']:.1f}x  |  wheel=zoom  w/a/s/d=pan  r=reset  Esc=cancel",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            state["cancel"] = True
        elif state["stage"] == "preview":
            if k == 13:
                state["done"] = True
            elif k == ord('r'):
                pts.clear()
                pitch_ref.clear()
                state["stage"] = "draw"
        elif state["stage"] == "pitch_ref":
            if k == 8:  # back to drawing
                state["stage"] = "draw"
        else:  # draw
            if k == 13 and len(pts) >= _MIN_ZONE_POINTS:
                state["stage"] = "pitch_ref"
            elif k == 8 and pts:
                pts.pop()
            elif k == ord('r'):
                state["zoom"] = 1.0
                state["pan_x"] = 0.0
                state["pan_y"] = 0.0
            elif k in (ord('w'), ord('a'), ord('s'), ord('d')):
                step = 60.0 / eff_scale()
                if k == ord('w'): state["pan_y"] -= step
                elif k == ord('s'): state["pan_y"] += step
                elif k == ord('a'): state["pan_x"] -= step
                elif k == ord('d'): state["pan_x"] += step
                clamp_pan()
        if state["done"] or state["cancel"]:
            break
    cv2.destroyWindow(win)
    if state["cancel"] or len(pts) < _MIN_ZONE_POINTS or not pitch_ref:
        raise click.ClickException(
            f"zone not defined (need >= {_MIN_ZONE_POINTS} polygon points + 1 pitch reference)"
        )
    return pts, pitch_ref[0]


def _resolve_zone(zone_path: Path, first_frame: np.ndarray, redraw: bool
                  ) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    """Return (polygon_int32, pitch_reference_or_None).

    Save format is `polygon_xy` + `pitch_reference_xy`. Legacy files with only
    `polygon_xy` (or `goal_corners_xy`) still load; directional check is then
    silently skipped.
    """
    if redraw:
        pts, pref = _pick_polygon(first_frame)
        h, w = first_frame.shape[:2]
        zone_path.write_text(json.dumps({
            "polygon_xy": [list(p) for p in pts],
            "pitch_reference_xy": list(pref),
            "video_width": w, "video_height": h,
        }, indent=2))
        click.echo(f"zone:   saved to {zone_path} ({len(pts)} polygon points + pitch ref {pref})")
        return np.array(pts, dtype=np.int32), pref
    if zone_path.exists():
        data = json.loads(zone_path.read_text())
        pts = data.get("polygon_xy") or data.get("goal_corners_xy")
        if pts is None:
            click.echo(f"zone:   {zone_path} has no polygon; skipping goal detection")
            return None, None
        pref = data.get("pitch_reference_xy")
        pref_t = tuple(pref) if pref is not None else None
        if pref_t is None:
            click.echo(f"zone:   loaded {zone_path} ({len(pts)} points, no pitch ref → "
                       f"directional check disabled)")
        else:
            click.echo(f"zone:   loaded {zone_path} ({len(pts)} points + pitch ref {pref_t})")
        return np.array(pts, dtype=np.int32), pref_t
    click.echo("zone:   none configured; skipping goal detection (use --draw-zone to create one)")
    return None, None


@click.command()
@click.option("--video", "video_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--conf", type=float, default=0.3)
@click.option("--imgsz", type=int, default=1280)
@click.option("--max-frames", type=int, default=None, help="deprecated; prefer --end-frame")
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
@click.option("--draw-zone", is_flag=True,
              help="open a picker on the start frame to (re)draw the goal polygon")
def main(video_path: Path, output_dir: Path | None, conf: float, imgsz: int,
         max_frames: int | None, start_frame: int, end_frame: int | None,
         draw_zone: bool) -> None:
    out_dir = output_dir if output_dir is not None else Path("output") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.mp4"
    json_path = out_dir / "results.json"
    zone_path = out_dir / _ZONE_FILENAME

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise click.ClickException(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is not None:
        stop_frame = min(end_frame, total_frames)
    elif max_frames is not None:
        stop_frame = min(start_frame + max_frames, total_frames)
    else:
        stop_frame = total_frames
    if start_frame >= stop_frame:
        raise click.ClickException(f"empty range: start_frame={start_frame} stop_frame={stop_frame}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = cap.read()
    if not ok:
        raise click.ClickException(f"cannot read start_frame={start_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    polygon, pitch_ref = _resolve_zone(zone_path, first_frame, redraw=draw_zone)
    goal_inward: tuple[float, float] | None = None
    if polygon is not None and pitch_ref is not None:
        cx0 = float(polygon[:, 0].mean())
        cy0 = float(polygon[:, 1].mean())
        goal_inward = (cx0 - pitch_ref[0], cy0 - pitch_ref[1])

    writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = PlayerDetector(conf_threshold=conf, image_size=imgsz)
    ball_detector = BallDetector()

    meta = {
        "video": str(video_path).replace("\\", "/"),
        "fps": float(fps),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "conf": conf,
        "imgsz": imgsz,
        "start_frame": start_frame,
        "end_frame": stop_frame,
        "goal_zone_polygon": polygon.tolist() if polygon is not None else None,
        "pitch_reference_xy": list(pitch_ref) if pitch_ref is not None else None,
        "goal_inward_vector": list(goal_inward) if goal_inward is not None else None,
        "goal_rule": {
            "recent_window": _RECENT_WINDOW,
            "min_outside_in_window": _MIN_OUTSIDE_IN_WINDOW,
            "cooldown_frames": _GOAL_COOLDOWN,
            "min_trigger_conf": _MIN_TRIGGER_CONF,
            "stationary_min_lifetime": _STATIONARY_MIN_LIFETIME,
            "stationary_max_spread": _STATIONARY_MAX_SPREAD,
            "stationary_reject_radius": _STATIONARY_REJECT_RADIUS,
        },
        "frames": [],
        "goal_events": [],
        "static_hotspots": [],
    }

    recent_inside: deque[bool] = deque(maxlen=_RECENT_WINDOW)
    track_positions: dict[int, deque] = defaultdict(lambda: deque(maxlen=_TRACK_HISTORY_MAX))
    static_hotspots: list[tuple[float, float]] = []
    prev_real_pos: tuple[float, float] | None = None
    last_goal_frame = -_GOAL_COOLDOWN
    goal_flash = 0
    total_players = 0
    frame_idx = start_frame

    with tqdm(total=stop_frame - start_frame, desc="detect", unit="f") as pbar:
        while frame_idx < stop_frame:
            ok, frame = cap.read()
            if not ok:
                break

            det = detector.infer(frame)
            players = []
            for (x1, y1, x2, y2), c in zip(det["boxes_xyxy"], det["confidences"]):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                players.append({"bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(c)})

            balls = ball_detector.infer_and_track(frame)
            for b in balls:
                x1, y1, x2, y2 = b["bbox_xyxy"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 255), 1 if b["predicted"] else 2)
                cv2.putText(frame, f"Ball {b['track_id']} {b['confidence']:.2f}",
                            (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if polygon is not None:
                for b in balls:
                    tid = int(b["track_id"])
                    x1, y1, x2, y2 = b["bbox_xyxy"]
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    track_positions[tid].append((cx, cy))
                    pos = track_positions[tid]
                    if len(pos) >= _STATIONARY_MIN_LIFETIME:
                        xs = [p[0] for p in pos]
                        ys = [p[1] for p in pos]
                        spread = max(max(xs) - min(xs), max(ys) - min(ys))
                        if spread <= _STATIONARY_MAX_SPREAD:
                            mx = sum(xs) / len(xs)
                            my = sum(ys) / len(ys)
                            if not any(abs(hx - mx) <= _STATIONARY_REJECT_RADIUS and
                                       abs(hy - my) <= _STATIONARY_REJECT_RADIUS
                                       for (hx, hy) in static_hotspots):
                                static_hotspots.append((mx, my))

                real = next((b for b in balls if not b["predicted"]), None)
                if real is not None:
                    x1, y1, x2, y2 = real["bbox_xyxy"]
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    curr_pos = (float(cx), float(cy))
                    curr_bbox = [float(x1), float(y1), float(x2), float(y2)]
                    inside = cv2.pointPolygonTest(polygon, curr_pos, False) >= 0
                    if prev_real_pos is not None:
                        entered, entry_pt = _segment_enters_polygon(
                            prev_real_pos, curr_pos, polygon)

                        def _near_hotspot(p: tuple[float, float]) -> bool:
                            return any(abs(hx - p[0]) <= _STATIONARY_REJECT_RADIUS
                                       and abs(hy - p[1]) <= _STATIONARY_REJECT_RADIUS
                                       for (hx, hy) in static_hotspots)

                        outside_recent = sum(1 for x in recent_inside if not x)
                        # Directional filter: ball must be traveling INTO the goal, not out.
                        # If pitch_reference is missing (legacy zone), skip this check.
                        direction_ok = True
                        if goal_inward is not None:
                            bdx = curr_pos[0] - prev_real_pos[0]
                            bdy = curr_pos[1] - prev_real_pos[1]
                            dot = bdx * goal_inward[0] + bdy * goal_inward[1]
                            direction_ok = dot > 0
                        # Hotspot rejection applies ONLY to the entry point. Rejecting on
                        # prev/curr as well killed real goals where the ball was momentarily
                        # at rest (set-piece, kickoff) or ended near a static-FP location
                        # after crossing. The trajectory geometry — where the segment
                        # crosses the polygon edge — is the only reliable signal.
                        if (entered
                                and direction_ok
                                and outside_recent >= _MIN_OUTSIDE_IN_WINDOW
                                and real["confidence"] >= _MIN_TRIGGER_CONF
                                and (frame_idx - last_goal_frame) >= _GOAL_COOLDOWN
                                and entry_pt is not None
                                and not _near_hotspot(entry_pt)):
                            meta["goal_events"].append({
                                "frame_idx": frame_idx,
                                "timestamp_s": frame_idx / fps,
                                "track_id": int(real["track_id"]),
                                "entry_point": [int(entry_pt[0]), int(entry_pt[1])],
                                "trigger_center": [int(cx), int(cy)],
                                "trigger_bbox_xyxy": curr_bbox,
                                "trigger_confidence": float(real["confidence"]),
                            })
                            goal_flash = _GOAL_FLASH_FRAMES
                            last_goal_frame = frame_idx
                    recent_inside.append(inside)
                    prev_real_pos = curr_pos

            if goal_flash > 0:
                cv2.putText(frame, "GOAL!", (int(width * 0.35), int(height * 0.18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 8, cv2.LINE_AA)
                goal_flash -= 1

            meta["frames"].append({"frame_idx": frame_idx, "timestamp_s": frame_idx / fps,
                                   "players": players, "ball": balls})
            writer.write(frame)
            total_players += len(players)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    detector.close()
    ball_detector.close()

    meta["static_hotspots"] = [[float(x), float(y)] for (x, y) in static_hotspots]
    json_path.write_text(json.dumps(meta, indent=2))

    processed = frame_idx - start_frame
    mean_players = total_players / processed if processed else 0.0
    click.echo(f"frames: {processed}  mean players/frame: {mean_players:.2f}  "
               f"goals: {len(meta['goal_events'])}  "
               f"static hotspots discovered: {len(static_hotspots)}")
    click.echo(f"video:  {annotated_path}")
    click.echo(f"json:   {json_path}")


if __name__ == "__main__":
    main()
