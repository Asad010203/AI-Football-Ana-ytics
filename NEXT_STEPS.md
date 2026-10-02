# Next Steps — realistic roadmap

Written 2026-09-19 after reading [run.py](run.py), [player_detector.py](player_detector.py), [ball_detector.py](ball_detector.py), and [CONTEXT.md](CONTEXT.md).

## Where the project actually stands today

Three detection pieces are wired up and running end-to-end. None of them is "done":

| Piece | Status | What's missing to call it "final" |
| --- | --- | --- |
| **Person detection** | YOLO11-x on class 0, per-frame only. Works. | No tracking, no team ID, no jersey number, no goalkeeper/referee split, no re-ID across occlusions. |
| **Ball detection** | YOLO26-x + Norfair Kalman, top-1 per frame, static-hotspot rejection. Works. | Still loses the ball on hard shots (Kalman `distance_threshold=180` at 4K can't associate a 200+ px jump). Confuses shoe patches / ad-board dots until the hotspot filter catches them. |
| **Goal detection** | Iteration 6 — trajectory crossing + altitude + hotspot + cooldown. Known to work on one test video. | Residual 2D depth ambiguity (ball flying above the crossbar can still fire). One video is not a test set. Fires haven't been graded against ground truth on multiple matches. |

The pipeline is a working prototype, not a shipping product. Everything below is what would move it forward — grouped by cost so the trade-offs are visible.

---

## Right now — small, concrete, high value (hours to a day)

These are the changes you could sit down and do today. Each one is well-scoped and doesn't need research.

### 1. Player tracking (unlocks everything downstream)
Currently every frame's players are independent — no IDs, no history. Add `model.track(..., persist=True, tracker="bytetrack.yaml")` (Ultralytics has ByteTrack built in) or mirror the Norfair setup from `ball_detector.py`. Once players have stable IDs, distance run, speed, and heatmaps become one-liners.

**Cost:** ~half a day. **Payoff:** every stat downstream depends on this.

### 2. Team classification by jersey color
For each player bbox, take the torso ROI (middle vertical third), average the dominant color in HSV, run k-means with k=3 (team A, team B, referee) over a rolling window of frames. Stable, no ML training.

**Cost:** a day. **Payoff:** possession, team formations, pass network — all need this.

### 3. Velocity-based goal confirmation
Already listed as CONTEXT.md future work #1. Compute mean ball speed in the ~10 frames before vs after the "crossing." A real goal decelerates hard (net); a flyover keeps flying. Cheap post-filter on the fires that already survive.

**Cost:** a few hours. **Payoff:** should kill most residual 2D-depth flyover false positives.

### 4. Ball-size sanity check
CONTEXT.md future work #2. From the 4 goal corners, expected ball radius at goal distance is derivable. Reject fires where the trigger detection is >2× or <0.5× the expected pixel size.

**Cost:** a few hours. **Payoff:** cheap extra false-positive filter.

### 5. Fix `_segment_enters_polygon` entry point when `curr` is already inside
[run.py:94](run.py#L94) — when `curr_pt` is inside, the function returns `curr_pt` as the entry point, but the *actual* entry is somewhere on the segment. This under-reports the entry x, which feeds into the altitude check and can bias it. Cheap fix: still sample the segment in the "curr inside" branch to find the first interior sample from `prev` outward.

**Cost:** ~30 min. **Payoff:** more accurate altitude check on fast shots.

### 6. Codec / output size
[run.py:357](run.py#L357) uses `mp4v` for a 4K annotated video → files are huge and unscrubbable. Try `avc1` (H.264) first, fall back to `mp4v`. Optionally add `--preview-scale 0.5` to downscale the annotated output.

**Cost:** ~30 min. **Payoff:** UX only, but real UX pain.

### 7. Graceful CPU fallback
Both detectors hard-code `device="cuda"`. On a machine without CUDA the whole pipeline crashes. `device = "cuda" if torch.cuda.is_available() else "cpu"` is one line and makes the code portable enough to at least *run* elsewhere (even if slowly).

**Cost:** ~10 min.

### 8. Move tuning constants to CLI or a config file
`_RECENT_WINDOW`, `_MIN_TRIGGER_CONF`, `_GOAL_COOLDOWN`, etc. are module-level in `run.py`. Sweeping them today means editing source. Expose as CLI flags (defaults unchanged) — makes tuning across videos painless.

**Cost:** a couple of hours.

### 9. Tests for the pure-geometry helpers
`_segment_enters_polygon` and `_passes_altitude_check` are pure functions with no I/O. A handful of pytest cases (ball far above, ball at goal, both endpoints outside but transiting, prev inside) would catch regressions before they burn a full re-run of a 5-minute video.

**Cost:** a couple of hours.

---

## Near-term — meaningful features (a few days each)

These need real work but are within reach and are the natural next milestones.

### 10. Pitch homography + mini-map
Add a second one-time picker for 4 known pitch-line intersections (halfway line + centre circle tangents, or corners + penalty box). Compute a homography to a canonical top-down pitch. Project every player and the ball onto a mini-map overlay. This is the single feature that turns "we drew boxes on a video" into "we have analytics data."

**Cost:** 2–3 days. **Depends on:** player tracking (#1).

### 11. Possession % over time
Nearest-player-to-ball per frame, resolved to a team by #2. Time-weighted rollup by team. Emit a small chart alongside `results.json`.

**Cost:** 1 day. **Depends on:** #1 + #2.

### 12. Broadcast scene-change reset
Frame-to-frame histogram diff > threshold → reset `recent_inside`, `prev_real_pos`, `last_goal_frame`, and optionally the ball tracker. Makes the pipeline safe on broadcast footage with camera cuts. CONTEXT.md limitations already flags this.

**Cost:** 1–2 days.

### 13. Ground-truth harness for goal detection
One video isn't enough. Label 5–10 clips with (frame_idx, is_goal) tuples, add a `scripts/eval_goals.py` that runs the pipeline and reports precision/recall against the labels. Without this, every "improvement" to the goal rule is a guess.

**Cost:** 1 day of code + however long labelling takes. **Payoff:** every future goal-rule tweak becomes measurable.

---

## Later — larger initiatives (weeks, or research-y)

These are on the horizon but not what to start with.

- **Camera calibration** — `cv2.solvePnP` on the 4 goal corners with real-world dimensions → full camera pose → proper 3D ball ray → ground-plane 3D position. Fixes the 2D-depth ambiguity properly. CONTEXT.md future work #3.
- **Ball-visible-in-net cue** — CONTEXT.md future work #5. High precision, adds latency.
- **Player re-ID across long occlusions** — appearance embeddings (e.g. OSNet). Big lift for the "who ran how far" number.
- **Event detection beyond goals** — passes, tackles, fouls, offside. Each is its own research problem; not to be underestimated.
- **Multi-camera fusion** — only meaningful if there are multiple camera feeds.

---

## Recommendation

If it were up to me, I'd do these three next, in this order:

1. **#1 Player tracking** — cheapest, single biggest force-multiplier for the rest of the roadmap.
2. **#3 Velocity-based goal confirmation + #4 Ball-size sanity** — a half-day combined; kills the biggest known goal false-positive class.
3. **#13 Ground-truth harness** — so #3 and every future goal-rule tweak stops being a vibe check.

After that, #2 (team ID) + #10 (mini-map) is where this becomes a real analytics tool rather than a detection demo.

## What I'd avoid starting on now

- Camera calibration and multi-camera work — high effort, only pays off after the analytics layer exists.
- Event detection (passes, tackles, offside) — each is a research project on its own; the current codebase has no foundation for them yet.
- Rewriting the goal rule again without a ground-truth harness first — you'll just re-play the iteration-1-through-6 loop with new failure modes.
