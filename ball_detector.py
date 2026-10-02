"""YOLO26 + Norfair Kalman tracker for soccer-ball detection & tracking.

Detects COCO class 32 (sports ball) with a large inference size and a low
confidence threshold, then tracks surviving detections across frames. The
Kalman filter keeps the track alive during short YOLO misses (e.g. motion
blur, brief occlusion behind a player), yielding a stable per-ball track_id.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from norfair.filter import OptimizedKalmanFilterFactory


class BallDetector:
    """Detect and track the soccer ball using YOLO26x + a Norfair tracker.

    The public entry point is :meth:`infer_and_track`, which runs YOLO on a
    single BGR frame, filters obvious false positives, and updates the tracker.
    Returned entries mix live detections and short Kalman-predicted extrapolations
    (marked with ``predicted=True``) so downstream code sees a continuous track
    even when the detector momentarily loses the ball.
    """

    SPORTS_BALL_CLASS_ID = 32  # COCO "sports ball"

    # Broadcast footage: the ball is a few pixels wide, so filters are aggressive.
    _MIN_AREA_FRAC = 2e-6   # smaller than this is almost certainly noise
    _MAX_AREA_FRAC = 2e-2   # larger than this is a player torso / logo / net
    _MAX_ASPECT = 1.8       # a ball's bbox is roughly square; reject elongated blobs

    def __init__(
        self,
        weights_path: str | Path = "Modals/yolov26/yolo26x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.40,
        image_size: int = 1920,
    ) -> None:
        """Load the YOLO26 checkpoint and build a Norfair Kalman tracker.

        The Kalman noise trade-off: low ``Q`` (process noise, 0.1) tells the
        filter the ball's motion is smooth frame-to-frame, so its predictions
        stay confident during misses. Moderate ``R`` (measurement noise, 4.0)
        acknowledges YOLO's center is slightly jittery on tiny objects and
        prevents the track from snapping to every noisy detection.
        """
        self._model = YOLO(str(weights_path))
        self._device = device
        self._conf = conf_threshold
        self._imgsz = image_size

        # distance_threshold ~180 px works for 1080p broadcast: the ball can
        # translate a fair distance between frames on a hard pass, but never
        # further than this without leaving frame.
        self._tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=180.0,
            hit_counter_max=8,          # keep predicting for up to 8 missed frames
            initialization_delay=0,     # emit a track as soon as we see the ball
            pointwise_hit_counter_max=8,
            filter_factory=OptimizedKalmanFilterFactory(R=4.0, Q=0.1),
        )

    def infer_and_track(self, frame_bgr: np.ndarray) -> list[dict]:
        """Run detection + tracking on one BGR frame.

        Returns a list of dicts, one per active track this frame:
        ``{'track_id': int, 'bbox_xyxy': [x1,y1,x2,y2],
           'confidence': float, 'predicted': bool}``.
        ``predicted=True`` means the Kalman filter extrapolated this position
        because YOLO did not produce a matching detection on this frame.
        """
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)

        result = self._model.predict(
            frame_bgr,
            classes=[self.SPORTS_BALL_CLASS_ID],
            conf=self._conf,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # Soccer has one ball in play. Sort by confidence, keep the single best
        # detection that also passes shape/size filters. This is the biggest
        # false-positive killer on broadcast footage where YOLO can flag
        # stationary background dots, shoe patches, or ad-board reflections.
        order = np.argsort(-confs)  # descending
        best: Detection | None = None
        for idx in order:
            x1, y1, x2, y2 = boxes[idx]
            c = float(confs[idx])
            if not self._passes_filters(x1, y1, x2, y2, c, frame_area):
                continue
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            best = Detection(
                points=np.array([[cx, cy]], dtype=np.float32),
                scores=np.array([c], dtype=np.float32),
                data={"bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                      "confidence": c},
            )
            break

        detections: list[Detection] = [best] if best is not None else []
        tracked = self._tracker.update(detections=detections)

        out: list[dict] = []
        for t in tracked:
            live = t.last_detection is not None and t.hit_counter == self._tracker.hit_counter_max
            data = t.last_detection.data if t.last_detection is not None else {}
            bbox = data.get("bbox_xyxy")
            conf = float(data.get("confidence", 0.0))
            if bbox is None or not live:
                # Kalman-only frame: synthesize a bbox around the predicted center
                # using the last-known box size (or a small default).
                cx, cy = float(t.estimate[0][0]), float(t.estimate[0][1])
                if bbox is not None:
                    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    bw = bh = 16.0
                bbox = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            out.append({
                "track_id": int(t.id),
                "bbox_xyxy": [float(v) for v in bbox],
                "confidence": conf,
                "predicted": not live,
            })
        return out

    def _passes_filters(self, x1: float, y1: float, x2: float, y2: float,
                        conf: float, frame_area: float) -> bool:
        """Reject detections that can't plausibly be a soccer ball."""
        if conf < self._conf:
            return False
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        area_frac = (w * h) / frame_area
        if area_frac < self._MIN_AREA_FRAC or area_frac > self._MAX_AREA_FRAC:
            return False
        # A ball is roughly circular, so its bounding box should be near-square.
        aspect = max(w / h, h / w)
        if aspect > self._MAX_ASPECT:
            return False
        return True

    def close(self) -> None:
        """Release the model and tracker."""
        del self._model
        del self._tracker
