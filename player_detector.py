"""YOLO11 wrapper for football player detection. Class 0 (person) only."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from ultralytics import YOLO


class PlayerDetector:
    PERSON_CLASS_ID = 0  # COCO person

    def __init__(
        self,
        weights_path: str | Path = "Modals/yolov11/yolo11x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        image_size: int = 1280,
    ) -> None:
        self._model = YOLO(str(weights_path))
        self._device = device
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._imgsz = image_size

    def infer(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Return {'boxes_xyxy': (N,4), 'confidences': (N,), 'class_ids': (N,)}."""
        result = self._model.predict(
            frame_bgr,
            classes=[self.PERSON_CLASS_ID],
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )[0]
        return {
            "boxes_xyxy": result.boxes.xyxy.cpu().numpy(),
            "confidences": result.boxes.conf.cpu().numpy(),
            "class_ids": result.boxes.cls.cpu().numpy().astype(np.int32),
        }

    def close(self) -> None:
        del self._model


if __name__ == "__main__":
    d = PlayerDetector()
    dummy = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    out = d.infer(dummy)
    print("boxes:", out["boxes_xyxy"].shape,
          "conf:", out["confidences"].shape,
          "cls:", out["class_ids"].shape)
    d.close()
    print("[ok]")
