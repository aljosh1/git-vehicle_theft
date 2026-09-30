"""
Vehicle and person detector using YOLOv8.

Wraps Ultralytics YOLO and filters COCO predictions to the four vehicle classes
(car, motorcycle, bus, truck) plus person. The specification requires all five.

Usage:
    detector = VehicleDetector(model_path="models/yolov8n.pt")
    detections = detector.detect(frame)

Each detection carries label, confidence, bbox, and optionally a track_id when
the built-in tracker is enabled.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.detection.base import (
    ANIMAL_CLASS_IDS,
    COCO_PERSON,
    VEHICLE_CLASS_IDS,
    Detection,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)

# Only import ultralytics when the module is actually used, so the server can
# still boot when it is missing.
try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    log.warning("ultralytics not installed - vehicle detection unavailable")


class VehicleDetector:
    """
    YOLOv8 vehicle and person detector.

    Returns only COCO classes 2/3/5/7 (car, motorcycle, bus, truck) and 0
    (person), filtered by `conf_threshold`.  Everything else (traffic lights,
    backpacks, umbrellas, …) is silently dropped.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float | None = None,
        device: str | None = None,
        enable_tracking: bool = False,
    ):
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed; run: pip install ultralytics"
            )

        self.model_path = (
            settings.resolve_model(model_path)
            if model_path
            else settings.resolve_model(settings.VEHICLE_MODEL_PATH)
        )
        self.conf_threshold = conf_threshold or settings.VEHICLE_CONF_THRESHOLD
        self.device = device or settings.DEVICE
        self.enable_tracking = enable_tracking

        if not self.model_path.exists():
            log.warning(
                "model weights not found at %s - downloading yolov8n.pt from "
                "Ultralytics (this happens once and takes ~10 seconds)",
                self.model_path,
            )
            self.model_path = Path("yolov8n.pt")

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        log.info(
            "vehicle detector ready: %s on %s (conf >= %.2f)",
            self.model_path.name,
            self.device,
            self.conf_threshold,
        )

        # Warmup: the first inference is 3–5× slower because of CUDA init /
        # model graph compilation.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.detect(dummy)
        log.debug("warmup inference complete")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on one frame.

        Args:
            frame: BGR image from OpenCV, any resolution.

        Returns:
            Detections for vehicles (car/motorcycle/bus/truck) and persons,
            sorted by confidence descending.
        """
        if self.enable_tracking:
            results = self.model.track(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
        else:
            results = self.model.predict(
                frame, conf=self.conf_threshold, verbose=False
            )

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Keep vehicles and persons only; discard everything else.
                if class_id == COCO_PERSON:
                    label = "person"
                elif class_id in VEHICLE_CLASS_IDS:
                    label = VEHICLE_CLASS_IDS[class_id]
                elif class_id in ANIMAL_CLASS_IDS:
                    label = ANIMAL_CLASS_IDS[class_id]
                else:
                    continue

                track_id = int(box.id[0].item()) if box.id is not None else None

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        class_id=class_id,
                        track_id=track_id,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes + labels on a copy of the frame.

        Returns:
            Annotated frame (BGR, same size as input).
        """
        annotated = frame.copy()
        for det in detections:
            if det.is_vehicle:
                color = (0, 255, 0)
            elif det.is_person:
                color = (255, 100, 0)
            elif det.is_animal:
                color = (0, 215, 255)
            else:
                color = (0, 215, 255)
            cv2.rectangle(
                annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2
            )
            label_text = f"{det.label} {det.confidence:.2f}"
            if det.track_id is not None:
                label_text += f" #{det.track_id}"

            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (det.x1, det.y1 - th - 6),
                (det.x1 + tw + 4, det.y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated
