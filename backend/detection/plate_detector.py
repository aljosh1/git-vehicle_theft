"""
License plate localisation.

Two strategies, chosen automatically:

1. **YOLOv8 plate model** - if `models/license_plate.pt` exists, a fine-tuned
   detector locates plates directly.  This is the accurate path and the one the
   training pipeline in `training/` produces.

2. **Classical OpenCV fallback** - if no plate weights are present, plates are
   found with edge density + contour + aspect-ratio filtering.  Accuracy is
   noticeably lower, but the system runs end to end, which is what makes the
   project demonstrable before the plate model has finished training.

Both return the same `list[Detection]` with `label="license_plate"`, so nothing
downstream knows or cares which path ran.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.detection.base import Detection
from backend.utils.logger import get_logger

log = get_logger(__name__)

try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    from inference_sdk import InferenceHTTPClient

    _ROBOFLOW_SDK_AVAILABLE = True
except ImportError:
    _ROBOFLOW_SDK_AVAILABLE = False

# Real plates are wider than tall. These bounds reject most non-plate contours
# (windows, badges, shadows) before OCR is ever attempted.
_MIN_ASPECT_RATIO = 1.5
_MAX_ASPECT_RATIO = 6.5
_MIN_PLATE_AREA = 400          # px^2; smaller regions carry no readable text
_MAX_PLATE_AREA_FRACTION = 0.3  # a "plate" filling 30% of the crop is wrong


class PlateDetector:
    """Locates license plate regions inside a frame or a vehicle crop."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float | None = None,
        device: str | None = None,
    ):
        self.conf_threshold = conf_threshold or settings.PLATE_CONF_THRESHOLD
        self.device = device or settings.DEVICE
        self.model = None
        self.rf_client = None
        self.rf_model_id = settings.ROBOFLOW_MODEL_ID
        self.mode = "opencv"

        path = settings.resolve_model(
            str(model_path) if model_path else settings.PLATE_MODEL_PATH
        )
        requested_backend = settings.PLATE_DETECTOR_BACKEND

        if requested_backend in {"auto", "yolo"} and _YOLO_AVAILABLE and path.exists():
            try:
                self.model = YOLO(str(path))
                self.model.to(self.device)
                self.mode = "yolo"
                log.info("plate detector ready: YOLOv8 (%s)", path.name)
                return
            except Exception:
                log.exception(
                    "failed to load plate model %s - falling back to OpenCV", path
                )

        if requested_backend in {"auto", "roboflow"}:
            self._init_roboflow()
            if self.mode == "roboflow":
                return

        if requested_backend == "yolo":
            log.warning("plate backend is forced to yolo but model could not be loaded")
        elif requested_backend == "roboflow":
            log.warning("plate backend is forced to roboflow but SDK or credentials are unavailable")
        elif requested_backend == "auto":
            log.warning(
                "no plate model at %s and roboflow unavailable - using the classical OpenCV fallback. "
                "Train one with training/train_plate_detector.py or set Roboflow credentials.",
                path,
            )

    def _init_roboflow(self) -> None:
        if not _ROBOFLOW_SDK_AVAILABLE:
            log.info("inference-sdk not installed; roboflow backend unavailable")
            return
        if not settings.ROBOFLOW_API_KEY.strip():
            log.info("ROBOFLOW_API_KEY not set; roboflow backend unavailable")
            return
        try:
            self.rf_client = InferenceHTTPClient(
                api_url=settings.ROBOFLOW_API_URL,
                api_key=settings.ROBOFLOW_API_KEY,
            )
            self.mode = "roboflow"
            log.info("plate detector ready: Roboflow hosted model (%s)", self.rf_model_id)
        except Exception:
            log.exception("failed to initialise Roboflow client")
            self.rf_client = None

    # ------------------------------------------------------------------ api --
    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Find plate regions.

        Args:
            image: BGR frame, or a vehicle crop (better - fewer false
                positives, since the search space excludes the background).

        Returns:
            Plate detections sorted by confidence descending.
        """
        if image is None or image.size == 0:
            return []
        if self.mode == "yolo":
            return self._detect_yolo(image)
        if self.mode == "roboflow":
            return self._detect_roboflow(image)
        return self._detect_opencv(image)

    # ----------------------------------------------------------- roboflow ----
    def _detect_roboflow(self, image: np.ndarray) -> list[Detection]:
        if self.rf_client is None:
            return []
        try:
            # The SDK accepts an OpenCV BGR ndarray, so live camera frames and
            # decoded upload images use exactly the same hosted inference path.
            result = self.rf_client.infer(image, model_id=self.rf_model_id)
        except Exception:
            log.exception("roboflow inference failed - skipping this frame")
            return []

        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        if not isinstance(predictions, list):
            log.warning("roboflow response has an invalid predictions field")
            return []

        image_height, image_width = image.shape[:2]
        detections: list[Detection] = []
        for item in predictions:
            if not isinstance(item, dict):
                continue
            try:
                conf = float(item.get("confidence", 0.0))
                x = float(item["x"])
                y = float(item["y"])
                width = float(item["width"])
                height = float(item["height"])
            except (KeyError, TypeError, ValueError):
                log.debug("ignoring malformed Roboflow prediction: %r", item)
                continue
            if conf < self.conf_threshold or width <= 0 or height <= 0:
                continue

            x1 = max(0, min(image_width, int(round(x - width / 2.0))))
            y1 = max(0, min(image_height, int(round(y - height / 2.0))))
            x2 = max(0, min(image_width, int(round(x + width / 2.0))))
            y2 = max(0, min(image_height, int(round(y + height / 2.0))))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    label="license_plate",
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    meta={"source": "roboflow", "class": item.get("class")},
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # --------------------------------------------------------------- yolo ----
    def _detect_yolo(self, image: np.ndarray) -> list[Detection]:
        try:
            results = self.model.predict(
                image, conf=self.conf_threshold, verbose=False
            )
        except Exception:
            log.exception("plate model inference failed - skipping this frame")
            return []

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label="license_plate",
                        confidence=float(box.conf[0].item()),
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # -------------------------------------------------------------- opencv ---
    def _detect_opencv(self, image: np.ndarray) -> list[Detection]:
        """
        Classical localisation.

        Plates are high-contrast rectangles with dense vertical edges (the
        characters).  The steps below isolate exactly that signature:

            bilateral filter  - smooth paint/noise but keep character edges
            Sobel x           - emphasise the vertical strokes of glyphs
            Otsu threshold    - binarise without a hand-tuned constant
            close morphology  - merge adjacent characters into one blob
            contour + filter  - keep plate-shaped rectangles only
        """
        try:
            h, w = image.shape[:2]
            frame_area = h * w

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # d=11 with matching sigmas: strong smoothing, edges preserved.
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, binary = cv2.threshold(
                sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            # Wide, short kernel: joins characters horizontally, not lines
            # vertically.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates: list[Detection] = []
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                if area < _MIN_PLATE_AREA or ch == 0:
                    continue
                if area > frame_area * _MAX_PLATE_AREA_FRACTION:
                    continue

                aspect = cw / ch
                if not (_MIN_ASPECT_RATIO <= aspect <= _MAX_ASPECT_RATIO):
                    continue

                # Confidence is heuristic: how plate-like the aspect ratio is,
                # centred on ~3.2:1 which is typical for real plates. Reported
                # honestly as a weak score so downstream code can prefer YOLO
                # results when both are available.
                aspect_score = 1.0 - min(abs(aspect - 3.2) / 3.2, 1.0)
                candidates.append(
                    Detection(
                        label="license_plate",
                        confidence=round(0.30 + 0.35 * aspect_score, 3),
                        x1=x,
                        y1=y,
                        x2=x + cw,
                        y2=y + ch,
                        meta={"source": "opencv_fallback", "aspect": round(aspect, 2)},
                    )
                )

            candidates.sort(key=lambda d: d.confidence, reverse=True)
            # More than three candidates means the heuristic is guessing; the
            # extras are almost always noise.
            return candidates[:3]

        except Exception:
            log.exception("classical plate localisation failed")
            return []
