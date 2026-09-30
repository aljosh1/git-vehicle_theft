"""
Threat detection - weapons and face-concealing masks.

This is the module that turns "a person is near a car" into "an *armed* person
is at the car", which is the difference between logging an incident and
dispatching one.

Two independent signals, because either alone is a gap
--------------------------------------------------------

1. **Weapons as objects.** A purpose-trained model (``THREAT_MODEL_PATH``) is
   used when present, giving `gun`, `knife`, `machete`, `baton` and so on. When
   no such weights are installed the detector falls back to the COCO `knife`
   class that ships inside the stock ``yolov8n.pt``, so the system is never
   silently blind - it is simply less sensitive, and says so in ``/api/info``.

2. **Face occlusion, as a mask proxy.** A balaclava is not a "mask" object to
   most detectors - it is simply a face that cannot be matched. But the moment
   a face is *covered*, the face recogniser returns UNKNOWN for a reason that is
   not bad lighting: too little of the face is visible. Measuring how much of
   the detected face region is usable turns that failure into a positive
   "identity concealed" signal. It needs no extra weights, so a stock install
   still detects a masked intruder.

The honest limitation
---------------------
A mask detector that has not been trained cannot be conjured from thresholds.
The occlusion path here is a *conservative* signal, and a face that is merely
dark, blurred or turned away can produce a low visibility score. It is therefore
reported as a distinct, lower-weight trigger (``face_occluded``) rather than
being folded into a confident ``weapon`` verdict, so the alerting maths stays
honest about how the conclusion was reached. A dedicated mask model, when
installed, replaces it and reports a real detection.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.detection.base import (
    COCO_KNIFE,
    MASK_LABELS,
    THREATENING_CLASS_NAMES,
    WEAPON_LABELS,
    Detection,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)

# Where a face recogniser's answer stops being trustworthy. YuNet reports a
# score in the final column; a low one means a poor, partial or covered face.
_MIN_FACE_DETECTION_SCORE = 0.75


def _normalise_class_name(name: str) -> str:
    """
    Normalise a model's class label to the shared threat vocabulary.

    Purpose-trained models are inconsistent: one emits `Machete`, another
    `machete_knife`, a third `Weapon`. Without normalisation the theft engine's
    allow-lists silently miss real detections, which is a failure that looks
    exactly like "the detector does not work".
    """
    cleaned = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned.endswith("s") and cleaned[:-1] in WEAPON_LABELS | THREATENING_CLASS_NAMES | MASK_LABELS:
        cleaned = cleaned[:-1]
    return cleaned


def _is_weapon_label(name: str) -> bool:
    return _normalise_class_name(name) in WEAPON_LABELS


def _is_mask_label(name: str) -> bool:
    return _normalise_class_name(name) in MASK_LABELS


class ThreatDetector:
    """
    Detects weapons and masks in a frame.

    Args:
        weapon_model_path: a purpose-trained weapons model. When None the
            configured ``THREAT_MODEL_PATH`` is used; when that is also empty the
            detector reports `using_coco_fallback` and relies on the COCO
            `knife` class from the main detector instead of loading a second
            network.
        fallback_detector: the main `VehicleDetector`, consulted for the COCO
            knife class when no purpose-trained model is available.
    """

    def __init__(
        self,
        weapon_model_path: str | Path | None = None,
        *,
        device: str | None = None,
        fallback_detector=None,
    ):
        self.device = device or settings.DEVICE
        self.conf_threshold = settings.THREAT_CONF_THRESHOLD
        self.fallback_detector = fallback_detector
        self.model = None
        self.using_coco_fallback = True

        configured = weapon_model_path or settings.THREAT_MODEL_PATH
        if configured:
            path = settings.resolve_model(str(configured))
            if path.exists():
                self._load(path)
            else:
                log.warning(
                    "THREAT_MODEL_PATH is set to %s but no such file exists - "
                    "falling back to the COCO knife class. Weapon and mask "
                    "coverage will be limited until the weights are installed.",
                    path,
                )
        else:
            log.info(
                "no threat model configured - weapons limited to the COCO "
                "'knife' class and masks detected by face occlusion only. "
                "Set THREAT_MODEL_PATH for full gun/knife/mask coverage."
            )

    def _load(self, path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics is not installed; threat detection unavailable")
            return

        try:
            self.model = YOLO(str(path))
            self.model.to(self.device)
            self.using_coco_fallback = False
            log.info(
                "threat detector ready: %s on %s (conf >= %.2f)",
                path.name, self.device, self.conf_threshold,
            )
        except Exception:
            log.exception("could not load the threat model at %s", path)

    @property
    def available(self) -> bool:
        """Is any weapon signal possible at all?"""
        return self.model is not None or self.fallback_detector is not None

    def detect(
        self, frame: np.ndarray, main_detections: list[Detection] | None = None
    ) -> list[Detection]:
        """
        Find weapons and masks.

        Args:
            frame: BGR image.
            main_detections: detections from the main vehicle/person detector,
                used to pick up its COCO `knife` predictions. The knife box is
                only emitted if the main detector produced one - re-running the
                whole COCO network a second time just to recover a class the
                first pass already saw would double inference cost for nothing.

        Returns:
            `Detection` objects with `label` in the weapon/mask vocabulary. Each
            carries `meta["source"]` so an operator can see *how* it was found -
            a real model detection and an occlusion inference are not equally
            trustworthy and must not be presented as though they were.
        """
        if not settings.THREAT_DETECTION_ENABLED or frame is None or frame.size == 0:
            return []

        found: list[Detection] = []

        if self.model is not None:
            found.extend(self._detect_with_model(frame))
        elif main_detections:
            found.extend(self._coco_knife_fallback(main_detections))

        if settings.FACE_OCCLUSION_ENABLED:
            found.extend(self._detect_occluded_faces(frame))

        return found

    # ------------------------------------------------------------------
    # Weapons
    # ------------------------------------------------------------------
    def _detect_with_model(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )

        found: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            for box in boxes:
                class_id = int(box.cls[0].item())
                raw_name = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else str(names[class_id])
                )
                name = _normalise_class_name(raw_name)

                # Ignore the model's own person/vehicle classes - the main
                # detector already handles those, and duplicating them would
                # double-count persons in the theft engine.
                if not (_is_weapon_label(name) or _is_mask_label(name)):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                found.append(
                    Detection(
                        label=name,
                        confidence=float(box.conf[0].item()),
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        class_id=class_id,
                        meta={"source": "threat_model", "raw_class": raw_name},
                    )
                )
        return found

    def _coco_knife_fallback(self, detections: list[Detection]) -> list[Detection]:
        """
        Recover knives the main detector already found.

        The stock `yolov8n.pt` predicts COCO class 79 (`knife`) but the vehicle
        detector's class filter discards it. Rather than run the network twice,
        the prediction is re-requested from the main detector's own results.
        """
        found: list[Detection] = []
        for det in detections:
            if det.class_id == COCO_KNIFE or _is_weapon_label(det.label):
                found.append(
                    Detection(
                        label=_normalise_class_name(det.label) or "knife",
                        confidence=det.confidence,
                        x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                        class_id=det.class_id,
                        meta={"source": "coco_knife_fallback"},
                    )
                )
        return found

    # ------------------------------------------------------------------
    # Masks, via face occlusion
    # ------------------------------------------------------------------
    def _detect_occluded_faces(self, frame: np.ndarray) -> list[Detection]:
        """
        Infer a concealed face from how well the detector could see it.

        A balaclava or mask leaves a face region that the detector *finds* but
        scores poorly - it is looking at cloth, not skin. An uncovered face
        scores high even in poor light, because the features are still there.
        So a low detection score is the available evidence of coverage, and it
        needs no extra weights.

        The result is labelled `face_occluded`, not `mask`, and is scored lower
        in the theft engine than a real mask detection. That distinction is the
        point: a dark corridor or a face turned away can also score poorly, and
        presenting that as a confident "mask detected" would be a claim the
        system has not earned. A dedicated mask model, when installed, supersedes
        this path entirely.
        """
        from backend.recognition import face_recognizer as face

        backend_name = face.init_backend()
        if backend_name is None or not str(backend_name).startswith("opencv-yunet"):
            # Only the YuNet path exposes a per-face detection score. Other
            # backends report a face or not, and inventing a visibility number
            # from nothing would be a guess dressed as a measurement.
            return []

        rows = face._detect_rows(frame)
        if rows is None or len(rows) == 0:
            return []

        height, width = frame.shape[:2]
        found: list[Detection] = []
        for row in rows:
            score = float(row[-1])
            if score >= _MIN_FACE_DETECTION_SCORE:
                continue        # a confidently detected face is visible

            x, y, w, h = (int(v) for v in row[:4])
            if w <= 0 or h <= 0:
                continue

            # Confidence in the *inference*, not in the detection: a score of
            # 0.3 is a stronger hint than one of 0.61.
            confidence = 1.0 - (score / _MIN_FACE_DETECTION_SCORE)

            found.append(
                Detection(
                    label="face_occluded",
                    confidence=round(max(0.0, min(1.0, confidence)), 3),
                    x1=max(0, x), y1=max(0, y),
                    x2=min(width, x + w), y2=min(height, y + h),
                    meta={
                        "source": "face_occlusion",
                        "detector_score": round(score, 3),
                        "inferred": True,
                    },
                )
            )
        return found

    # ------------------------------------------------------------------
    def annotate(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """
        Draw threat overlays.

        Weapons are outlined in red and masks in magenta, both with a hatched
        feel from a double border, so they cannot be mistaken for the blue
        person boxes the main detector draws.
        """
        annotated = frame.copy()
        for det in detections:
            if det.is_weapon:
                colour = (0, 0, 255)          # red
                thickness = 4
            elif det.label == "face_occluded":
                colour = (255, 0, 255)        # magenta
                thickness = 3
            else:
                colour = (0, 165, 255)        # orange: mask, or a threatening tool
                thickness = 3

            cv2.rectangle(
                annotated, (det.x1, det.y1), (det.x2, det.y2), colour, thickness
            )
            # A second inset line makes the threat read clearly even in a
            # thumbnail on a phone.
            if det.x2 - det.x1 > 12 and det.y2 - det.y1 > 12:
                inset = 4
                cv2.rectangle(
                    annotated,
                    (det.x1 + inset, det.y1 + inset),
                    (det.x2 - inset, det.y2 - inset),
                    colour, 1,
                )

            suffix = ""
            if det.meta.get("inferred"):
                # Mark the inference as such on the overlay. A guard must never
                # be told "MASK" when the system only inferred occlusion.
                suffix = " (inferred)"

            text = f"!! {det.label.upper()} {det.confidence:.2f}{suffix}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            top = max(0, det.y1 - th - 8)
            cv2.rectangle(
                annotated, (det.x1, top), (det.x1 + tw + 8, top + th + 8), colour, -1
            )
            cv2.putText(
                annotated, text, (det.x1 + 4, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return annotated


def describe_threat_capability(threat_detector: ThreatDetector) -> dict[str, object]:
    """
    Report what threat detection can actually do right now.

    Surfaced through `/api/info` so a demonstration never claims gun detection
    when only a knife class is loaded.
    """
    return {
        "enabled": settings.THREAT_DETECTION_ENABLED,
        "weapons_model": "coco_knife_only" if threat_detector.using_coco_fallback else "threat_model",
        "gun_detection": not threat_detector.using_coco_fallback,
        "mask_model": not threat_detector.using_coco_fallback,
        "mask_inference": settings.FACE_OCCLUSION_ENABLED,
        "confidence": threat_detector.conf_threshold,
    }
