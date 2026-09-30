"""Focused tests for hosted Roboflow plate detection integration."""

from __future__ import annotations

import numpy as np

from backend.core import pipeline as pipeline_module
from backend.core.pipeline import Pipeline
from backend.detection.base import Detection
from backend.detection.plate_detector import PlateDetector


class FakeRoboflowClient:
    def __init__(self, result: object):
        self.result = result
        self.calls: list[tuple[np.ndarray, str]] = []

    def infer(self, image: np.ndarray, *, model_id: str) -> object:
        self.calls.append((image, model_id))
        return self.result


def roboflow_detector(result: object, threshold: float = 0.35) -> PlateDetector:
    detector = PlateDetector.__new__(PlateDetector)
    detector.conf_threshold = threshold
    detector.device = "cpu"
    detector.model = None
    detector.rf_client = FakeRoboflowClient(result)
    detector.rf_model_id = "vehicle-license-plate-1hdcy/1"
    detector.mode = "roboflow"
    return detector


def test_roboflow_predictions_are_filtered_clipped_and_sorted() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = roboflow_detector(
        {
            "predictions": [
                {
                    "x": 195,
                    "y": 50,
                    "width": 30,
                    "height": 20,
                    "confidence": 0.82,
                    "class": "plate",
                },
                {
                    "x": 60,
                    "y": 40,
                    "width": 40,
                    "height": 10,
                    "confidence": 0.91,
                    "class": "license-plate",
                },
                {
                    "x": 20,
                    "y": 20,
                    "width": 10,
                    "height": 5,
                    "confidence": 0.2,
                },
                {"confidence": 0.99},
            ]
        }
    )

    detections = detector.detect(image)

    assert [item.confidence for item in detections] == [0.91, 0.82]
    assert detections[0].bbox == (40, 35, 80, 45)
    assert detections[1].bbox == (180, 40, 200, 60)
    assert detections[0].meta == {
        "source": "roboflow",
        "class": "license-plate",
    }
    assert detector.rf_client.calls == [(image, "vehicle-license-plate-1hdcy/1")]


def test_roboflow_invalid_predictions_field_returns_no_detections() -> None:
    detector = roboflow_detector({"predictions": {"not": "a list"}})

    assert detector.detect(np.zeros((20, 20, 3), dtype=np.uint8)) == []


def test_pipeline_translates_plate_box_to_full_frame_without_ocr(monkeypatch) -> None:
    local_plate = Detection(
        label="license_plate",
        confidence=0.88,
        x1=10,
        y1=20,
        x2=70,
        y2=40,
        meta={"source": "roboflow", "class": "plate"},
    )

    class FakePlateDetector:
        def detect(self, image: np.ndarray) -> list[Detection]:
            assert image.shape == (100, 200, 3)
            return [local_plate]

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.plate_detector = FakePlateDetector()
    pipeline._plate_cache = {}
    monkeypatch.setattr(pipeline_module, "read_plate", lambda image: None)

    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    vehicle = Detection(
        label="car",
        confidence=0.95,
        x1=100,
        y1=80,
        x2=300,
        y2=180,
        track_id=7,
    )

    text, confidence, matched_id, in_database, stolen, owner_id, plates = (
        pipeline._recognise_plate(frame, vehicle)
    )

    assert (text, confidence, matched_id, in_database, stolen, owner_id) == (
        None,
        None,
        None,
        None,
        False,
        None,
    )
    assert len(plates) == 1
    assert plates[0].bbox == (110, 100, 170, 120)
    assert plates[0].track_id == 7
    assert plates[0].meta["source"] == "roboflow"
    assert plates[0].meta["vehicle_track_id"] == 7
