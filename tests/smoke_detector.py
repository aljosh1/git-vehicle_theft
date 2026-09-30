"""Manual smoke test for the YOLOv8 vehicle detector.

    python tests/smoke_detector.py

Verifies the model loads, inference runs, and annotation works. Uses a
synthetic frame, so zero detections is a correct result - what is being
checked is that the pipeline executes without error and how fast it is.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.detection.vehicle_detector import VehicleDetector  # noqa: E402


def main() -> None:
    detector = VehicleDetector()

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    timings = []
    for _ in range(5):
        start = time.perf_counter()
        detections = detector.detect(frame)
        timings.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(timings) / len(timings)
    print(f"inference   : {mean_ms:.1f} ms mean over 5 runs")
    print(f"throughput  : {1000 / mean_ms:.1f} FPS")
    print(f"detections  : {len(detections)} (0 is expected on noise)")

    annotated = detector.annotate(frame, detections)
    assert annotated.shape == frame.shape
    print(f"annotate    : OK {annotated.shape}")
    print("\nsmoke test PASSED")


if __name__ == "__main__":
    main()
