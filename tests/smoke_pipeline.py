"""
End-to-end integration test.

    python tests/smoke_pipeline.py

Generates a short synthetic video, runs the real pipeline over it, and verifies
that frames are processed, detections are logged and the threat engine scores
them. This is the "integrate all modules" check: it exercises capture →
detection → OCR → face → scoring → database in one pass, which no unit test
does.

A synthetic clip contains no real cars, so zero detections is a correct result.
What is being proven is that the whole chain executes without error and reports
honest throughput - not that YOLOv8 can find a car in random noise.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use a scratch database so a demonstration run never pollutes data/vtds.db.
_TMP_DB = Path(tempfile.gettempdir()) / "vtds_pipeline_smoke.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from backend.config import settings  # noqa: E402
from backend.core.pipeline import Pipeline  # noqa: E402
from backend.database import crud  # noqa: E402
from backend.database.base import init_db, session_scope  # noqa: E402
from backend.utils.logger import setup_logging  # noqa: E402

FRAME_COUNT = 40
WIDTH, HEIGHT = 640, 480


def make_test_video(path: Path) -> None:
    """
    Write a synthetic clip: a grey scene with a moving rectangle.

    Deliberately not random noise per frame - a moving solid shape gives the
    tracker something coherent to follow, so the track-id plumbing is exercised
    rather than skipped.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a VideoWriter (missing codec?)")

    for index in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        x = 40 + index * 12
        cv2.rectangle(frame, (x, 240), (x + 150, 350), (70, 70, 180), -1)
        cv2.rectangle(frame, (x + 30, 320), (x + 110, 342), (235, 235, 235), -1)
        cv2.putText(
            frame, "ABC123XY", (x + 36, 338),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def main() -> None:
    setup_logging("WARNING")     # keep the output readable
    init_db()

    print("=== integration smoke test ===\n")

    video = Path(tempfile.gettempdir()) / "vtds_smoke.mp4"
    print(f"[1/5] generating a {FRAME_COUNT}-frame test video")
    make_test_video(video)
    print(f"      {video} ({video.stat().st_size / 1024:.0f} KB)")

    print("[2/5] registering a vehicle whose plate appears in the clip")
    with session_scope() as db:
        from backend.database import schemas

        owner = crud.get_user_by_email(db, "smoke@test.example.com")
        if owner is None:
            owner = crud.create_user(
                db,
                schemas.UserCreate(
                    email="smoke@test.example.com",
                    password="Smoke@12345",
                    full_name="Smoke Test Owner",
                ),
            )
        if crud.get_vehicle_by_plate(db, "ABC123XY") is None:
            crud.create_vehicle(
                db,
                schemas.VehicleCreate(
                    owner_id=owner.id, plate_display="ABC-123XY", make="Toyota"
                ),
            )
    print("      ABC-123XY registered")

    print("[3/5] starting the pipeline over the clip")
    # Alerts disabled: this is a functional check, not a notification test, and
    # nobody wants a smoke test emailing the security desk.
    pipeline = Pipeline(
        camera_id="smoke-cam", source=str(video), enable_tracking=True, enable_alerts=False
    )
    pipeline.start()

    deadline = time.time() + 120
    # Wait for the pipeline to finish the clip. The capture thread clears
    # `running` at end-of-file, so that flag - not a frame count - is the
    # completion signal: the grabber drains a local file far faster than
    # inference consumes it, so most frames are legitimately dropped by the
    # keep-newest queue and the processed count is always lower than the total.
    while time.time() < deadline and pipeline.running:
        time.sleep(0.5)

    frames = pipeline._frame_count
    _, result = pipeline.get_latest_frame()
    pipeline.stop()

    print(f"      processed {frames} frames")
    if result is not None:
        print(f"      last frame : {result.processing_ms:.0f} ms  ({result.fps:.1f} FPS)")
        print(f"      vehicles   : {len(result.vehicles)}")
        print(f"      persons    : {len(result.persons)}")
        print(f"      plate      : {result.plate_text or 'not read'}")
        print(f"      threat     : {result.threat_score}/100 ({result.threat_level_name})")

    print("[4/5] checking the detection log")
    with session_scope() as db:
        logs = crud.list_detection_logs(db, camera_id="smoke-cam", limit=500)
        stats = crud.dashboard_stats(db)
    print(f"      {len(logs)} detection rows written")
    print(f"      {stats.total_vehicles} vehicles, {stats.total_detections} detections")

    print("[5/5] verifying component wiring")
    checks = {
        "frames processed": frames > 0,
        "vehicle detector loaded": pipeline.vehicle_detector.model is not None,
        "plate detector ready": pipeline.plate_detector.mode in {"yolo", "roboflow", "opencv"},
        "threat engine responded": result is not None,
        "database reachable": stats.total_vehicles > 0,
    }
    for name, passed in checks.items():
        print(f"      {'PASS' if passed else 'FAIL'}  {name}")

    video.unlink(missing_ok=True)

    if all(checks.values()):
        print("\nintegration smoke test PASSED")
        print(
            f"\nnote: measured {result.fps:.1f} FPS on {settings.DEVICE}. "
            f"The brief targets {settings.TARGET_FPS} FPS - see README.md."
        )
    else:
        print("\nintegration smoke test FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
