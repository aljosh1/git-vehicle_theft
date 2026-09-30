"""
Detection log queries.

    GET /api/detections          recent detections, filterable
    GET /api/detections/live     the newest frame's detections as JSON

`/live` is what the Live Monitoring page polls for its side panel: the MJPEG
stream carries the pixels, this carries the structured results that go beside
it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.get("", response_model=list[schemas.DetectionLogRead])
def list_detections(
    camera_id: str | None = None,
    hours: int | None = Query(None, ge=1, le=24 * 30),
    min_threat: int = Query(0, ge=0, le=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    return crud.list_detection_logs(
        db,
        camera_id=camera_id,
        since=since,
        min_threat=min_threat,
        skip=skip,
        limit=limit,
    )


@router.get("/live")
def live_detections(
    camera_id: str = Query("cam-0"),
    user: User = Depends(get_current_user),
):
    """
    The most recent frame's detections, straight from memory.

    Reads the pipeline's last result rather than the database, so the panel
    stays in step with the video even under heavy write load.
    """
    from backend.core.pipeline import get_pipeline

    pipeline = get_pipeline(camera_id)
    if pipeline is None or not pipeline.running:
        return {"running": False, "detections": [], "threat": None}

    _, result = pipeline.get_latest_frame()
    if result is None:
        return {"running": True, "detections": [], "threat": None}

    return {
        "running": True,
        "camera_id": camera_id,
        "frame_number": result.frame_number,
        "fps": round(result.fps, 1),
        "processing_ms": round(result.processing_ms, 1),
        "detections": [
            d.to_dict()
            for d in (result.vehicles + result.persons + result.animals + result.plates)
        ],
        "vehicle_count": len(result.vehicles),
        "person_count": len(result.persons),
        "animal_count": len(result.animals),
        "vehicle_analyses": [item.to_dict() for item in result.vehicle_analyses],
        "plate": {
            "text": result.plate_text,
            "confidence": result.plate_confidence,
            "in_database": result.plate_in_database,
            "flagged_stolen": result.vehicle_flagged_stolen,
        },
        "person_status": (
            result.person_status.value if result.person_status else None
        ),
        "face_similarity": result.face_similarity,
        "matched_user_id": result.matched_user_id,
        "expected_owner_id": result.expected_owner_id,
        "owner_verification": (
            result.owner_verification.value if result.owner_verification else None
        ),
        "threat": {
            "score": result.threat_score,
            "level": result.threat_level_name,
            "reason": result.threat_reason,
        },
    }
