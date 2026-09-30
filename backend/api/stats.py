"""
Dashboard statistics.

    GET /api/stats            headline counters
    GET /api/stats/timeseries per-day detections and alerts
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.get("", response_model=schemas.DashboardStats)
def dashboard_stats(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Headline figures for the dashboard tiles.

    Live pipeline state (running / current FPS) is merged in from the pipeline
    registry, so the dashboard needs one request rather than two.
    """
    stats = crud.dashboard_stats(db)

    try:
        from backend.core.pipeline import _ACTIVE_PIPELINES

        running = [p for p in _ACTIVE_PIPELINES.values() if p.running]
        stats.pipeline_running = bool(running)
        if running:
            _, result = running[0].get_latest_frame()
            stats.current_fps = round(result.fps, 1) if result else 0.0
    except Exception:
        log.exception("could not read live pipeline state")

    return stats


@router.get("/timeseries", response_model=list[schemas.TimeSeriesPoint])
def detections_timeseries(
    days: int = Query(7, ge=1, le=90),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return crud.detections_timeseries(db, days=days)
