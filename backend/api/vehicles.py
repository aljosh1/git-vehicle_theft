"""
Vehicle management.

    GET    /api/vehicles              list (owners see only their own)
    POST   /api/vehicles              register
    GET    /api/vehicles/{id}
    PATCH  /api/vehicles/{id}
    DELETE /api/vehicles/{id}
    POST   /api/vehicles/{id}/image   upload a vehicle photo

Ownership is enforced on every single-vehicle route: an owner may only see and
edit their own vehicles, while an administrator sees everything.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.config import settings
from backend.core.security import random_filename
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User, UserRole, Vehicle
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _authorise(vehicle: Vehicle | None, user: User) -> Vehicle:
    """404 for missing, 403 for someone else's. Raises; never returns None."""
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if user.role is not UserRole.ADMIN and vehicle.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This vehicle belongs to another owner",
        )
    return vehicle


@router.get("", response_model=list[schemas.VehicleRead])
def list_vehicles(
    owner_id: int | None = None,
    search: str | None = Query(None, description="match plate, make or model"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # A non-admin's owner_id filter is overridden, not merely defaulted - so
    # passing ?owner_id=<someone else> cannot leak another owner's fleet.
    if user.role is not UserRole.ADMIN:
        owner_id = user.id
    vehicles = crud.list_vehicles(
        db, owner_id=owner_id, search=search, skip=skip, limit=limit
    )
    return [crud.to_vehicle_read(v) for v in vehicles]


@router.post("", response_model=schemas.VehicleRead, status_code=status.HTTP_201_CREATED)
def register_vehicle(
    payload: schemas.VehicleCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Register a vehicle. Owners may only register vehicles to themselves."""
    if user.role is not UserRole.ADMIN:
        payload.owner_id = user.id
    try:
        vehicle = crud.create_vehicle(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return crud.to_vehicle_read(vehicle)


@router.get("/{vehicle_id}", response_model=schemas.VehicleRead)
def get_vehicle(
    vehicle_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vehicle = _authorise(crud.get_vehicle(db, vehicle_id), user)
    return crud.to_vehicle_read(vehicle)


@router.patch("/{vehicle_id}", response_model=schemas.VehicleRead)
def update_vehicle(
    vehicle_id: int,
    payload: schemas.VehicleUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vehicle = _authorise(crud.get_vehicle(db, vehicle_id), user)
    if user.role is not UserRole.ADMIN:
        payload.owner_id = None      # owners cannot transfer a vehicle away
    try:
        updated = crud.update_vehicle(db, vehicle, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return crud.to_vehicle_read(updated)


@router.delete("/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vehicle(
    vehicle_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vehicle = _authorise(crud.get_vehicle(db, vehicle_id), user)
    crud.delete_vehicle(db, vehicle)


@router.post("/{vehicle_id}/image", response_model=schemas.VehicleRead)
async def upload_vehicle_image(
    vehicle_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Attach a photo to a vehicle record.

    The stored filename is generated, never taken from the upload: a client-
    supplied name like `../../backend/main.py` would otherwise let an upload
    escape the media directory.
    """
    vehicle = _authorise(crud.get_vehicle(db, vehicle_id), user)

    contents = await file.read()
    if len(contents) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image must be 8 MB or smaller",
        )

    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    settings.ensure_directories()
    filename = random_filename(suffix)
    (settings.vehicles_dir / filename).write_bytes(contents)

    vehicle.vehicle_image = f"vehicles/{filename}"
    db.commit()
    db.refresh(vehicle)
    log.info("stored image for vehicle %s: %s", vehicle.plate_display, filename)
    return crud.to_vehicle_read(vehicle)
