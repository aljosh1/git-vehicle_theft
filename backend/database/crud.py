"""
CRUD helpers - every database read and write in one place.

The routers call these functions and never build queries themselves.  That
keeps SQL out of the HTTP layer, makes the data access unit-testable without a
running server, and means a change to (say) how plates are matched happens in
exactly one function.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.core.security import DUMMY_PASSWORD_HASH, hash_password, verify_password
from backend.database.models import (
    Alert,
    AlertStatus,
    DetectionLog,
    FaceEmbedding,
    ThreatLevel,
    User,
    UserRole,
    Vehicle,
)
from backend.database import schemas
from backend.utils.logger import get_logger

log = get_logger(__name__)

# How many times to redraw an owner code that somehow collides before giving up.
# Eight random characters from a 32-character alphabet is a ~4e-12 chance per
# draw, so exhausting this is a sign something is wrong, not bad luck.
_OWNER_CODE_ATTEMPTS = 12


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
#  Users
# =============================================================================
def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(func.lower(User.email) == email.lower()))


def list_users(
    db: Session,
    *,
    role: UserRole | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[User]:
    stmt = select(User)
    if role is not None:
        stmt = stmt.where(User.role == role)
    if search:
        pattern = f"%{search.lower()}%"
        stmt = stmt.where(
            func.lower(User.full_name).like(pattern)
            | func.lower(User.email).like(pattern)
            | func.lower(User.owner_code).like(pattern)
        )
    stmt = stmt.order_by(User.created_at.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


def create_user(db: Session, payload: schemas.UserCreate) -> User:
    """Create an account. Raises ValueError if the email is already taken."""
    if get_user_by_email(db, payload.email):
        raise ValueError(f"an account already exists for {payload.email}")

    user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        role=payload.role,
        full_name=payload.full_name,
        phone_number=payload.phone_number,
        address=payload.address,
        notify_email=payload.notify_email,
        notify_sms=payload.notify_sms,
        owner_code=issue_owner_code(db),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("created %s account: %s (%s)", user.role.value, user.email, user.owner_code)
    return user


def issue_owner_code(db: Session) -> str:
    """
    Draw an owner code that is not already in use.

    Uniqueness is checked in Python rather than left to the database constraint
    because the caller needs the value in order to build the `User`; catching the
    IntegrityError and retrying would work too, but it would abort and reissue
    the whole transaction for what is a trivially rare event on a 32-character
    alphabet of 8-character codes.
    """
    from backend.core.barcode import generate_owner_code

    for _ in range(_OWNER_CODE_ATTEMPTS):
        code = generate_owner_code()
        if get_user_by_owner_code(db, code) is None:
            return code
    raise RuntimeError(
        f"could not find a free owner code in {_OWNER_CODE_ATTEMPTS} attempts"
    )


def get_user_by_owner_code(db: Session, code: str) -> User | None:
    """
    Find an account by its membership-card code.

    The raw scanned value is normalised first, because scanners disagree about
    case and separators - `vtd 4f9a 2c71` and `VTD-4F9A-2C71` are the same card.
    """
    from backend.core.barcode import normalise_owner_code

    normalised = normalise_owner_code(code)
    if normalised is None:
        return None
    return db.scalar(select(User).where(User.owner_code == normalised))


def update_user(db: Session, user: User, payload: schemas.UserUpdate) -> User:
    data = payload.model_dump(exclude_unset=True)
    if password := data.pop("password", None):
        user.hashed_password = hash_password(password)
    for field, value in data.items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user: User) -> None:
    db.delete(user)          # cascades to vehicles + face embeddings
    db.commit()
    log.info(
        "deleted account %s - its code %s is now free to reissue",
        user.email, user.owner_code,
    )



def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None:
        verify_password(password, DUMMY_PASSWORD_HASH)
        return None
    if not verify_password(password, user.hashed_password):
        log.warning("failed login attempt for %s", email)
        return None
    if not user.is_active:
        log.warning("login attempt on disabled account %s", email)
        return None

    # Merge or re-fetch to ensure the instance belongs to the active session
    user = db.merge(user)
    user.last_login_at = _utcnow()
    db.commit()
    db.refresh(user)
    return user


def to_user_read(db: Session, user: User) -> schemas.UserRead:
    """Attach the counts the dashboard's owner table displays."""
    dto = schemas.UserRead.model_validate(user)
    dto.vehicle_count = db.scalar(
        select(func.count(Vehicle.id)).where(Vehicle.owner_id == user.id)
    ) or 0
    dto.face_count = db.scalar(
        select(func.count(FaceEmbedding.id)).where(
            FaceEmbedding.owner_id == user.id, FaceEmbedding.is_active.is_(True)
        )
    ) or 0
    return dto


# =============================================================================
#  Vehicles
# =============================================================================
def get_vehicle(db: Session, vehicle_id: int) -> Vehicle | None:
    return db.get(Vehicle, vehicle_id)


def get_vehicle_by_plate(db: Session, plate: str) -> Vehicle | None:
    """Exact lookup on the normalised plate. Used by the theft engine."""
    normalised = Vehicle.normalise_plate(plate)
    if not normalised:
        return None
    return db.scalar(
        select(Vehicle).where(
            Vehicle.plate_number == normalised, Vehicle.is_active.is_(True)
        )
    )


def list_vehicles(
    db: Session,
    *,
    owner_id: int | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[Vehicle]:
    stmt = select(Vehicle)
    if owner_id is not None:
        stmt = stmt.where(Vehicle.owner_id == owner_id)
    if search:
        pattern = f"%{search.upper()}%"
        stmt = stmt.where(
            Vehicle.plate_number.like(pattern)
            | func.upper(Vehicle.make).like(pattern)
            | func.upper(Vehicle.model).like(pattern)
        )
    stmt = stmt.order_by(Vehicle.registration_date.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


def create_vehicle(db: Session, payload: schemas.VehicleCreate) -> Vehicle:
    normalised = Vehicle.normalise_plate(payload.plate_display)
    if not normalised:
        raise ValueError("plate number must contain letters or digits")
    if get_vehicle_by_plate(db, normalised):
        raise ValueError(f"plate {payload.plate_display} is already registered")
    if db.get(User, payload.owner_id) is None:
        raise ValueError(f"no owner with id {payload.owner_id}")

    vehicle = Vehicle(
        owner_id=payload.owner_id,
        plate_number=normalised,
        plate_display=payload.plate_display.upper().strip(),
        make=payload.make,
        model=payload.model,
        year=payload.year,
        color=payload.color,
        vehicle_type=payload.vehicle_type,
        vin=payload.vin,
        notes=payload.notes,
    )
    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    log.info("registered vehicle %s for owner %s", vehicle.plate_display, vehicle.owner_id)
    return vehicle


def update_vehicle(db: Session, vehicle: Vehicle, payload: schemas.VehicleUpdate) -> Vehicle:
    data = payload.model_dump(exclude_unset=True)
    if display := data.pop("plate_display", None):
        normalised = Vehicle.normalise_plate(display)
        existing = get_vehicle_by_plate(db, normalised)
        if existing and existing.id != vehicle.id:
            raise ValueError(f"plate {display} belongs to another vehicle")
        vehicle.plate_number = normalised
        vehicle.plate_display = display.upper().strip()
    for field, value in data.items():
        setattr(vehicle, field, value)
    db.commit()
    db.refresh(vehicle)
    return vehicle


def delete_vehicle(db: Session, vehicle: Vehicle) -> None:
    db.delete(vehicle)
    db.commit()


def to_vehicle_read(vehicle: Vehicle) -> schemas.VehicleRead:
    dto = schemas.VehicleRead.model_validate(vehicle)
    if vehicle.owner is not None:
        dto.owner_name = vehicle.owner.full_name
        dto.owner_phone = vehicle.owner.phone_number
        active_faces = [face for face in vehicle.owner.face_embeddings if face.is_active]
        dto.owner_face_count = len(active_faces)
        if active_faces:
            latest = max(active_faces, key=lambda face: face.created_at)
            dto.owner_face_image = latest.source_image
    return dto


# =============================================================================
#  Face embeddings
# =============================================================================
def add_face_embedding(
    db: Session,
    *,
    owner_id: int,
    vector: np.ndarray,
    model_name: str,
    detector_backend: str,
    source_image: str | None = None,
    quality_score: float | None = None,
) -> FaceEmbedding:
    """Persist one L2-normalised face vector as a float32 BLOB."""
    vec = np.asarray(vector, dtype=np.float32).ravel()
    record = FaceEmbedding(
        owner_id=owner_id,
        vector=vec.tobytes(),
        dim=int(vec.size),
        model_name=model_name,
        detector_backend=detector_backend,
        source_image=source_image,
        quality_score=quality_score,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_face_embeddings(
    db: Session, *, owner_id: int | None = None, active_only: bool = True
) -> list[FaceEmbedding]:
    stmt = select(FaceEmbedding)
    if owner_id is not None:
        stmt = stmt.where(FaceEmbedding.owner_id == owner_id)
    if active_only:
        stmt = stmt.where(FaceEmbedding.is_active.is_(True))
    return list(db.scalars(stmt.order_by(FaceEmbedding.created_at.desc())))


def load_face_gallery(db: Session) -> list[tuple[int, np.ndarray]]:
    """
    Every active embedding as `(owner_id, vector)`.

    The recogniser calls this once at startup and after each enrolment, then
    matches in memory - a database round-trip per frame would never hold 30 FPS.
    """
    gallery: list[tuple[int, np.ndarray]] = []
    for row in list_face_embeddings(db):
        vec = np.frombuffer(row.vector, dtype=np.float32)
        if vec.size == row.dim:
            gallery.append((row.owner_id, vec))
        else:
            log.error(
                "face embedding %s is corrupt (dim=%s, bytes imply %s) - skipped",
                row.id, row.dim, vec.size,
            )
    return gallery


def delete_face_embedding(db: Session, record: FaceEmbedding) -> None:
    db.delete(record)
    db.commit()


# =============================================================================
#  Detection logs
# =============================================================================
def create_detection_log(db: Session, **fields) -> DetectionLog:
    entry = DetectionLog(**fields)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_detection_logs(
    db: Session,
    *,
    camera_id: str | None = None,
    since: datetime | None = None,
    min_threat: int = 0,
    skip: int = 0,
    limit: int = 100,
) -> list[DetectionLog]:
    stmt = select(DetectionLog)
    if camera_id:
        stmt = stmt.where(DetectionLog.camera_id == camera_id)
    if since:
        stmt = stmt.where(DetectionLog.detected_at >= since)
    if min_threat:
        stmt = stmt.where(DetectionLog.threat_score >= min_threat)
    stmt = stmt.order_by(DetectionLog.detected_at.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


# =============================================================================
#  Alerts
# =============================================================================
def create_alert(db: Session, **fields) -> Alert:
    alert = Alert(**fields)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    log.warning(
        "ALERT #%s %s score=%s - %s",
        alert.id, alert.threat_level.value, alert.threat_score, alert.reason,
    )
    return alert


def get_alert(db: Session, alert_id: int) -> Alert | None:
    return db.get(Alert, alert_id)


def list_alerts(
    db: Session,
    *,
    status: AlertStatus | None = None,
    threat_level: ThreatLevel | None = None,
    since: datetime | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[Alert]:
    stmt = select(Alert)
    if status is not None:
        stmt = stmt.where(Alert.status == status)
    if threat_level is not None:
        stmt = stmt.where(Alert.threat_level == threat_level)
    if since is not None:
        stmt = stmt.where(Alert.created_at >= since)
    stmt = stmt.order_by(Alert.created_at.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


def update_alert_status(
    db: Session,
    alert: Alert,
    *,
    status: AlertStatus,
    user_id: int | None = None,
    notes: str | None = None,
) -> Alert:
    alert.status = status
    alert.acknowledged_by = user_id
    alert.acknowledged_at = _utcnow()
    if notes is not None:
        alert.resolution_notes = notes
    db.commit()
    db.refresh(alert)
    return alert


def mark_alert_delivered(
    db: Session,
    alert: Alert,
    *,
    email_sent: bool,
    sms_sent: bool,
    error: str | None = None,
) -> None:
    alert.email_sent = email_sent
    alert.sms_sent = sms_sent
    alert.notified_at = _utcnow()
    alert.delivery_error = error
    db.commit()


def set_alert_video_clip(db: Session, alert: Alert, video_clip_path: str | None) -> None:
    alert.video_clip_path = video_clip_path
    db.commit()


def recent_alert_exists(
    db: Session, *, camera_id: str, plate_number: str | None, within_seconds: int
) -> bool:
    """
    Anti-spam guard: has an equivalent alert already fired recently?

    Without this a thief standing in frame for two minutes at 30 FPS would
    generate thousands of identical emails.
    """
    cutoff = _utcnow() - timedelta(seconds=within_seconds)
    stmt = select(func.count(Alert.id)).where(
        Alert.camera_id == camera_id, Alert.created_at >= cutoff
    )
    stmt = stmt.where(
        Alert.plate_number == plate_number
        if plate_number
        else Alert.plate_number.is_(None)
    )
    return bool(db.scalar(stmt))


# =============================================================================
#  Dashboard aggregates
# =============================================================================
def dashboard_stats(db: Session) -> schemas.DashboardStats:
    start_of_day = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    def count(model, *conditions) -> int:
        stmt = select(func.count(model.id))
        for condition in conditions:
            stmt = stmt.where(condition)
        return db.scalar(stmt) or 0

    return schemas.DashboardStats(
        total_owners=count(User, User.role == UserRole.OWNER),
        total_vehicles=count(Vehicle),
        total_detections=count(DetectionLog),
        detections_today=count(DetectionLog, DetectionLog.detected_at >= start_of_day),
        total_alerts=count(Alert),
        alerts_today=count(Alert, Alert.created_at >= start_of_day),
        open_alerts=count(Alert, Alert.status == AlertStatus.NEW),
        vehicles_detected_today=db.scalar(
            select(func.count(func.distinct(DetectionLog.plate_number))).where(
                DetectionLog.detected_at >= start_of_day,
                DetectionLog.plate_number.is_not(None),
            )
        ) or 0,
        unauthorized_today=count(
            DetectionLog,
            DetectionLog.detected_at >= start_of_day,
            DetectionLog.threat_score >= 60,
        ),
    )


def detections_timeseries(db: Session, days: int = 7) -> list[schemas.TimeSeriesPoint]:
    """Per-day detection and alert counts for the dashboard chart."""
    points: list[schemas.TimeSeriesPoint] = []
    today = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for offset in range(days - 1, -1, -1):
        day_start = today - timedelta(days=offset)
        day_end = day_start + timedelta(days=1)
        points.append(
            schemas.TimeSeriesPoint(
                label=day_start.strftime("%a"),
                detections=db.scalar(
                    select(func.count(DetectionLog.id)).where(
                        DetectionLog.detected_at >= day_start,
                        DetectionLog.detected_at < day_end,
                    )
                ) or 0,
                alerts=db.scalar(
                    select(func.count(Alert.id)).where(
                        Alert.created_at >= day_start, Alert.created_at < day_end
                    )
                ) or 0,
            )
        )
    return points
