"""
ORM models - the five tables required by the specification.

    users            owners + administrators (role-based access)
    vehicles         registered vehicle records
    face_embeddings  128/512-d face vectors belonging to an owner
    detection_logs   every frame-level detection worth keeping
    alerts           theft incidents that were dispatched

Relationships
-------------
    User 1---N Vehicle          (a person may register several vehicles)
    User 1---N FaceEmbedding    (several photos per owner = better recognition)
    Vehicle 1---N DetectionLog  (nullable: unknown vehicles are logged too)
    Vehicle 1---N Alert         (nullable: an alert may precede identification)

`ondelete` rules are declared on the foreign keys and enforced because
`base.py` switches on `PRAGMA foreign_keys`.  Deleting an owner removes their
face embeddings (meaningless without the person) but *nulls* the owner column
on detection logs, because an incident record must survive the deletion of the
account it referenced - that is an evidentiary requirement, not a style choice.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (naive `utcnow()` is deprecated)."""
    return datetime.now(timezone.utc)


# =============================================================================
#  Enumerations - stored as strings so the raw DB stays human-readable
# =============================================================================
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    OWNER = "owner"


class VehicleType(str, enum.Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    OTHER = "other"


class ThreatLevel(str, enum.Enum):
    NONE = "none"          # score 0-19
    LOW = "low"            # 20-39
    MEDIUM = "medium"      # 40-59
    HIGH = "high"          # 60-79
    CRITICAL = "critical"  # 80-100


class AlertStatus(str, enum.Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class PersonStatus(str, enum.Enum):
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"        # a face was seen but not confidently matched
    NO_FACE = "no_face"        # a person was detected, face not visible


class OwnerVerificationStatus(str, enum.Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    NOT_RECOGNIZED = "not_recognized"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
#  1. Users
# =============================================================================
class User(Base):
    """
    An administrator or a vehicle owner.

    Only `hashed_password` is stored - see `backend.core.security`.  The plain
    password never touches the database or the log files.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # --- credentials -----------------------------------------------------
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, native_enum=False, length=20),
        default=UserRole.OWNER,
        index=True,
    )

    # --- profile ---------------------------------------------------------
    full_name: Mapped[str] = mapped_column(String(150))
    phone_number: Mapped[str | None] = mapped_column(String(32))
    address: Mapped[str | None] = mapped_column(String(255))
    profile_image: Mapped[str | None] = mapped_column(String(512))

    # The unique credential printed on this person's membership card, e.g.
    # "VTD-4F9A-2C71". Unique and non-nullable: it is the deterministic
    # counterpart to probabilistic face matching, so two people must never
    # share one. See `backend/core/barcode.py`.
    owner_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)

    # --- account state ---------------------------------------------------
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_email: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_sms: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)

    # --- relationships ---------------------------------------------------
    vehicles: Mapped[list["Vehicle"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    face_embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    def __repr__(self) -> str:
        return f"<User id={self.id} {self.email} role={self.role.value}>"


# =============================================================================
#  2. Vehicles
# =============================================================================
class Vehicle(Base):
    """
    A registered vehicle.

    `plate_number` is stored **normalised** (upper-case, no spaces or dashes:
    "ABC-123XY" -> "ABC123XY") so that OCR output can be compared without the
    formatting noise real plates carry.  `plate_display` keeps the pretty form
    for the dashboard and alert messages.
    """

    __tablename__ = "vehicles"
    __table_args__ = (
        UniqueConstraint("plate_number", name="uq_vehicles_plate_number"),
        Index("ix_vehicles_owner_active", "owner_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # --- identification --------------------------------------------------
    plate_number: Mapped[str] = mapped_column(String(20), index=True)
    plate_display: Mapped[str] = mapped_column(String(24))

    # --- description -----------------------------------------------------
    make: Mapped[str | None] = mapped_column(String(60))       # Toyota
    model: Mapped[str | None] = mapped_column(String(60))      # Camry
    year: Mapped[int | None] = mapped_column(Integer)
    color: Mapped[str | None] = mapped_column(String(40))
    vehicle_type: Mapped[VehicleType] = mapped_column(
        SAEnum(VehicleType, native_enum=False, length=20),
        default=VehicleType.CAR,
    )
    vin: Mapped[str | None] = mapped_column(String(40))

    # --- media -----------------------------------------------------------
    vehicle_image: Mapped[str | None] = mapped_column(String(512))

    # --- state -----------------------------------------------------------
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Set by the owner when the car is genuinely reported stolen: any sighting
    # then escalates straight to CRITICAL regardless of who is driving.
    is_flagged_stolen: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    registration_date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    # --- relationships ---------------------------------------------------
    owner: Mapped["User"] = relationship(back_populates="vehicles")
    detection_logs: Mapped[list["DetectionLog"]] = relationship(
        back_populates="vehicle", passive_deletes=True
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="vehicle", passive_deletes=True
    )

    @staticmethod
    def normalise_plate(raw: str) -> str:
        """"abc 123-xy" -> "ABC123XY". The single source of truth for matching."""
        return "".join(ch for ch in (raw or "").upper() if ch.isalnum())

    @property
    def description(self) -> str:
        """"2019 Toyota Camry (Silver)" for alert bodies and the dashboard."""
        bits = [str(self.year) if self.year else "", self.make or "", self.model or ""]
        label = " ".join(b for b in bits if b).strip() or "Unspecified vehicle"
        return f"{label} ({self.color})" if self.color else label

    def __repr__(self) -> str:
        return f"<Vehicle id={self.id} plate={self.plate_display}>"


# =============================================================================
#  3. Face embeddings
# =============================================================================
class FaceEmbedding(Base):
    """
    One face vector for one owner.

    Storage format
    --------------
    The raw float32 vector is kept as a BLOB (`vector`) rather than JSON: for a
    512-d Facenet512 embedding that is 2 KB instead of ~6 KB of text, and it
    loads into NumPy with a single `frombuffer` call - which matters because the
    recogniser reloads every registered embedding into memory at startup.
    `dim` and `model_name` are stored alongside so a database populated with one
    backbone is never silently compared against vectors from another.
    """

    __tablename__ = "face_embeddings"
    __table_args__ = (
        Index("ix_face_owner_active", "owner_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    vector: Mapped[bytes] = mapped_column(LargeBinary)
    dim: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(60), default="Facenet512")
    detector_backend: Mapped[str] = mapped_column(String(40), default="retinaface")

    source_image: Mapped[str | None] = mapped_column(String(512))
    quality_score: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    owner: Mapped["User"] = relationship(back_populates="face_embeddings")

    def __repr__(self) -> str:
        return (
            f"<FaceEmbedding id={self.id} owner={self.owner_id} "
            f"dim={self.dim} model={self.model_name}>"
        )


# =============================================================================
#  4. Detection logs
# =============================================================================
class DetectionLog(Base):
    """
    A saved observation from the surveillance pipeline.

    Not every frame is written - at 30 FPS that would be 2.6 M rows a day.  The
    pipeline records a log row when something *changes*: a new vehicle enters
    the scene, a plate is read, a person is classified, or a threat is scored.
    """

    __tablename__ = "detection_logs"
    __table_args__ = (
        Index("ix_logs_time_camera", "detected_at", "camera_id"),
        Index("ix_logs_threat", "threat_level", "detected_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # --- where / when ----------------------------------------------------
    camera_id: Mapped[str] = mapped_column(String(60), default="cam-0", index=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True
    )
    frame_number: Mapped[int | None] = mapped_column(Integer)

    # --- what was seen ---------------------------------------------------
    object_class: Mapped[str] = mapped_column(String(40))     # car / person / ...
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # Bounding box in absolute pixels, top-left origin.
    bbox_x1: Mapped[int | None] = mapped_column(Integer)
    bbox_y1: Mapped[int | None] = mapped_column(Integer)
    bbox_x2: Mapped[int | None] = mapped_column(Integer)
    bbox_y2: Mapped[int | None] = mapped_column(Integer)
    track_id: Mapped[int | None] = mapped_column(Integer, index=True)

    # --- plate recognition ----------------------------------------------
    plate_number: Mapped[str | None] = mapped_column(String(20), index=True)
    plate_text_raw: Mapped[str | None] = mapped_column(String(40))
    plate_confidence: Mapped[float | None] = mapped_column(Float)
    plate_in_database: Mapped[bool | None] = mapped_column(Boolean)

    # --- face recognition ------------------------------------------------
    person_status: Mapped[PersonStatus | None] = mapped_column(
        SAEnum(PersonStatus, native_enum=False, length=20)
    )
    face_similarity: Mapped[float | None] = mapped_column(Float)
    matched_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    expected_owner_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    owner_verification: Mapped[OwnerVerificationStatus | None] = mapped_column(
        SAEnum(OwnerVerificationStatus, native_enum=False, length=20)
    )

    # --- assessment ------------------------------------------------------
    threat_score: Mapped[int] = mapped_column(Integer, default=0)
    threat_level: Mapped[ThreatLevel] = mapped_column(
        SAEnum(ThreatLevel, native_enum=False, length=20),
        default=ThreatLevel.NONE,
    )
    reason: Mapped[str | None] = mapped_column(Text)

    # --- evidence --------------------------------------------------------
    snapshot_path: Mapped[str | None] = mapped_column(String(512))
    processing_ms: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)

    vehicle_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicles.id", ondelete="SET NULL"), index=True
    )
    vehicle: Mapped["Vehicle | None"] = relationship(back_populates="detection_logs")

    def __repr__(self) -> str:
        return (
            f"<DetectionLog id={self.id} {self.object_class} "
            f"threat={self.threat_score}>"
        )


# =============================================================================
#  5. Alerts
# =============================================================================
class Alert(Base):
    """
    A dispatched theft incident.

    Everything the specification requires in a notification is a column here, so
    an alert can be re-sent or reviewed months later without reconstructing it
    from the detection logs: vehicle info, plate, timestamp, GPS, the captured
    face image, and the screenshot / clip evidence.

    Delivery is recorded per channel (`email_sent`, `sms_sent`,
    `delivery_error`) because a notification system that cannot prove it
    delivered is not an alerting system.
    """

    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alerts_status_time", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # --- classification --------------------------------------------------
    threat_level: Mapped[ThreatLevel] = mapped_column(
        SAEnum(ThreatLevel, native_enum=False, length=20),
        default=ThreatLevel.HIGH,
        index=True,
    )
    threat_score: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str] = mapped_column(Text)
    # Machine-readable trigger list, e.g. "unknown_person,plate_not_registered".
    triggers: Mapped[str | None] = mapped_column(String(255))

    # --- subject ---------------------------------------------------------
    vehicle_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicles.id", ondelete="SET NULL"), index=True
    )
    plate_number: Mapped[str | None] = mapped_column(String(20), index=True)
    person_status: Mapped[PersonStatus | None] = mapped_column(
        SAEnum(PersonStatus, native_enum=False, length=20)
    )

    # --- where / when ----------------------------------------------------
    camera_id: Mapped[str] = mapped_column(String(60), default="cam-0")
    site_name: Mapped[str | None] = mapped_column(String(120))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True
    )

    # --- evidence --------------------------------------------------------
    snapshot_path: Mapped[str | None] = mapped_column(String(512))
    face_image_path: Mapped[str | None] = mapped_column(String(512))
    video_clip_path: Mapped[str | None] = mapped_column(String(512))

    # --- delivery audit trail -------------------------------------------
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    sms_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime)
    delivery_error: Mapped[str | None] = mapped_column(Text)

    # --- operator workflow ----------------------------------------------
    status: Mapped[AlertStatus] = mapped_column(
        SAEnum(AlertStatus, native_enum=False, length=20),
        default=AlertStatus.NEW,
        index=True,
    )
    acknowledged_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime)
    resolution_notes: Mapped[str | None] = mapped_column(Text)

    detection_log_id: Mapped[int | None] = mapped_column(
        ForeignKey("detection_logs.id", ondelete="SET NULL")
    )

    vehicle: Mapped["Vehicle | None"] = relationship(back_populates="alerts")

    @property
    def google_maps_url(self) -> str | None:
        """Clickable location for the email/SMS body."""
        if self.latitude is None or self.longitude is None:
            return None
        return f"https://maps.google.com/?q={self.latitude},{self.longitude}"

    def __repr__(self) -> str:
        return (
            f"<Alert id={self.id} {self.threat_level.value} "
            f"score={self.threat_score} status={self.status.value}>"
        )


__all__ = [
    "User",
    "Vehicle",
    "FaceEmbedding",
    "DetectionLog",
    "Alert",
    "UserRole",
    "VehicleType",
    "ThreatLevel",
    "AlertStatus",
    "PersonStatus",
]
