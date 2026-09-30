"""
Pydantic schemas - the API's request/response contract.

Three variants per entity, following the standard FastAPI convention:

    XxxCreate   what a client may send when creating   (accepts a password)
    XxxUpdate   partial edit; every field optional     (PATCH semantics)
    XxxRead     what the server returns                (never a password hash)

Keeping these separate from the ORM models is what stops a hashed password or
an internal file path from leaking into a JSON response by accident.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.database.models import (
    AlertStatus,
    OwnerVerificationStatus,
    PersonStatus,
    ThreatLevel,
    UserRole,
    VehicleType,
)

# `from_attributes` lets FastAPI build a response straight from an ORM object.
_ORM = ConfigDict(from_attributes=True)


# =============================================================================
#  Auth
# =============================================================================
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserRead"


# =============================================================================
#  Users
# =============================================================================
class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=150)
    phone_number: str | None = Field(default=None, max_length=32)
    address: str | None = Field(default=None, max_length=255)
    notify_email: bool = True
    notify_sms: bool = True


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)
    role: UserRole = UserRole.OWNER

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        """Reject the weakest passwords at the edge rather than in the router."""
        if not any(c.isdigit() for c in v):
            raise ValueError("password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("password must contain at least one letter")
        return v


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=150)
    phone_number: str | None = None
    address: str | None = None
    notify_email: bool | None = None
    notify_sms: bool | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserRead(UserBase):
    model_config = _ORM

    id: int
    role: UserRole
    is_active: bool
    owner_code: str
    profile_image: str | None = None
    created_at: datetime
    last_login_at: datetime | None = None
    face_count: int = 0          # populated by crud for the owner list view
    vehicle_count: int = 0


class OwnerBarcodeRead(BaseModel):
    """Everything needed to render a membership card, in one response."""

    owner_id: int
    owner_code: str
    full_name: str
    card_path: str
    barcode_path: str
    qr_path: str


# =============================================================================
#  Vehicles
# =============================================================================
class VehicleBase(BaseModel):
    plate_display: str = Field(min_length=2, max_length=24)
    make: str | None = Field(default=None, max_length=60)
    model: str | None = Field(default=None, max_length=60)
    year: int | None = Field(default=None, ge=1900, le=2100)
    color: str | None = Field(default=None, max_length=40)
    vehicle_type: VehicleType = VehicleType.CAR
    vin: str | None = Field(default=None, max_length=40)
    notes: str | None = None


class VehicleCreate(VehicleBase):
    owner_id: int


class VehicleUpdate(BaseModel):
    plate_display: str | None = Field(default=None, min_length=2, max_length=24)
    make: str | None = None
    model: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    color: str | None = None
    vehicle_type: VehicleType | None = None
    vin: str | None = None
    notes: str | None = None
    is_active: bool | None = None
    is_flagged_stolen: bool | None = None
    owner_id: int | None = None


class VehicleRead(VehicleBase):
    model_config = _ORM

    id: int
    owner_id: int
    plate_number: str
    vehicle_image: str | None = None
    is_active: bool
    is_flagged_stolen: bool
    registration_date: datetime
    owner_name: str | None = None       # flattened for the dashboard table
    owner_phone: str | None = None
    owner_face_image: str | None = None
    owner_face_count: int = 0


# =============================================================================
#  Face embeddings
# =============================================================================
class FaceEmbeddingRead(BaseModel):
    model_config = _ORM

    id: int
    owner_id: int
    model_name: str
    dim: int
    source_image: str | None = None
    quality_score: float | None = None
    is_active: bool
    created_at: datetime


class FaceEnrollResult(BaseModel):
    """Returned after uploading owner face photos."""

    owner_id: int
    enrolled: int
    failed: int
    messages: list[str] = []


class BarcodeScanResult(BaseModel):
    """
    Outcome of scanning a membership card.

    `verified` is deliberately three-valued through `status`: a scan that
    decoded nothing, a scan that decoded a code nobody owns, and a scan that
    matched a real account are different operational situations and a guard
    needs to be told which one occurred.
    """

    status: str = Field(
        description="decoded | unknown_code | invalid_payload | no_barcode"
    )
    verified: bool = False
    payload: str | None = None
    normalised_code: str | None = None
    owner_id: int | None = None
    owner_name: str | None = None
    owner_email: str | None = None
    is_active: bool | None = None
    vehicle_count: int | None = None
    message: str | None = None


# =============================================================================
#  Detection logs
# =============================================================================
class DetectionLogRead(BaseModel):
    model_config = _ORM

    id: int
    camera_id: str
    detected_at: datetime
    object_class: str
    confidence: float
    track_id: int | None = None
    plate_number: str | None = None
    plate_confidence: float | None = None
    plate_in_database: bool | None = None
    person_status: PersonStatus | None = None
    face_similarity: float | None = None
    matched_user_id: int | None = None
    expected_owner_id: int | None = None
    owner_verification: OwnerVerificationStatus | None = None
    threat_score: int
    threat_level: ThreatLevel
    reason: str | None = None
    snapshot_path: str | None = None
    vehicle_id: int | None = None
    fps: float | None = None


# =============================================================================
#  Alerts
# =============================================================================
class AlertRead(BaseModel):
    model_config = _ORM

    id: int
    threat_level: ThreatLevel
    threat_score: int
    title: str
    reason: str
    triggers: str | None = None
    vehicle_id: int | None = None
    plate_number: str | None = None
    person_status: PersonStatus | None = None
    camera_id: str
    site_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    created_at: datetime
    snapshot_path: str | None = None
    face_image_path: str | None = None
    video_clip_path: str | None = None
    email_sent: bool
    sms_sent: bool
    delivery_error: str | None = None
    status: AlertStatus
    acknowledged_at: datetime | None = None
    resolution_notes: str | None = None


class AlertUpdate(BaseModel):
    status: AlertStatus
    resolution_notes: str | None = None


# =============================================================================
#  Dashboard statistics
# =============================================================================
class DashboardStats(BaseModel):
    total_owners: int
    total_vehicles: int
    total_detections: int
    detections_today: int
    total_alerts: int
    alerts_today: int
    open_alerts: int
    vehicles_detected_today: int
    unauthorized_today: int
    current_fps: float = 0.0
    pipeline_running: bool = False


class TimeSeriesPoint(BaseModel):
    label: str
    detections: int
    alerts: int


# Resolve the forward reference in Token.user
Token.model_rebuild()
