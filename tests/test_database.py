"""
Component test for the database layer (Step 1).

Run with:
    pytest tests/ -v

Uses a throwaway SQLite file per test session, so it never touches the real
`data/vtds.db`.  This is the "test each component" requirement applied to the
foundation: the AI modules get their own test files as they are implemented.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Point the engine at a temporary database *before* importing anything that
# creates it - settings are read at import time.
_TMP_DB = Path(tempfile.gettempdir()) / "vtds_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from backend.core.security import (  # noqa: E402
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.database import crud, schemas  # noqa: E402
from backend.database.base import Base, SessionLocal, engine  # noqa: E402
from backend.database.models import (  # noqa: E402
    OwnerVerificationStatus,
    PersonStatus,
    ThreatLevel,
    UserRole,
    Vehicle,
    VehicleType,
)


@pytest.fixture()
def db():
    """A clean schema for every test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# =============================================================================
#  Security
# =============================================================================
def test_password_hash_is_not_reversible():
    hashed = hash_password("Secret@12345")
    assert hashed != "Secret@12345"
    assert verify_password("Secret@12345", hashed)
    assert not verify_password("wrong-password", hashed)


def test_same_password_gets_different_hashes():
    """A per-password salt means identical passwords must not collide."""
    assert hash_password("Secret@12345") != hash_password("Secret@12345")


def test_verify_password_survives_a_corrupt_hash():
    assert verify_password("anything", "not-a-real-hash") is False


def test_jwt_round_trip():
    token = create_access_token(user_id=7, email="a@b.c", role="admin")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "7"
    assert payload["role"] == "admin"


def test_tampered_jwt_is_rejected():
    token = create_access_token(user_id=7, email="a@b.c", role="owner")
    assert decode_access_token(token[:-3] + "abc") is None


# =============================================================================
#  Users
# =============================================================================
def test_create_and_authenticate_user(db):
    user = crud.create_user(
        db,
        schemas.UserCreate(
            email="Owner@Example.com",
            password="Owner@12345",
            full_name="Test Owner",
            phone_number="+2348000000000",
        ),
    )
    assert user.email == "owner@example.com"      # normalised to lower case
    assert user.role is UserRole.OWNER
    assert crud.authenticate_user(db, "owner@example.com", "Owner@12345") is not None
    assert crud.authenticate_user(db, "owner@example.com", "nope") is None


def test_duplicate_email_is_rejected(db):
    payload = schemas.UserCreate(
        email="dup@example.com", password="Owner@12345", full_name="First"
    )
    crud.create_user(db, payload)
    with pytest.raises(ValueError, match="already exists"):
        crud.create_user(db, payload)


def test_disabled_account_cannot_authenticate(db):
    user = crud.create_user(
        db,
        schemas.UserCreate(
            email="off@example.com", password="Owner@12345", full_name="Disabled"
        ),
    )
    crud.update_user(db, user, schemas.UserUpdate(is_active=False))
    assert crud.authenticate_user(db, "off@example.com", "Owner@12345") is None


def test_weak_password_is_rejected_by_the_schema():
    with pytest.raises(ValueError):
        schemas.UserCreate(
            email="w@example.com", password="alphabets", full_name="Weak"
        )


# =============================================================================
#  Vehicles
# =============================================================================
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ABC-123XY", "ABC123XY"),
        ("abc 123 xy", "ABC123XY"),
        (" lag-456kj ", "LAG456KJ"),
        ("", ""),
    ],
)
def test_plate_normalisation(raw, expected):
    assert Vehicle.normalise_plate(raw) == expected


def test_register_vehicle_and_lookup_by_plate(db):
    owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="grace@example.com", password="Owner@12345", full_name="Grace"
        ),
    )
    vehicle = crud.create_vehicle(
        db,
        schemas.VehicleCreate(
            owner_id=owner.id,
            plate_display="ABC-123XY",
            make="Toyota",
            model="Camry",
            year=2019,
            color="Silver",
            vehicle_type=VehicleType.CAR,
        ),
    )
    assert vehicle.plate_number == "ABC123XY"
    assert vehicle.description == "2019 Toyota Camry (Silver)"

    # A plate read by OCR without the dash must still match the registration.
    assert crud.get_vehicle_by_plate(db, "abc123xy").id == vehicle.id
    assert crud.get_vehicle_by_plate(db, "ZZZ999") is None


def test_vehicle_response_exposes_linked_owner_face(db):
    import numpy as np

    owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="face-owner@example.com",
            password="Owner@12345",
            full_name="Face Owner",
        ),
    )
    crud.add_face_embedding(
        db,
        owner_id=owner.id,
        vector=np.array([0.6, 0.8], dtype=np.float32),
        model_name="test-model",
        detector_backend="test-detector",
        source_image="faces/owner.jpg",
    )
    vehicle = crud.create_vehicle(
        db,
        schemas.VehicleCreate(owner_id=owner.id, plate_display="FAC-123AA"),
    )

    response = crud.to_vehicle_read(vehicle)

    assert response.owner_id == owner.id
    assert response.owner_name == "Face Owner"
    assert response.owner_face_count == 1
    assert response.owner_face_image == "faces/owner.jpg"


def test_duplicate_plate_is_rejected(db):
    owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="d@example.com", password="Owner@12345", full_name="Dup Owner"
        ),
    )
    spec = schemas.VehicleCreate(owner_id=owner.id, plate_display="XYZ-111AA")
    crud.create_vehicle(db, spec)
    with pytest.raises(ValueError, match="already registered"):
        crud.create_vehicle(db, spec)


def test_deleting_an_owner_cascades_to_their_vehicles(db):
    owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="c@example.com", password="Owner@12345", full_name="Cascade Owner"
        ),
    )
    crud.create_vehicle(
        db, schemas.VehicleCreate(owner_id=owner.id, plate_display="CAS-001AA")
    )
    crud.delete_user(db, owner)
    assert crud.get_vehicle_by_plate(db, "CAS001AA") is None


# =============================================================================
#  Detection verification persistence
# =============================================================================
def test_detection_log_persists_owner_verification(db):
    expected_owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="expected@example.com",
            password="Owner@12345",
            full_name="Expected Owner",
        ),
    )
    matched_owner = crud.create_user(
        db,
        schemas.UserCreate(
            email="matched@example.com",
            password="Owner@12345",
            full_name="Matched Owner",
        ),
    )
    vehicle = crud.create_vehicle(
        db,
        schemas.VehicleCreate(
            owner_id=expected_owner.id,
            plate_display="OWN-123AA",
        ),
    )

    log = crud.create_detection_log(
        db,
        camera_id="cam-test",
        object_class="car",
        confidence=0.92,
        plate_number=vehicle.plate_number,
        plate_in_database=True,
        person_status=PersonStatus.UNAUTHORIZED,
        matched_user_id=matched_owner.id,
        expected_owner_id=expected_owner.id,
        owner_verification=OwnerVerificationStatus.MISMATCH,
        threat_score=85,
        threat_level=ThreatLevel.CRITICAL,
        vehicle_id=vehicle.id,
    )
    response = schemas.DetectionLogRead.model_validate(log)

    assert response.matched_user_id == matched_owner.id
    assert response.expected_owner_id == expected_owner.id
    assert response.owner_verification is OwnerVerificationStatus.MISMATCH


# =============================================================================
#  Dashboard aggregates
# =============================================================================
def test_dashboard_stats_on_an_empty_database(db):
    stats = crud.dashboard_stats(db)
    assert stats.total_owners == 0
    assert stats.total_vehicles == 0
    assert stats.open_alerts == 0


def test_detections_timeseries_returns_one_point_per_day(db):
    points = crud.detections_timeseries(db, days=7)
    assert len(points) == 7
    assert all(p.detections == 0 for p in points)
