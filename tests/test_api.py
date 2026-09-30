"""
End-to-end API tests using FastAPI's TestClient.

Covers the auth flow and the authorisation rules that protect one owner's data
from another - the security properties most worth having a regression test on.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_TMP_DB = Path(tempfile.gettempdir()) / "vtds_api_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.alerts import dispatcher  # noqa: E402
from backend.database import crud, schemas  # noqa: E402
from backend.database.base import Base, SessionLocal, engine  # noqa: E402
from backend.database.models import (  # noqa: E402
    OwnerVerificationStatus,
    PersonStatus,
    ThreatLevel,
    UserRole,
)
from backend.main import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        crud.create_user(
            db,
            schemas.UserCreate(
                email="admin@test.example.com",
                password="Admin@12345",
                full_name="Test Admin",
                role=UserRole.ADMIN,
            ),
        )
        crud.create_user(
            db,
            schemas.UserCreate(
                email="owner1@test.example.com",
                password="Owner@12345",
                full_name="Owner One",
            ),
        )
        crud.create_user(
            db,
            schemas.UserCreate(
                email="owner2@test.example.com",
                password="Owner@12345",
                full_name="Owner Two",
            ),
        )
    finally:
        db.close()

    with TestClient(app) as test_client:
        yield test_client


def login(client, email: str, password: str = "Owner@12345") -> dict[str, str]:
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# =============================================================================
#  System
# =============================================================================
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_info_reports_capabilities(client):
    body = client.get("/api/info").json()
    assert "capabilities" in body
    assert set(body["capabilities"]) >= {"opencv", "torch", "ultralytics"}


def test_all_routers_are_mounted(client):
    routes = client.get("/api/info").json()["mounted_routes"]
    for expected in ("/api/auth/login", "/api/vehicles", "/api/alerts", "/api/stats"):
        assert any(r.startswith(expected) for r in routes), f"{expected} not mounted"


def test_detection_api_exposes_owner_verification(client):
    db = SessionLocal()
    try:
        log = crud.create_detection_log(
            db,
            camera_id="cam-test",
            object_class="car",
            confidence=0.93,
            plate_number="OWN123AA",
            plate_in_database=True,
            person_status=PersonStatus.UNAUTHORIZED,
            matched_user_id=3,
            expected_owner_id=2,
            owner_verification=OwnerVerificationStatus.MISMATCH,
            threat_score=85,
            threat_level=ThreatLevel.CRITICAL,
        )
        log_id = log.id
    finally:
        db.close()

    response = client.get(
        "/api/detections",
        headers=login(client, "admin@test.example.com", "Admin@12345"),
    )

    assert response.status_code == 200
    row = next(item for item in response.json() if item["id"] == log_id)
    assert row["matched_user_id"] == 3
    assert row["expected_owner_id"] == 2
    assert row["owner_verification"] == "mismatch"


# =============================================================================
#  Authentication
# =============================================================================
def test_login_returns_a_token_and_profile(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "Admin@12345"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["role"] == "admin"
    assert "hashed_password" not in body["user"]


def test_login_with_a_bad_password_is_401(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "wrong"},
    )
    assert response.status_code == 401


def test_unknown_and_wrong_password_are_indistinguishable(client):
    """Otherwise the endpoint is an account-enumeration oracle."""
    unknown = client.post(
        "/api/auth/login",
        json={"email": "nobody@test.example.com", "password": "whatever1"},
    )
    wrong = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "whatever1"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_protected_route_requires_a_token(client):
    assert client.get("/api/auth/me").status_code in (401, 403)


def test_garbage_token_is_rejected(client):
    response = client.get(
        "/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_self_registration_cannot_grant_admin(client):
    """A privilege-escalation attempt through the public register endpoint."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "sneaky@test.example.com",
            "password": "Sneaky@12345",
            "full_name": "Sneaky User",
            "role": "admin",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "owner", "role must be forced to owner"


def test_duplicate_registration_is_409(client):
    payload = {
        "email": "dupe@test.example.com",
        "password": "Dupe@12345",
        "full_name": "Duplicate User",
    }
    assert client.post("/api/auth/register", json=payload).status_code == 201
    assert client.post("/api/auth/register", json=payload).status_code == 409


# =============================================================================
#  Owner management (admin only)
# =============================================================================
def test_owners_list_requires_admin(client):
    owner_headers = login(client, "owner1@test.example.com")
    assert client.get("/api/owners", headers=owner_headers).status_code == 403

    admin_headers = login(client, "admin@test.example.com", "Admin@12345")
    response = client.get("/api/owners", headers=admin_headers)
    assert response.status_code == 200
    assert len(response.json()) >= 3


def test_admin_cannot_delete_their_own_account(client):
    headers = login(client, "admin@test.example.com", "Admin@12345")
    me = client.get("/api/auth/me", headers=headers).json()
    response = client.delete(f"/api/owners/{me['id']}", headers=headers)
    assert response.status_code == 400


# =============================================================================
#  Vehicles
# =============================================================================
def test_register_and_retrieve_a_vehicle(client):
    headers = login(client, "owner1@test.example.com")
    owner = client.get("/api/auth/me", headers=headers).json()
    db = SessionLocal()
    try:
        crud.add_face_embedding(
            db,
            owner_id=owner["id"],
            vector=np.array([0.6, 0.8], dtype=np.float32),
            model_name="test-model",
            detector_backend="test-detector",
            source_image="faces/owner-one.jpg",
        )
    finally:
        db.close()

    response = client.post(
        "/api/vehicles",
        headers=headers,
        json={
            "owner_id": 999,          # must be overridden to the caller
            "plate_display": "ABC-123XY",
            "make": "Toyota",
            "model": "Camry",
            "year": 2019,
            "color": "Silver",
            "vehicle_type": "car",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["plate_number"] == "ABC123XY"
    assert body["owner_id"] != 999, "owner_id must be forced to the caller"
    assert body["owner_face_image"] == "faces/owner-one.jpg"
    assert body["owner_face_count"] == 1


def test_owners_cannot_see_each_others_vehicles(client):
    headers1 = login(client, "owner1@test.example.com")
    client.post(
        "/api/vehicles",
        headers=headers1,
        json={"owner_id": 0, "plate_display": "OWN-111AA"},
    )

    headers2 = login(client, "owner2@test.example.com")
    visible = client.get("/api/vehicles", headers=headers2).json()
    assert all(v["plate_number"] != "OWN111AA" for v in visible)


def test_owner_id_filter_cannot_leak_another_fleet(client):
    """?owner_id=<someone else> must be overridden, not honoured."""
    headers1 = login(client, "owner1@test.example.com")
    created = client.post(
        "/api/vehicles",
        headers=headers1,
        json={"owner_id": 0, "plate_display": "LEK-222BB"},
    ).json()

    headers2 = login(client, "owner2@test.example.com")
    leaked = client.get(
        f"/api/vehicles?owner_id={created['owner_id']}", headers=headers2
    ).json()
    assert leaked == []


def test_owner_cannot_edit_another_owners_vehicle(client):
    headers1 = login(client, "owner1@test.example.com")
    created = client.post(
        "/api/vehicles",
        headers=headers1,
        json={"owner_id": 0, "plate_display": "PRV-333CC"},
    ).json()

    headers2 = login(client, "owner2@test.example.com")
    response = client.patch(
        f"/api/vehicles/{created['id']}", headers=headers2, json={"color": "Pink"}
    )
    assert response.status_code == 403


def test_duplicate_plate_is_409(client):
    headers = login(client, "owner1@test.example.com")
    payload = {"owner_id": 0, "plate_display": "DUP-444DD"}
    assert client.post("/api/vehicles", headers=headers, json=payload).status_code == 201
    assert client.post("/api/vehicles", headers=headers, json=payload).status_code == 409


def test_admin_sees_every_vehicle(client):
    owner_headers = login(client, "owner1@test.example.com")
    client.post(
        "/api/vehicles",
        headers=owner_headers,
        json={"owner_id": 0, "plate_display": "ADM-555EE"},
    )
    admin_headers = login(client, "admin@test.example.com", "Admin@12345")
    plates = {v["plate_number"] for v in client.get("/api/vehicles", headers=admin_headers).json()}
    assert "ADM555EE" in plates


# =============================================================================
#  Dashboard + alerts
# =============================================================================
def test_test_email_requires_admin(client):
    headers = login(client, "owner1@test.example.com")
    response = client.post("/api/alerts/test-email", headers=headers)
    assert response.status_code == 403


def test_admin_test_email_uses_configured_sender(client, monkeypatch):
    """
    The test email goes to the configured recipients, and only those.

    `ALERT_EMAIL_TO` is patched rather than read from the environment, so the
    assertion does not depend on whatever the developer's local `.env` happens
    to contain - the test asserts a property of the code, not of the machine.
    """
    from backend.api import alerts as alerts_module

    calls = {}

    def fake_send_email(**kwargs):
        calls.update(kwargs)
        return True, None

    monkeypatch.setattr("backend.api.alerts.send_email_alert", fake_send_email)
    monkeypatch.setattr(alerts_module.settings, "ALERT_EMAIL_TO", "security@example.com")
    # The bootstrap admin is a placeholder address and must not be appended.
    monkeypatch.setattr(
        alerts_module.settings, "ADMIN_EMAIL", "admin@vtds.example.com"
    )

    headers = login(client, "admin@test.example.com", "Admin@12345")
    response = client.post("/api/alerts/test-email", headers=headers)

    assert response.status_code == 200
    assert response.json()["sent"] is True
    assert calls["recipients"] == ["security@example.com"]
    assert calls["subject"].endswith("SMTP test")


def test_disabled_notification_channels_do_not_create_delivery_error(client, monkeypatch):
    from backend.alerts.dispatcher import _send_notifications
    from backend.database.models import AlertStatus, ThreatLevel

    db = SessionLocal()
    try:
        alert = crud.create_alert(
            db,
            threat_level=ThreatLevel.HIGH,
            threat_score=80,
            title="Test alert",
            reason="Test reason",
            camera_id="test-camera",
            status=AlertStatus.NEW,
        )
        monkeypatch.setattr(dispatcher, "send_email_alert", lambda **kwargs: (False, "disabled"))
        monkeypatch.setattr(dispatcher, "send_sms_alert", lambda **kwargs: (False, "disabled"))
        monkeypatch.setattr(dispatcher.settings, "ALERTS_EMAIL_ENABLED", False)
        monkeypatch.setattr(dispatcher.settings, "ALERTS_SMS_ENABLED", False)

        _send_notifications(db, alert)
        db.refresh(alert)
        assert alert.email_sent is False
        assert alert.sms_sent is False
        assert alert.delivery_error is None
    finally:
        db.close()


def test_dashboard_stats(client):
    headers = login(client, "admin@test.example.com", "Admin@12345")
    body = client.get("/api/stats", headers=headers).json()
    assert body["total_owners"] >= 2
    assert body["pipeline_running"] is False


def test_timeseries_returns_the_requested_span(client):
    headers = login(client, "admin@test.example.com", "Admin@12345")
    body = client.get("/api/stats/timeseries?days=7", headers=headers).json()
    assert len(body) == 7


def test_alert_history_is_empty_initially(client):
    headers = login(client, "admin@test.example.com", "Admin@12345")
    assert client.get("/api/alerts", headers=headers).json() == []


def test_mjpeg_requires_a_running_pipeline(client):
    headers = login(client, "admin@test.example.com", "Admin@12345")
    token = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "Admin@12345"},
    ).json()["access_token"]
    response = client.get(f"/api/stream/mjpeg?camera_id=cam-0&token={token}")
    assert response.status_code == 409


def test_stream_control_requires_admin(client):
    headers = login(client, "owner1@test.example.com")
    response = client.get("/api/stream/status", headers=headers)
    assert response.status_code == 403


def test_image_analysis_requires_admin(client):
    headers = login(client, "owner1@test.example.com")
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 403


def test_image_analysis_rejects_unsupported_extension(client):
    """
    A GIF is rejected with 415, not 400.

    Upload validation is content-based: a GIF is a real, decodable image, just
    not one this system handles - which is a different problem from "the bytes
    are corrupt" and deserves its own status code so a client can react to it.
    """
    headers = login(client, "admin@test.example.com", "Admin@12345")
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.gif", b"GIF89a" + b"\x00" * 16, "image/gif")},
    )
    assert response.status_code == 415


def test_image_analysis_rejects_invalid_image_bytes(client):
    """
    Random bytes wearing a .jpg extension are rejected as an unsupported format.

    Previously this was 400, because validation looked at the extension and
    "not-an-image" looked like a perfectly good .jpg. The content check now
    catches it first: there is no JPEG magic number, so 415 is the accurate
    answer - this file is not an image in any form, rather than a damaged one.
    """
    headers = login(client, "admin@test.example.com", "Admin@12345")
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 415
