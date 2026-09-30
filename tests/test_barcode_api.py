"""
End-to-end tests for membership cards and the alert delivery channels.

These exercise the HTTP surface rather than the internals, because the security
properties that matter here are about *access*: a card must not be readable by
another owner, a scan must not verify a disabled account, and the test-email
endpoint must send to exactly the configured recipients and nobody else.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_TMP_DB = Path(tempfile.gettempdir()) / "vtds_barcode_api_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.core.barcode import render_card  # noqa: E402
from backend.database import crud, schemas  # noqa: E402
from backend.database.base import Base, SessionLocal, engine  # noqa: E402
from backend.database.models import UserRole  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        admin = crud.create_user(
            db,
            schemas.UserCreate(
                email="admin@test.example.com",
                password="Admin@12345",
                full_name="Test Admin",
                role=UserRole.ADMIN,
            ),
        )
        owner1 = crud.create_user(
            db,
            schemas.UserCreate(
                email="owner1@test.example.com",
                password="Owner@12345",
                full_name="Owner One",
            ),
        )
        owner2 = crud.create_user(
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
        yield test_client, admin.id, owner1.id, owner2.id


def login(client, email: str, password: str = "Owner@12345") -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"email": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


# =============================================================================
#  Owner codes are unique and present
# =============================================================================
def test_every_account_receives_a_unique_owner_code(client):
    _, _, owner1_id, owner2_id = client
    db = SessionLocal()
    try:
        first = crud.get_user(db, owner1_id)
        second = crud.get_user(db, owner2_id)
    finally:
        db.close()

    assert first.owner_code and second.owner_code
    assert first.owner_code != second.owner_code


def test_owner_code_is_returned_by_the_owners_api(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.get("/api/owners", headers=headers)
    assert response.status_code == 200

    body = response.json()
    assert all("owner_code" in row for row in body)
    assert all(row["owner_code"].startswith("VTD-") for row in body)


def test_owners_can_be_searched_by_their_code(client):
    test_client, _, owner1_id, _ = client
    db = SessionLocal()
    try:
        code = crud.get_user(db, owner1_id).owner_code
    finally:
        db.close()

    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.get(f"/api/owners?search={code}", headers=headers)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [owner1_id]


# =============================================================================
#  Card retrieval and access control
# =============================================================================
def test_owner_can_fetch_their_own_card(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")
    response = test_client.get(f"/api/barcode/{owner1_id}", headers=headers)
    assert response.status_code == 200

    body = response.json()
    assert body["owner_code"].startswith("VTD-")
    assert body["card_path"].endswith(".png")
    assert body["barcode_path"].endswith(".png")
    assert body["qr_path"].endswith(".png")


def test_owner_cannot_fetch_another_owners_card(client):
    """A card is a credential; leaking it defeats the point of having one."""
    test_client, _, _, owner2_id = client
    headers = login(test_client, "owner1@test.example.com")
    response = test_client.get(f"/api/barcode/{owner2_id}", headers=headers)
    assert response.status_code == 403


def test_card_endpoint_requires_authentication(client):
    test_client, _, owner1_id, _ = client
    assert test_client.get(f"/api/barcode/{owner1_id}").status_code == 401


def test_card_images_are_served_over_the_media_mount(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")
    body = test_client.get(f"/api/barcode/{owner1_id}", headers=headers).json()

    for key in ("card_path", "barcode_path", "qr_path"):
        image = test_client.get(f"/media/{body[key]}")
        assert image.status_code == 200, key
        assert image.headers["content-type"] == "image/png"


def test_downloading_the_card_works_with_a_query_token(client):
    """An <img> or download link cannot send an Authorization header."""
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")
    token = headers["Authorization"].split()[1]

    response = test_client.get(f"/api/barcode/{owner1_id}/card?token={token}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000


def test_card_download_rejects_a_bad_token(client):
    test_client, _, owner1_id, _ = client
    assert test_client.get(
        f"/api/barcode/{owner1_id}/card?token=not-a-real-token"
    ).status_code == 401


def test_admin_can_regenerate_card_images(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.post(
        f"/api/barcode/regenerate/{owner1_id}", headers=headers
    )
    assert response.status_code == 200
    assert response.json()["owner_code"].startswith("VTD-")


def test_non_admin_cannot_regenerate_a_card(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")
    assert test_client.post(
        f"/api/barcode/regenerate/{owner1_id}", headers=headers
    ).status_code == 403


# =============================================================================
#  Scanning
# =============================================================================
def test_scanning_a_rendered_card_verifies_the_owner(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")

    db = SessionLocal()
    try:
        owner = crud.get_user(db, owner1_id)
        code, name = owner.owner_code, owner.full_name
    finally:
        db.close()

    card = render_card(code, name=name)
    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        files={"file": ("card.jpg", encode_jpeg(card), "image/jpeg")},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "decoded"
    assert body["verified"] is True
    assert body["owner_id"] == owner1_id
    assert body["owner_name"] == name
    assert body["is_active"] is True


def test_scanning_by_typed_code_tolerates_case_and_spacing(client):
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")

    db = SessionLocal()
    try:
        code = crud.get_user(db, owner1_id).owner_code
    finally:
        db.close()

    messy = code.lower().replace("-", " ")
    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        data={"code_override": messy},
        files={"file": ("unused.jpg", b"", "image/jpeg")},
    )
    body = response.json()
    assert body["status"] == "decoded"
    assert body["owner_id"] == owner1_id


def test_scanning_an_unknown_code_is_reported_distinctly(client):
    """A deleted account and a wrong card are different problems."""
    test_client, _, _, _ = client
    headers = login(test_client, "owner1@test.example.com")

    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        data={"code_override": "VTD-AAAA-2222"},
        files={"file": ("unused.jpg", b"", "image/jpeg")},
    )
    body = response.json()
    assert body["status"] == "unknown_code"
    assert body["verified"] is False
    assert body["owner_id"] is None


def test_scanning_a_foreign_barcode_is_reported_distinctly(client):
    """A retail barcode is not a membership card - it must not resolve."""
    test_client, _, _, _ = client
    headers = login(test_client, "owner1@test.example.com")

    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        data={"code_override": "9781234567897"},
        files={"file": ("unused.jpg", b"", "image/jpeg")},
    )
    body = response.json()
    assert body["status"] == "invalid_payload"
    assert body["verified"] is False


def test_scanning_a_blank_image_reports_no_barcode(client):
    test_client, _, _, _ = client
    headers = login(test_client, "owner1@test.example.com")

    blank = np.full((300, 400, 3), 255, np.uint8)
    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        files={"file": ("blank.jpg", encode_jpeg(blank), "image/jpeg")},
    )
    body = response.json()
    assert body["status"] == "no_barcode"
    assert body["verified"] is False


def test_scanning_a_disabled_account_does_not_verify(client):
    """
    Disabling an account is an administrative action; a stale card must not
    keep working around it.

    The scan is performed by an administrator, because a disabled owner can no
    longer log in - which is itself the point of disabling them.
    """
    test_client, admin_id, owner1_id, _ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")

    db = SessionLocal()
    try:
        owner = crud.get_user(db, owner1_id)
        code = owner.owner_code
        owner.is_active = False
        db.commit()
    finally:
        db.close()

    response = test_client.post(
        "/api/barcode/scan",
        headers=headers,
        data={"code_override": code},
        files={"file": ("unused.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "decoded"
    assert body["is_active"] is False
    assert "DISABLED" in body["message"]


def test_scanning_requires_authentication(client):
    test_client, _, _, _ = client
    response = test_client.post(
        "/api/barcode/scan",
        data={"code_override": "VTD-AAAA-2222"},
        files={"file": ("unused.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 401


def test_lookup_by_path_resolves_a_code(client):
    """A hardware reader with no camera sends text, not an image."""
    test_client, _, owner1_id, _ = client
    headers = login(test_client, "owner1@test.example.com")

    db = SessionLocal()
    try:
        code = crud.get_user(db, owner1_id).owner_code
    finally:
        db.close()

    response = test_client.get(f"/api/barcode/lookup/{code}", headers=headers)
    assert response.status_code == 200
    assert response.json()["owner_id"] == owner1_id


# =============================================================================
#  Alert delivery channels
# =============================================================================
def test_alert_channels_report_configuration(client):
    test_client, *_ = client
    headers = login(test_client, "owner1@test.example.com")
    response = test_client.get("/api/alerts/channels", headers=headers)
    assert response.status_code == 200

    body = response.json()
    assert "email" in body and "sms" in body
    assert isinstance(body["email"]["enabled"], bool)
    assert isinstance(body["sms"]["enabled"], bool)


def test_test_email_targets_only_the_configured_recipients(client, monkeypatch):
    """
    No address may be substituted for the configured ones.

    An earlier build appended a hard-coded personal address as a fallback, so a
    stock deployment mailed a stranger. Recipients must come only from
    configuration and the owner whose vehicle was involved.
    """
    from backend.api import alerts as alerts_module

    calls = {}
    monkeypatch.setattr(
        alerts_module,
        "send_email_alert",
        lambda **kwargs: (calls.update(kwargs), (True, None))[1],
    )
    monkeypatch.setattr(alerts_module.settings, "ALERT_EMAIL_TO", "security@example.com")
    monkeypatch.setattr(
        alerts_module.settings, "ADMIN_EMAIL", "admin@vtds.example.com"
    )

    test_client, *_ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.post("/api/alerts/test-email", headers=headers)

    assert response.status_code == 200
    assert calls["recipients"] == ["security@example.com"]


def test_dispatcher_never_invents_a_recipient(client, monkeypatch):
    """The default placeholder admin must not become a mail destination."""
    from backend.alerts import dispatcher

    db = SessionLocal()
    try:
        monkeypatch.setattr(dispatcher.settings, "ALERT_EMAIL_TO", "")
        monkeypatch.setattr(dispatcher.settings, "ADMIN_EMAIL", "admin@vtds.example.com")

        calls = {}
        monkeypatch.setattr(
            dispatcher,
            "send_email_alert",
            lambda **kwargs: (calls.update(kwargs), (True, None))[1],
        )
        monkeypatch.setattr(
            dispatcher, "send_sms_alert", lambda **kwargs: (True, None)
        )

        from backend.database.models import AlertStatus, ThreatLevel

        alert = crud.create_alert(
            db,
            threat_level=ThreatLevel.HIGH,
            threat_score=80,
            title="Recipient test",
            reason="No vehicle",
            camera_id="test-cam",
            status=AlertStatus.NEW,
        )
        dispatcher._send_notifications(db, alert)
    finally:
        db.close()

    assert calls["recipients"] == []


def test_test_sms_requires_recipients(client, monkeypatch):
    """Reporting a clear configuration error beats a silent no-op."""
    from backend.api import alerts as alerts_module

    test_client, *_ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    monkeypatch.setattr(alerts_module.settings, "ALERT_SMS_TO", "")

    db = SessionLocal()
    try:
        admin = crud.get_user(db, 1)
        admin.phone_number = None
        db.commit()
    finally:
        db.close()

    response = test_client.post("/api/alerts/test-sms", headers=headers)
    assert response.status_code == 503
    assert "recipient" in response.json()["detail"].lower()


def test_test_sms_requires_admin(client):
    test_client, *_ = client
    headers = login(test_client, "owner1@test.example.com")
    assert test_client.post("/api/alerts/test-sms", headers=headers).status_code == 403


# =============================================================================
#  Upload validation shared by the image endpoints
# =============================================================================
def test_unsupported_image_extension_is_415_not_400(client):
    """A GIF is a decodable image this system does not handle - hence 415."""
    test_client, *_ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.gif", b"GIF89a" + b"\x00" * 16, "image/gif")},
    )
    assert response.status_code == 415
    assert "not a supported image" in response.json()["detail"].lower()


def test_unsupported_extension_is_rejected_on_the_face_endpoint_too(client):
    test_client, *_ = client
    headers = login(test_client, "owner1@test.example.com")
    response = test_client.post(
        "/api/stream/verify-face-image",
        headers=headers,
        files={"file": ("face.gif", b"GIF89a" + b"\x00" * 16, "image/gif")},
    )
    assert response.status_code == 415


def test_corrupt_image_bytes_are_415_not_400(client):
    """
    Content-based validation catches a fake image before the extension does.

    `b"not-an-image"` carries no JPEG magic number, so it is not a damaged
    image - it is not an image at all, and 415 says so more accurately than the
    400 this returned when validation only inspected the file extension.
    """
    test_client, *_ = client
    headers = login(test_client, "admin@test.example.com", "Admin@12345")
    response = test_client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 415
