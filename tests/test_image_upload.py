"""
Image upload validation.

The regression that prompted these: a `.jfif` upload was rejected with 415.
JFIF is not a separate format - it is the "JFIF" interchange variant of the
JPEG container, emitted by phone cameras, messaging apps and screen-capture
tools. OpenCV decodes it perfectly; only the file extension was refusing it.

The deeper point is that extension-based validation is wrong in *both*
directions at once, and these tests cover both:

* it rejects valid images (JFIF, and anything with no extension at all, as sent
  by a browser `blob` upload);
* it accepts invalid ones (an MP4 renamed to `.jpg`), which then fail deep in
  the decoder as a confusing 400.

So validation is done on the bytes, and these tests assert that.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import UploadFile

_TMP_DB = Path(tempfile.gettempdir()) / "vtds_jfif_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.stream import _read_image_upload, sniff_image_format  # noqa: E402
from backend.database.base import Base, engine  # noqa: E402
from backend.database.models import UserRole  # noqa: E402
from backend.main import app  # noqa: E402


def encode(fmt: str, image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(fmt, image)
    assert ok
    return buffer.tobytes()


def to_jfif(jpeg_bytes: bytes) -> bytes:
    """
    Rewrite a JPEG with a JFIF APP0 marker.

    This is the real-world difference between `.jpg` and `.jfif`: same container,
    same compressed data, different identification segment. Anything that can
    read one can read the other.
    """
    return (
        b"\xff\xd8\xff\xe0"
        + b"\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + jpeg_bytes[4:]
    )


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    from backend.database import crud, schemas
    from backend.database.base import SessionLocal

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
    finally:
        db.close()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def headers(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "Admin@12345"},
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def photo() -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.random((80, 80, 3)) * 255).astype(np.uint8)


# =============================================================================
#  Sniffing
# =============================================================================
def test_jfif_is_recognised_as_jpeg(photo):
    jfif = to_jfif(encode(".jpg", photo))
    assert sniff_image_format(jfif[:16]) == "JPEG"


def test_png_is_recognised(photo):
    assert sniff_image_format(encode(".png", photo)[:16]) == "PNG"


def test_bmp_is_recognised(photo):
    assert sniff_image_format(encode(".bmp", photo)[:16]) == "BMP"


def test_tiff_is_recognised(photo):
    assert sniff_image_format(encode(".tiff", photo)[:16]) == "TIFF (little-endian)"


def test_webp_is_recognised(photo):
    assert sniff_image_format(encode(".webp", photo)[:16]) == "WebP"


def test_riff_container_that_is_not_webp_is_rejected(photo):
    """RIFF is a container; a WAV must not be mistaken for a WebP image."""
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
    assert sniff_image_format(wav[:16]) is None


@pytest.mark.parametrize(
    "junk",
    [
        b"GIF89a",                       # a GIF: unsupported, and correctly refused
        b"not an image at all",
        b"\x00" * 32,
        b"%PDF-1.4",                     # a PDF renamed to .jpg
        b"PK\x03\x04",                   # a zip / docx
    ],
)
def test_non_images_are_rejected(junk):
    assert sniff_image_format(junk[:16]) is None


# =============================================================================
#  End-to-end upload behaviour
# =============================================================================
def make_upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(data))


@pytest.mark.parametrize("filename", ["photo.jpg", "photo.jpeg", "photo.jfif", "photo.JFIF", ""])
def test_jpeg_variants_are_accepted(photo, filename):
    """
    The bug this fixes: a `.jfif` upload was refused with 415.

    Asserted against the validation helper rather than the whole endpoint,
    because reaching the analysis endpoint loads the YOLO models - which would
    test the detector rather than the format check, and take minutes.
    """
    payload = (
        to_jfif(encode(".jpg", photo))
        if filename.lower().endswith("jfif")
        else encode(".jpg", photo)
    )
    contents = asyncio.run(_read_image_upload(make_upload(filename, payload)))
    assert contents == payload


def test_a_jfif_file_with_no_extension_is_accepted(photo):
    """
    A browser `blob` upload can arrive with an empty filename.

    Rejecting on extension alone would refuse it; sniffing the bytes accepts it.
    """
    payload = to_jfif(encode(".jpg", photo))
    contents = asyncio.run(_read_image_upload(make_upload("", payload)))
    assert contents == payload


@pytest.mark.parametrize("fmt", [".png", ".bmp", ".tiff", ".webp"])
def test_other_supported_formats_are_accepted(photo, fmt):
    payload = encode(fmt, photo)
    contents = asyncio.run(_read_image_upload(make_upload(f"photo{fmt}", payload)))
    assert contents == payload


def test_a_mp4_renamed_to_jpg_is_rejected(client, headers):
    """
    The other direction: extension checking accepts this, content checking does not.

    Without sniffing, an MP4 named `vehicle.jpg` passes validation and then
    fails inside OpenCV as an unhelpful 400.
    """
    fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("vehicle.jpg", fake_mp4, "image/jpeg")},
    )
    assert response.status_code == 415
    assert "not a supported image" in response.json()["detail"].lower()


def test_a_gif_is_rejected_with_a_useful_message(client, headers):
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("animation.gif", b"GIF89a" + b"\x00" * 32, "image/gif")},
    )
    assert response.status_code == 415
    detail = response.json()["detail"]
    # The message must mention JFIF, because that is the confusion it exists
    # to resolve.
    assert "JFIF" in detail


def test_an_empty_file_is_400_not_415(client, headers):
    response = client.post(
        "/api/stream/analyse-image",
        headers=headers,
        files={"file": ("photo.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_face_verification_rejects_a_non_image_the_same_way(client, headers):
    """The content check is shared, so every image endpoint behaves alike."""
    response = client.post(
        "/api/stream/verify-face-image",
        headers=headers,
        files={"file": ("face.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32, "image/jpeg")},
    )
    assert response.status_code == 415
