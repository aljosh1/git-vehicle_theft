"""
Membership card barcodes.

GET  /api/barcode/{user_id}      the card, barcode and QR for one owner
POST /api/barcode/scan           read a card image and resolve it to an account

Design notes
------------
Images are rendered once and cached under `data/barcodes/`, keyed by the owner
code. The card is a *credential*, not a live view: regenerating it on every page
load would be wasted work, and a stable URL means the dashboard can render an
`<img>` straight from a JSON field with no extra round trip.

Scanning is a *lookup*, not an authentication. A barcode proves only "this card
was presented"; it says nothing about whether the person holding it is the owner.
The endpoint therefore reports who the card belongs to and whether that account
is active, and leaves the access decision to the caller - exactly the split that
face recognition makes in the other direction.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user, get_user_from_query_token, require_admin
from backend.config import settings
from backend.core.barcode import (
    BarcodeError,
    decode_barcode,
    encode_code128,
    encode_qr,
    normalise_owner_code,
    render_card,
)
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User, UserRole
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _authorise(owner_id: int, user: User) -> None:
    """A user may fetch their own card; only an administrator may fetch any."""
    if user.role is not UserRole.ADMIN and user.id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only view your own membership card",
        )


def _barcode_dir():
    settings.ensure_directories()
    path = settings.data_dir / "barcodes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_images(user: User) -> dict[str, str]:
    """
    Render the owner's card images if they are not already on disk.

    Returns the media-relative paths. Rendering is idempotent and keyed on the
    owner code, so the code is baked into the filename: a code change (which
    only happens if a row is re-backfilled) produces new files rather than
    silently serving a stale card.
    """
    directory = _barcode_dir()
    code = user.owner_code

    card_path = directory / f"card_{code}.png"
    barcode_path = directory / f"barcode_{code}.png"
    qr_path = directory / f"qr_{code}.png"

    try:
        if not card_path.exists():
            card = render_card(code, name=user.full_name)
            if not cv2.imwrite(str(card_path), card):
                raise BarcodeError(f"could not write {card_path}")
        if not barcode_path.exists():
            if not cv2.imwrite(str(barcode_path), encode_code128(code)):
                raise BarcodeError(f"could not write {barcode_path}")
        if not qr_path.exists():
            if not cv2.imwrite(str(qr_path), encode_qr(code)):
                raise BarcodeError(f"could not write {qr_path}")
    except BarcodeError:
        log.exception("failed to render barcode images for %s", code)
        raise HTTPException(
            status_code=500, detail="could not render the membership card"
        ) from None

    return {
        "card_path": f"barcodes/{card_path.name}",
        "barcode_path": f"barcodes/{barcode_path.name}",
        "qr_path": f"barcodes/{qr_path.name}",
    }


@router.get("/{user_id}", response_model=schemas.OwnerBarcodeRead)
def get_owner_barcode(
    user_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The membership card, its Code 128 barcode and its QR code."""
    _authorise(user_id, user)

    owner = crud.get_user(db, user_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Owner not found")

    paths = _ensure_images(owner)
    return schemas.OwnerBarcodeRead(
        owner_id=owner.id,
        owner_code=owner.owner_code,
        full_name=owner.full_name,
        **paths,
    )


@router.get("/{user_id}/card", response_class=FileResponse)
def download_owner_card(
    user_id: int,
    user: User = Depends(get_user_from_query_token),
    db: Session = Depends(get_db),
):
    """
    The printable card image itself.

    Authenticated from a query parameter rather than a header, because this is
    fetched by an `<img>` tag or a download link, and a browser will not attach
    an Authorization header to either. The token therefore appears in the
    access log for this route - acceptable for a read-only image behind a
    short-lived JWT, and the reason this dependency is not used anywhere else.
    """
    _authorise(user_id, user)
    owner = crud.get_user(db, user_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Owner not found")

    paths = _ensure_images(owner)
    return FileResponse(
        settings.data_dir / paths["card_path"],
        media_type="image/png",
        filename=f"membership-card-{owner.owner_code}.png",
    )


@router.post("/regenerate/{user_id}", response_model=schemas.OwnerBarcodeRead)
def regenerate_owner_barcode(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Re-render the card images, e.g. after the owner's name changed.

    Only the *images* are rebuilt - the owner code itself is not reissued here,
    because silently changing a person's credential would invalidate the card
    they are already carrying.
    """
    owner = crud.get_user(db, user_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Owner not found")

    for kind in ("card", "barcode", "qr"):
        (_barcode_dir() / f"{kind}_{owner.owner_code}.png").unlink(missing_ok=True)

    paths = _ensure_images(owner)
    log.info("regenerated membership card images for owner %s", owner.id)
    return schemas.OwnerBarcodeRead(
        owner_id=owner.id,
        owner_code=owner.owner_code,
        full_name=owner.full_name,
        **paths,
    )


@router.post("/scan", response_model=schemas.BarcodeScanResult)
async def scan_barcode(
    file: UploadFile = File(...),
    code_override: str | None = Form(
        None, description="verify this typed/scanned code instead of the image"
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Read a membership card and resolve it to an account.

    Accepts either an uploaded photo of the card or a code typed or piped in
    from a hardware reader. The four outcomes are reported separately because
    they mean different things to whoever is at the gate:

    * ``no_barcode``      - the image contains no readable symbol (a photo of
      the wrong thing, or too blurry). The guard should ask again.
    * ``invalid_payload`` - a symbol decoded, but it is not one of our codes.
      Probably a retail barcode, or a card from another system.
    * ``unknown_code``    - a well-formed code that no account holds. Either a
      deleted account or a fabricated card.
    * ``decoded``         - matched, and the response carries the owner.
    """
    if code_override and code_override.strip():
        payload = code_override.strip()
        return _resolve(payload, db, source="typed")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="image file is empty")
    if len(contents) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="image must be 20 MB or smaller",
        )

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="file is not a valid image")

    payload = decode_barcode(image)
    if not payload:
        log.info("barcode scan by %s found no readable symbol in %s",
                 user.id, file.filename)
        return schemas.BarcodeScanResult(
            status="no_barcode",
            verified=False,
            message=(
                "No barcode or QR code could be read from that image. "
                "Hold the card flat and fill the frame with it."
            ),
        )
    return _resolve(payload, db, source="image")


def _resolve(payload: str, db: Session, *, source: str) -> schemas.BarcodeScanResult:
    """Map a decoded payload onto an account, or explain why it does not."""
    normalised = normalise_owner_code(payload)
    if normalised is None:
        log.info("barcode scan (%s) decoded a foreign payload: %r", source, payload)
        return schemas.BarcodeScanResult(
            status="invalid_payload",
            verified=False,
            payload=payload,
            message=(
                "That barcode is not a membership card for this system "
                f"(read: {payload!r})."
            ),
        )

    owner = crud.get_user_by_owner_code(db, normalised)
    if owner is None:
        log.warning("barcode scan found unregistered code %s", normalised)
        return schemas.BarcodeScanResult(
            status="unknown_code",
            verified=False,
            payload=payload,
            normalised_code=normalised,
            message=(
                f"Code {normalised} is well-formed but belongs to no account. "
                "It may have been deleted, or the card may be fabricated."
            ),
        )

    dto = crud.to_user_read(db, owner)
    log.info(
        "barcode scan (%s) matched %s -> %s", source, normalised, owner.email
    )
    return schemas.BarcodeScanResult(
        status="decoded",
        verified=True,
        payload=payload,
        normalised_code=normalised,
        owner_id=owner.id,
        owner_name=owner.full_name,
        owner_email=owner.email,
        is_active=owner.is_active,
        vehicle_count=dto.vehicle_count,
        message=(
            f"Valid card for {owner.full_name}"
            + ("" if owner.is_active else " - but the account is DISABLED")
        ),
    )


@router.get("/lookup/{code}", response_model=schemas.BarcodeScanResult)
def lookup_code(
    code: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Resolve a code from the URL path, for a hardware reader that posts text.

    Kept separate from `/scan` so a reader with no camera can integrate without
    having to send a dummy image file.
    """
    return _resolve(code, db, source="path")
