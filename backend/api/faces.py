"""
Owner face enrolment.

    POST   /api/faces/{owner_id}/enroll   upload one or more face photos
    GET    /api/faces/{owner_id}          list an owner's enrolled embeddings
    DELETE /api/faces/embedding/{id}      remove one embedding

Enrolment quality determines recognition quality, so the endpoint reports per
file what happened rather than silently accepting a photo it could not use.
Three to five photos per owner - varied angle and lighting - is the practical
sweet spot.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.api.stream import sniff_image_format
from backend.config import settings
from backend.core.security import random_filename
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User, UserRole
from backend.recognition import face_recognizer as face
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _authorise(owner_id: int, user: User) -> None:
    if user.role is not UserRole.ADMIN and user.id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only manage your own face data",
        )


@router.post("/{owner_id}/enroll", response_model=schemas.FaceEnrollResult)
async def enroll_faces(
    owner_id: int,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Register face photos for an owner.

    For each file: decode → detect exactly one face → embed → store. A photo
    with no detectable face, or with several faces, is rejected with a reason
    instead of being stored as an unusable embedding.
    """
    _authorise(owner_id, user)

    owner = crud.get_user(db, owner_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Owner not found")

    backend_name = face.init_backend()
    if backend_name is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No face recognition backend is available on the server. "
                "See models/README.md."
            ),
        )

    settings.ensure_directories()
    enrolled, failed, messages = 0, 0, []

    for upload in files:
        name = upload.filename or "unnamed"
        try:
            contents = await upload.read()
            if len(contents) > _MAX_IMAGE_BYTES:
                failed += 1
                messages.append(f"{name}: larger than 8 MB")
                continue

            # Validate the bytes, not the extension. A `.jfif` enrolment photo
            # is a perfectly good JPEG and was being rejected outright, while a
            # renamed non-image would have been stored and failed later during
            # embedding.
            if sniff_image_format(contents[:16]) is None:
                failed += 1
                messages.append(f"{name}: not a supported image format")
                continue

            image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                failed += 1
                messages.append(f"{name}: not a decodable image")
                continue

            boxes = face.detect_faces(image)
            if not boxes:
                failed += 1
                messages.append(f"{name}: no face detected")
                continue
            if len(boxes) > 1:
                failed += 1
                messages.append(
                    f"{name}: {len(boxes)} faces found - enrol a photo of one person"
                )
                continue

            # Keep registration and surveillance preprocessing identical
            # (detection + alignment + normalisation) to avoid score drift.
            embedding, _ = face.embed_from_image(image)
            if embedding is None:
                failed += 1
                messages.append(f"{name}: embedding extraction failed")
                continue

            x1, y1, x2, y2 = boxes[0]

            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".jfif", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                suffix = ".jpg"
            filename = random_filename(suffix)
            (settings.faces_dir / filename).write_bytes(contents)

            # Face area relative to the image is a decent proxy for crop
            # quality: a face 8 px across embeds poorly however good the model.
            quality = float((x2 - x1) * (y2 - y1)) / float(image.shape[0] * image.shape[1])

            crud.add_face_embedding(
                db,
                owner_id=owner_id,
                vector=embedding,
                model_name=backend_name,
                detector_backend=backend_name,
                source_image=f"faces/{filename}",
                quality_score=round(quality, 4),
            )
            enrolled += 1
            messages.append(f"{name}: enrolled")

        except Exception as exc:
            failed += 1
            messages.append(f"{name}: {type(exc).__name__}")
            log.exception("face enrolment failed for %s", name)

    # Refresh every running pipeline so a newly enrolled owner is recognised
    # immediately, without a server restart.
    if enrolled:
        _refresh_running_galleries(db)

    log.info("enrolled %d face(s) for owner %s (%d failed)", enrolled, owner_id, failed)
    return schemas.FaceEnrollResult(
        owner_id=owner_id, enrolled=enrolled, failed=failed, messages=messages
    )


def _refresh_running_galleries(db: Session) -> None:
    try:
        from backend.core.pipeline import _ACTIVE_PIPELINES

        gallery = crud.load_face_gallery(db)
        for pipeline in _ACTIVE_PIPELINES.values():
            pipeline.face_gallery.rebuild(gallery)
    except Exception:
        log.exception("could not refresh the running pipelines' face galleries")


@router.get("/{owner_id}", response_model=list[schemas.FaceEmbeddingRead])
def list_owner_faces(
    owner_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorise(owner_id, user)
    return crud.list_face_embeddings(db, owner_id=owner_id, active_only=False)


@router.delete("/embedding/{embedding_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_embedding(
    embedding_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    records = crud.list_face_embeddings(db, active_only=False)
    record = next((r for r in records if r.id == embedding_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    _authorise(record.owner_id, user)
    crud.delete_face_embedding(db, record)
    _refresh_running_galleries(db)
