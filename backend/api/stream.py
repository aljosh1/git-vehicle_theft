"""
Live monitoring: pipeline control and the MJPEG video stream.

    POST /api/stream/start     start a camera pipeline
    POST /api/stream/stop      stop it
    GET  /api/stream/status    which pipelines are running
    GET  /api/stream/mjpeg     the annotated live feed
    POST /api/stream/upload    analyse an uploaded video file

Why MJPEG
---------
The feed is served as `multipart/x-mixed-replace`, which a plain `<img>` tag
renders with no client-side decoder and no WebRTC signalling.  For a
single-viewer surveillance dashboard on a LAN that is the simplest thing that
works; WebRTC would only earn its complexity with many concurrent remote
viewers.
"""

from __future__ import annotations

import threading
import time
import difflib
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user, get_user_from_query_token, require_admin
from backend.config import settings
from backend.core.pipeline import Pipeline, _ACTIVE_PIPELINES, get_pipeline
from backend.core.security import random_filename
from backend.database.models import PersonStatus
from backend.recognition import face_recognizer as face
from backend.database import crud
from backend.database.base import get_db
from backend.database.models import User, Vehicle
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

_MAX_VIDEO_BYTES = 200 * 1024 * 1024
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_PIPELINE: Pipeline | None = None
_IMAGE_PIPELINE_LOCK = threading.Lock()

# Image suffixes the decoder is asked to handle. `.jfif` is the one that caused
# real rejections: it is not a distinct format but the JPEG container's
# "JFIF" interchange variant, and phone cameras, WhatsApp and screen-capture
# tools all emit it. OpenCV decodes it perfectly - the upload was being refused
# purely because of its file extension.
_ALLOWED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jfif", ".jpe",          # JPEG and its JFIF/JFEx variants
    ".png",                                    # PNG
    ".bmp", ".dib",                            # Windows bitmap
    ".webp",
    ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm",                    # Netpbm
    ".sr", ".ras",                              # Sun raster
    ".jp2",                                     # JPEG 2000
}

# Magic-number prefixes, checked when the filename does not settle the question
# (or is absent, as with a camera upload or a `blob` form field). Without this a
# `.jfif` file with no extension is rejected, and any junk file renamed to `.jpg`
# gets through to a confusing decode error.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),                       # also covers JFIF/JFEx
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF (little-endian)"),
    (b"MM\x00*", "TIFF (big-endian)"),
    (b"P1", "PGM"), (b"P2", "PGM"), (b"P3", "PPM"),
    (b"P4", "PBM"), (b"P5", "PGM"), (b"P6", "PPM"),
    (b"\x00\x00\x00\x0cjP  ", "JPEG 2000"),
    (b"\xff\x0a", "JPEG 2000"),
    (b"RIFF", "WebP / RIFF"),
)


def sniff_image_format(head: bytes) -> str | None:
    """
    Identify an image from its leading bytes.

    Returns a human-readable format name, or None when the bytes are not a
    recognised image. Used in preference to the filename wherever the filename
    is not authoritative, because a browser upload can arrive with a
    meaningless name and a mismatched `Content-Type`.
    """
    for magic, name in _IMAGE_MAGIC:
        if head.startswith(magic):
            if name == "WebP / RIFF":
                # RIFF is a container; the payload type decides. Accept only
                # WebP so a WAV or AVI is not mistaken for an image.
                return "WebP" if head[8:12] == b"WEBP" else None
            return name
    return None


async def _read_image_upload(file: UploadFile) -> bytes:
    """
    Validate and read an uploaded image, shared by every image endpoint.

    Rejection is based on the *content*, not the filename. A `.jfif` upload is a
    valid JPEG and is accepted; a `holiday.jpg` that is actually an MP4 is
    refused, which is the failure mode extension-based checking gets wrong in
    both directions.

    Three failure modes, three status codes, so a client can act on each:
    415 for a format the system cannot decode, 413 for an oversized file, 400
    for empty or corrupt bytes.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="image file is empty")
    if len(contents) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="image must be 20 MB or smaller",
        )

    # Trust the bytes. The declared content type and the filename are hints
    # from the client and are routinely wrong; the magic number is not.
    if sniff_image_format(contents[:16]) is None:
        suffix = Path(file.filename or "").suffix.lower()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"'{suffix or 'that file'}' is not a supported image format. "
                "Upload a JPEG, PNG, WebP, BMP or TIFF - JFIF files are "
                "accepted (they are a JPEG variant)."
            ),
        )
    return contents


def _image_pipeline() -> Pipeline:
    """Reuse loaded model weights across image requests."""
    global _IMAGE_PIPELINE
    if _IMAGE_PIPELINE is None:
        _IMAGE_PIPELINE = Pipeline(
            camera_id="dataset-image", source="image", enable_tracking=False, enable_alerts=False
        )
    return _IMAGE_PIPELINE


def _result_payload(result) -> dict:
    """Serialize a frame result for image-analysis clients."""
    return {
        "frame_number": result.frame_number,
        "processing_ms": round(result.processing_ms, 1),
        "fps_equivalent": round(result.fps, 2),
        "detections": [
            detection.to_dict()
            for detection in (
                result.vehicles
                + result.persons
                + result.animals
                + result.plates
                + result.weapons
                + result.masks
                + result.occluded_faces
            )
        ],
        "vehicle_count": len(result.vehicles),
        "person_count": len(result.persons),
        "animal_count": len(result.animals),
        "weapon_count": len(result.weapons),
        "mask_count": len(result.masks),
        "occluded_face_count": len(result.occluded_faces),
        # A flat list of names, because a client deciding whether to raise an
        # alarm wants "is there a gun" without walking nested detection dicts.
        "weapons": [d.label for d in result.weapons],
        "masks": [d.label for d in result.masks],
        "is_armed": result.is_armed,
        "is_concealed": result.is_concealed,
        "vehicle_analyses": [item.to_dict() for item in result.vehicle_analyses],
        "plate": {
            "text": result.plate_text,
            "confidence": result.plate_confidence,
            "in_database": result.plate_in_database,
            "flagged_stolen": result.vehicle_flagged_stolen,
        },
        "matched_vehicle_id": result.matched_vehicle_id,
        "person_status": result.person_status.value if result.person_status else None,
        "face_similarity": result.face_similarity,
        "matched_user_id": result.matched_user_id,
        "expected_owner_id": result.expected_owner_id,
        "owner_verification": (
            result.owner_verification.value if result.owner_verification else None
        ),
        "threat": {
            "score": result.threat_score,
            "level": result.threat_level_name,
            "reason": result.threat_reason,
            "triggers": result.threat_triggers,
            "is_armed": result.is_armed,
            "is_concealed": result.is_concealed,
            "should_alert": result.threat_score >= settings.THREAT_ALERT_THRESHOLD,
        },
    }


def _decode_image_upload(contents: bytes) -> np.ndarray | None:
    """Decode an uploaded image without relying on filename suffixes."""
    encoded = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is not None and frame.size > 0:
        return frame
    return None


def _owner_verification_label(matched_user_id: int | None, expected_owner_id: int | None) -> str:
    """
    Compare a recognised face against the identity the caller expected.

    The distinction between `not_applicable` and `not_recognized` is
    operational, not cosmetic: the first means "we were not asked to check", the
    second means "we checked and could not identify anyone". An operator must be
    able to tell those apart.
    """
    if expected_owner_id is None:
        return "not_applicable"
    if matched_user_id is None:
        return "not_recognized"
    if matched_user_id == expected_owner_id:
        return "verified"
    return "mismatch"


@router.post("/start")
def start_pipeline(
    camera_id: str = Body("cam-0"),
    source: str = Body("0", description="webcam index, RTSP URL, or file path"),
    enable_tracking: bool = Body(True),
    enable_alerts: bool = Body(True),
    persist_data: bool = Body(True, description="save detections/alerts to storage"),
    admin: User = Depends(require_admin),
):
    """
    Start surveillance on a camera.

    `source` accepts a webcam index ("0"), an RTSP/HTTP URL for CCTV, or a path
    to a video file - which is what makes the system demonstrable without any
    camera hardware present.
    """
    existing = get_pipeline(camera_id)
    if existing and existing.running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"pipeline {camera_id} is already running",
        )

    # An all-digits source is a device index, not a filename.
    resolved: str | int = int(source) if source.isdigit() else source

    try:
        pipeline = Pipeline(
            camera_id=camera_id,
            source=resolved,
            enable_tracking=enable_tracking,
            enable_alerts=enable_alerts,
            enable_persistence=persist_data,
        )
        pipeline.start()
    except Exception as exc:
        log.exception("could not start pipeline %s", camera_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"could not start the pipeline: {type(exc).__name__}: {exc}",
        ) from exc

    # Give the capture thread a moment to fail on a bad source, so the caller
    # gets a real error instead of a "started" that immediately dies.
    time.sleep(1.0)
    if not pipeline.running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"could not open video source {source!r}",
        )

    return {
        "running": True,
        "camera_id": camera_id,
        "source": source,
        "tracking": enable_tracking,
        "alerts": enable_alerts,
        "persist_data": persist_data,
    }


@router.post("/stop")
def stop_pipeline(
    camera_id: str = Body("cam-0", embed=True),
    admin: User = Depends(require_admin),
):
    pipeline = get_pipeline(camera_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"no pipeline {camera_id}")
    pipeline.stop()
    return {"running": False, "camera_id": camera_id}


@router.get("/status")
def pipeline_status(admin: User = Depends(require_admin)):
    """Every known pipeline, with its live throughput."""
    out = []
    for camera_id, pipeline in _ACTIVE_PIPELINES.items():
        _, result = pipeline.get_latest_frame()
        out.append(
            {
                "camera_id": camera_id,
                "running": pipeline.running,
                "source": str(pipeline.source),
                "fps": round(result.fps, 1) if result else 0.0,
                "frames_processed": pipeline._frame_count,
                "face_gallery_size": pipeline.face_gallery.size,
                "plate_detector_mode": pipeline.plate_detector.mode,
                "persist_data": pipeline.enable_persistence,
            }
        )
    return {"pipelines": out}


@router.get("/mjpeg")
def mjpeg_stream(
    camera_id: str = Query("cam-0"),
    user: User = Depends(get_user_from_query_token),
):
    """
    Annotated live feed as `multipart/x-mixed-replace`.

    Authenticated by query token because a browser will not attach an
    Authorization header to an `<img src>` request.
    """
    pipeline = get_pipeline(camera_id)
    if pipeline is None or not pipeline.running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"pipeline {camera_id} is not running - start it first",
        )

    def frame_generator():
        boundary = b"--frame\r\n"
        # Cap the send rate at the target FPS: pushing frames faster than the
        # pipeline produces them just re-sends identical JPEGs.
        min_interval = 1.0 / max(1, settings.TARGET_FPS)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        last_sent = 0.0

        while pipeline.running:
            now = time.perf_counter()
            if now - last_sent < min_interval:
                time.sleep(min_interval / 4)
                continue
            last_sent = now

            frame, _ = pipeline.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

        log.info("MJPEG stream for %s ended", camera_id)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"},
    )


@router.post("/analyse-image")
async def analyse_image(
    file: UploadFile = File(...),
    dispatch_alert: bool = Query(
        False, description="Store and send an alert when the image crosses the threshold"
    ),
    admin: User = Depends(require_admin),
):
    """Run the complete detection and theft-matching pipeline on one image."""
    contents = await _read_image_upload(file)

    frame = _decode_image_upload(contents)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="file is not a valid image")

    try:
        # Model inference objects are not guaranteed to be thread-safe. Serialise
        # image requests while retaining loaded weights between requests.
        with _IMAGE_PIPELINE_LOCK:
            annotated, result = _image_pipeline().analyse_image(
                frame, dispatch_alerts=dispatch_alert
            )
    except Exception as exc:
        log.exception("image analysis failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"image analysis failed: {type(exc).__name__}: {exc}",
        ) from exc

    settings.ensure_directories()
    analysis_dir = settings.data_dir / "analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_name = random_filename(".jpg")
    output_path = analysis_dir / output_name
    if not cv2.imwrite(
        str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    ):
        raise HTTPException(status_code=500, detail="could not store annotated image")

    payload = _result_payload(result)
    payload.update(
        {
            "filename": file.filename,
            "annotated_image": f"analyses/{output_name}",
            "alert_dispatch_enabled": dispatch_alert,
        }
    )
    return payload


@router.post("/verify-image-plate")
async def verify_image_plate(
    file: UploadFile = File(...),
    plate_input: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify that an uploaded image's detected plate matches the typed plate."""
    contents = await _read_image_upload(file)

    typed_normalised = Vehicle.normalise_plate(plate_input)
    if not typed_normalised:
        raise HTTPException(status_code=400, detail="plate_input is invalid")

    frame = _decode_image_upload(contents)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="file is not a valid image")

    with _IMAGE_PIPELINE_LOCK:
        _, result = _image_pipeline().analyse_image(frame, dispatch_alerts=False)

    detected_text = result.plate_text
    detected_normalised = Vehicle.normalise_plate(detected_text or "")

    typed_vehicle = crud.get_vehicle_by_plate(db, typed_normalised)
    detected_vehicle = (
        crud.get_vehicle_by_plate(db, detected_normalised)
        if detected_normalised
        else None
    )

    if not detected_normalised:
        verification = "no_plate_detected"
    elif detected_normalised == typed_normalised:
        verification = "verified_match"
    else:
        verification = "mismatch"

    return {
        "filename": file.filename,
        "input_plate": plate_input,
        "input_plate_normalised": typed_normalised,
        "input_plate_registered": typed_vehicle is not None,
        "detected_plate": detected_text,
        "detected_plate_normalised": detected_normalised or None,
        "detected_plate_registered": detected_vehicle is not None,
        "matched_vehicle_id": result.matched_vehicle_id,
        "verification": verification,
        "threat": {
            "score": result.threat_score,
            "level": result.threat_level_name,
            "reason": result.threat_reason,
        },
    }


@router.post("/verify-face-image")
async def verify_face_image(
    file: UploadFile = File(...),
    expected_owner_id: int | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Verify whether a supplied human image matches a registered owner.

    The response is written to be *explainable* rather than just true/false: a
    verdict a guard cannot interrogate is a verdict they will not trust. So it
    carries the matched identity, the observed similarity, the cutoff that
    produced the decision, and the gallery size - because "not recognised" on an
    empty gallery is a completely different statement from "not recognised"
    among two hundred enrolled owners.
    """
    contents = await _read_image_upload(file)

    frame = _decode_image_upload(contents)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="file is not a valid image")

    backend_name = face.init_backend()
    if backend_name is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no face recognition backend is available",
        )

    gallery_data = crud.load_face_gallery(db)
    gallery = face.FaceGallery(gallery_data)
    match = face.identify_person(frame, gallery)
    threshold = face.match_threshold()

    if match.status is PersonStatus.NO_FACE:
        verification = "no_face_detected"
    else:
        verification = _owner_verification_label(match.owner_id, expected_owner_id)

    # Resolve the identity so the UI can name the person rather than showing a
    # bare numeric id. Non-admins may not read the owner directory, so this is
    # deliberately limited to a first name.
    matched = crud.get_user(db, match.owner_id) if match.owner_id else None

    return {
        "filename": file.filename,
        "face_backend": backend_name,
        "person_status": match.status.value,
        "recognized": bool(match.status is PersonStatus.AUTHORIZED and match.owner_id),
        "matched_user_id": match.owner_id,
        "matched_user_name": matched.full_name if matched else None,
        "face_similarity": match.similarity,
        "match_threshold": threshold,
        "expected_owner_id": expected_owner_id,
        "owner_verification": verification,
        "gallery_size": gallery.size,
        "gallery_is_empty": gallery.size == 0,
        "face_bbox": list(match.bbox) if match.bbox else None,
    }


@router.post("/verify-plate-input")
def verify_plate_input(
    plate_input: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verify whether a typed plate belongs to a registered vehicle."""
    normalised = Vehicle.normalise_plate(plate_input)
    if not normalised:
        raise HTTPException(status_code=400, detail="plate_input is invalid")

    exact = crud.get_vehicle_by_plate(db, normalised)
    if exact is not None:
        return {
            "input_plate": plate_input,
            "input_plate_normalised": normalised,
            "verified": True,
            "match_type": "exact",
            "matched_vehicle_id": exact.id,
            "matched_plate": exact.plate_display,
            "owner_id": exact.owner_id,
            "flagged_stolen": bool(exact.is_flagged_stolen),
        }

    candidates = crud.list_vehicles(db, limit=300)
    best_vehicle = None
    best_ratio = 0.0
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, normalised, candidate.plate_number).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_vehicle = candidate

    fuzzy_ok = best_vehicle is not None and best_ratio >= float(settings.PLATE_FUZZY_THRESHOLD)

    return {
        "input_plate": plate_input,
        "input_plate_normalised": normalised,
        "verified": bool(fuzzy_ok),
        "match_type": "fuzzy" if fuzzy_ok else "none",
        "matched_vehicle_id": best_vehicle.id if fuzzy_ok and best_vehicle else None,
        "matched_plate": best_vehicle.plate_display if fuzzy_ok and best_vehicle else None,
        "owner_id": best_vehicle.owner_id if fuzzy_ok and best_vehicle else None,
        "fuzzy_score": round(best_ratio, 4),
        "fuzzy_threshold": float(settings.PLATE_FUZZY_THRESHOLD),
    }


@router.post("/verify-vehicle-image")
async def verify_vehicle_image(
    file: UploadFile = File(...),
    expected_plate: str | None = Form(None),
    user: User = Depends(get_current_user),
):
    """Verify vehicle registration from an uploaded vehicle image via plate OCR."""
    contents = await _read_image_upload(file)

    frame = _decode_image_upload(contents)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="file is not a valid image")

    expected_norm = Vehicle.normalise_plate(expected_plate or "") if expected_plate else None

    with _IMAGE_PIPELINE_LOCK:
        annotated, result = _image_pipeline().analyse_image(frame, dispatch_alerts=False)

    analyses = []
    for item in result.vehicle_analyses:
        detected_norm = Vehicle.normalise_plate(item.plate_text or "") if item.plate_text else None
        if expected_norm and detected_norm:
            match_state = "verified_match" if detected_norm == expected_norm else "mismatch"
        elif expected_norm and not detected_norm:
            match_state = "no_plate_detected"
        else:
            match_state = "verified_registered" if item.plate_in_database else "unverified"

        analyses.append(
            {
                "vehicle_track_id": item.vehicle.track_id,
                "plate_text": item.plate_text,
                "plate_confidence": item.plate_confidence,
                "plate_in_database": item.plate_in_database,
                "matched_vehicle_id": item.matched_vehicle_id,
                "expected_owner_id": item.expected_owner_id,
                "match_state": match_state,
            }
        )

    settings.ensure_directories()
    analysis_dir = settings.data_dir / "analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_name = random_filename(".jpg")
    output_path = analysis_dir / output_name
    cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return {
        "filename": file.filename,
        "expected_plate": expected_plate,
        "expected_plate_normalised": expected_norm,
        "annotated_image": f"analyses/{output_name}",
        "vehicle_count": len(result.vehicles),
        "plate_detected": bool(result.plate_text),
        "verified": any(item.get("plate_in_database") is True for item in analyses),
        "vehicle_analyses": analyses,
        "threat": {
            "score": result.threat_score,
            "level": result.threat_level_name,
            "reason": result.threat_reason,
        },
    }


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Query("upload-0"),
    admin: User = Depends(require_admin),
):
    """
    Analyse an uploaded video file.

    This is the demonstration path: no camera hardware needed, and the same
    clip can be replayed for a reproducible result during an evaluation.
    """
    contents = await file.read()
    if len(contents) > _MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="video must be 200 MB or smaller",
        )

    settings.ensure_directories()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    path = upload_dir / random_filename(suffix)
    path.write_bytes(contents)
    log.info("stored uploaded video %s (%.1f MB)", path.name, len(contents) / 1e6)

    existing = get_pipeline(camera_id)
    if existing and existing.running:
        existing.stop()

    pipeline = Pipeline(camera_id=camera_id, source=str(path), enable_tracking=True)
    pipeline.start()
    time.sleep(1.0)
    if not pipeline.running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="the uploaded file could not be opened as a video",
        )

    return {
        "running": True,
        "camera_id": camera_id,
        "filename": path.name,
        "stream_url": f"/api/stream/mjpeg?camera_id={camera_id}",
    }
