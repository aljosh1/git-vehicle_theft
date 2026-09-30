"""
Alert dispatcher - turns a threat assessment into a stored, notified incident.

Responsibilities, in order:

1. **Cooldown check.** A thief standing in frame for two minutes at 30 FPS would
   otherwise generate thousands of identical emails. `ALERT_COOLDOWN_SECONDS`
   suppresses repeats for the same camera + plate.
2. **Evidence capture.** Writes the annotated frame and the cropped face to
   `data/evidence/`, so the incident is reviewable after the fact.
3. **Database record.** Creates the `alerts` row *before* attempting delivery,
   so the incident survives even if SMTP and Twilio are both down.
4. **Notification fan-out.** Emails and texts the vehicle's owner (when known)
   plus the security desk, honouring each user's notification preferences.
5. **Delivery audit.** Records per-channel success and any error back onto the
   row - an alerting system that cannot prove delivery is not an alerting
   system.

Dispatch runs on a background thread so a slow SMTP handshake never stalls the
video pipeline.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.alerts.email_alert import send_email_alert
from backend.alerts.sms_alert import send_sms_alert
from backend.config import settings
from backend.core.security import random_filename
from backend.core.theft_engine import ThreatAssessment
from backend.database import crud
from backend.database.base import session_scope
from backend.database.models import Alert, ThreatLevel, User, Vehicle
from backend.utils.logger import get_logger

log = get_logger(__name__)


def _is_placeholder_email(address: str | None) -> bool:
    """
    True when an address is obviously not a real destination.

    Reserved documentation domains (`example.com`, `example.org`, `example.net`)
    and the project's own `*.example.com` bootstrap admin are filtered out so a
    stock `.env` cannot silently subscribe every demo alert to a mailbox that
    does not exist. A hard-coded personal address is *not* substituted here -
    silently mailing a stranger because the operator forgot to set
    `ALERT_EMAIL_TO` is exactly the kind of behaviour nobody expects from a
    security tool.
    """
    value = (address or "").strip().lower()
    if not value or "@" not in value:
        return True
    domain = value.rsplit("@", 1)[-1]
    # RFC 2606 reserves example.com/.org/.net, and this project's own bootstrap
    # admin lives under *.example.com. None of them can receive mail, so
    # subscribing an alert to one silently loses it.
    return (
        domain in {"example.com", "example.org", "example.net"}
        or domain.endswith(".example.com")
    )


def dispatch_alert(
    db: Session,
    assessment: ThreatAssessment,
    *,
    camera_id: str,
    frame: np.ndarray | None = None,
    face_crop: np.ndarray | None = None,
    detection_log_id: int | None = None,
    notify: bool = True,
) -> Alert | None:
    """
    Record and notify a theft incident.

    Args:
        db: active session.
        assessment: the verdict from `TheftEngine.assess()`.
        camera_id: which camera saw it.
        frame: annotated frame to save as evidence.
        face_crop: the detected face, saved separately - it is the single most
            useful image for identifying a suspect.
        detection_log_id: links the alert back to its detection row.
        notify: set False in tests to skip email/SMS entirely.

    Returns:
        The created `Alert`, or None if the cooldown suppressed it.
    """
    if not assessment.should_alert:
        return None

    # --- 1. cooldown -------------------------------------------------------
    if crud.recent_alert_exists(
        db,
        camera_id=camera_id,
        plate_number=assessment.plate_number,
        within_seconds=settings.ALERT_COOLDOWN_SECONDS,
    ):
        log.debug(
            "alert suppressed by the %ds cooldown (camera=%s plate=%s)",
            settings.ALERT_COOLDOWN_SECONDS, camera_id, assessment.plate_number,
        )
        return None

    # --- 2. evidence -------------------------------------------------------
    snapshot_path = _save_evidence(frame, prefix="frame")
    face_path = _save_evidence(face_crop, prefix="face")

    # --- 3. database record ------------------------------------------------
    vehicle: Vehicle | None = (
        crud.get_vehicle(db, assessment.matched_vehicle_id)
        if assessment.matched_vehicle_id
        else None
    )
    title = _build_title(assessment, vehicle)

    alert = crud.create_alert(
        db,
        threat_level=assessment.level,
        threat_score=assessment.score,
        title=title,
        reason=assessment.reason,
        triggers=assessment.triggers_csv,
        vehicle_id=assessment.matched_vehicle_id,
        plate_number=assessment.plate_number,
        person_status=assessment.person_status,
        camera_id=camera_id,
        site_name=settings.SITE_NAME,
        latitude=settings.SITE_LATITUDE or None,
        longitude=settings.SITE_LONGITUDE or None,
        snapshot_path=snapshot_path,
        face_image_path=face_path,
        detection_log_id=detection_log_id,
    )

    # --- 4/5. notify on a background thread --------------------------------
    if notify and (settings.ALERTS_EMAIL_ENABLED or settings.ALERTS_SMS_ENABLED):
        threading.Thread(
            target=_notify_in_background,
            args=(alert.id,),
            name=f"alert-notify-{alert.id}",
            daemon=True,
        ).start()

    return alert


def _notify_in_background(alert_id: int) -> None:
    """
    Send the notifications in a fresh session.

    A background thread must never reuse the caller's session - SQLAlchemy
    sessions are not thread-safe. `session_scope()` gives this thread its own.
    """
    try:
        with session_scope() as db:
            alert = crud.get_alert(db, alert_id)
            if alert is None:
                log.error("alert %s vanished before notification", alert_id)
                return
            _send_notifications(db, alert)
    except Exception:
        log.exception("background notification failed for alert %s", alert_id)


def _send_notifications(db: Session, alert: Alert) -> None:
    """Fan out to the owner and the security desk, then record the outcome."""
    vehicle = crud.get_vehicle(db, alert.vehicle_id) if alert.vehicle_id else None
    owner: User | None = vehicle.owner if vehicle else None

    # Recipients: the security desk always, plus the bootstrap admin when it is
    # a real address, plus the vehicle owner if they opted in.
    email_to = list(settings.email_recipients)
    email_seen = {item.lower() for item in email_to}
    admin_email = (settings.ADMIN_EMAIL or "").strip()
    if not _is_placeholder_email(admin_email) and admin_email.lower() not in email_seen:
        email_to.append(admin_email)
        email_seen.add(admin_email.lower())

    sms_to = list(settings.sms_recipients)
    if owner is not None:
        if owner.notify_email and owner.email and owner.email.lower() not in email_seen:
            email_to.append(owner.email)
            email_seen.add(owner.email.lower())
        if owner.notify_sms and owner.phone_number and owner.phone_number not in sms_to:
            sms_to.append(owner.phone_number)

    timestamp = alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    images = [Path(p) for p in (alert.face_image_path, alert.snapshot_path) if p]
    images = [settings.data_dir / p if not Path(p).is_absolute() else Path(p) for p in images]

    email_ok, email_error = send_email_alert(
        recipients=email_to,
        subject=f"[{alert.threat_level.value.upper()}] {alert.title}",
        threat_level=alert.threat_level.value,
        threat_score=alert.threat_score,
        reason=alert.reason,
        vehicle_description=vehicle.description if vehicle else None,
        plate_number=alert.plate_number,
        owner_name=owner.full_name if owner else None,
        timestamp=timestamp,
        site_name=alert.site_name,
        latitude=alert.latitude,
        longitude=alert.longitude,
        image_paths=images,
    )

    sms_ok, sms_error = send_sms_alert(
        recipients=sms_to,
        threat_level=alert.threat_level.value,
        threat_score=alert.threat_score,
        plate_number=alert.plate_number,
        vehicle_description=vehicle.description if vehicle else None,
        site_name=alert.site_name,
        timestamp=timestamp,
        reason=alert.reason,
        maps_url=alert.google_maps_url,
    )

    # Disabled channels are intentional configuration, not delivery failures.
    errors = []
    if settings.ALERTS_EMAIL_ENABLED and email_error:
        errors.append(f"email: {email_error}")
    if settings.ALERTS_SMS_ENABLED and sms_error:
        errors.append(f"sms: {sms_error}")
    crud.mark_alert_delivered(
        db,
        alert,
        email_sent=email_ok,
        sms_sent=sms_ok,
        error="; ".join(errors) if errors else None,
    )
    log.info(
        "alert %s notification: email=%s sms=%s", alert.id, email_ok, sms_ok
    )


def _save_evidence(image: np.ndarray | None, *, prefix: str) -> str | None:
    """
    Write an evidence image and return its path relative to `data/`.

    Storing a relative path keeps the database portable - the same row works
    after the project is moved or the deployment root changes.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None
    try:
        settings.ensure_directories()
        filename = f"{prefix}_{random_filename('.jpg')}"
        absolute = settings.evidence_dir / filename
        ok = cv2.imwrite(
            str(absolute), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not ok:
            log.error("cv2.imwrite refused to write %s", absolute)
            return None
        return f"evidence/{filename}"
    except Exception:
        log.exception("could not save %s evidence", prefix)
        return None


def _build_title(assessment: ThreatAssessment, vehicle: Vehicle | None) -> str:
    """
    A short headline for the email subject and the dashboard list.

    An armed or masked intruder leads, whatever the vehicle situation is. A
    subject line reading "Suspicious activity involving a car" is one a security
    guard will triage past at 3am; "CRITICAL: ARMED intruder at a car" is not.
    """
    if vehicle is not None:
        subject = f"{vehicle.plate_display} ({vehicle.description})"
    elif assessment.plate_number:
        subject = f"unregistered plate {assessment.plate_number}"
    else:
        subject = "an unidentified vehicle"

    lead = _threat_headline(assessment)
    if lead is None:
        if assessment.level is ThreatLevel.CRITICAL:
            return f"Critical threat involving {subject}"
        return f"Suspicious activity involving {subject}"

    if assessment.level is ThreatLevel.CRITICAL:
        return f"CRITICAL: {lead} involving {subject}"
    return f"{lead} involving {subject}"


def _threat_headline(assessment: ThreatAssessment) -> str | None:
    """The weapon/mask phrase that should head the alert, if there is one."""
    if assessment.weapons:
        weapon = assessment.weapons[0].replace("_", " ")
        plural = "s" if len(assessment.weapons) > 1 else ""
        return f"ARMED intruder carrying a {weapon}{plural}"
    if assessment.masks:
        mask = assessment.masks[0].replace("_", " ")
        return f"Masked intruder ({mask})"
    if assessment.occluded_faces:
        return "Intruder with face concealed"
    return None



def format_alert_for_console(alert: Alert) -> str:
    """Human-readable one-liner for the terminal during a live demonstration."""
    return (
        f"\n{'=' * 62}\n"
        f"  THREAT LEVEL: {alert.threat_level.value.upper()}  "
        f"(score {alert.threat_score}/100)\n"
        f"  REASON      : {alert.reason}\n"
        f"  PLATE       : {alert.plate_number or 'not read'}\n"
        f"  CAMERA      : {alert.camera_id} @ {alert.site_name}\n"
        f"  TIME        : {alert.created_at:%Y-%m-%d %H:%M:%S}\n"
        f"  EVIDENCE    : {alert.snapshot_path or 'none'}\n"
        f"{'=' * 62}"
    )