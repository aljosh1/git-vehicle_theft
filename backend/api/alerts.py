"""
Alert history and incident workflow.

    GET   /api/alerts             filterable history
    GET   /api/alerts/{id}
    PATCH /api/alerts/{id}        acknowledge / resolve / mark false positive

Marking an alert as a false positive is not cosmetic: it is the feedback signal
an evaluator should use to justify re-tuning the scoring weights in
`backend/core/theft_engine.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.alerts.email_alert import send_email_alert
from backend.alerts.sms_alert import send_sms_alert
from backend.api.deps import get_current_user, require_admin
from backend.config import settings
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import AlertStatus, ThreatLevel, User, UserRole
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


def _channel_status() -> dict[str, object]:
    """
    Which notification channels are actually armed.

    The dashboard greys out the "Send test" buttons from this, so an operator is
    never told an SMS went out when `ALERTS_SMS_ENABLED=false` means it silently
    did not.
    """
    return {
        "email": {
            "enabled": settings.ALERTS_EMAIL_ENABLED,
            "provider": settings.EMAIL_PROVIDER,
            "configured": bool(
                settings.email_recipients
                and (
                    (
                        settings.EMAIL_PROVIDER == "resend"
                        and settings.RESEND_API_KEY
                    )
                    or (settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD)
                )
            ),
            "recipients": settings.email_recipients,
        },
        "sms": {
            "enabled": settings.ALERTS_SMS_ENABLED,
            "configured": bool(
                settings.TWILIO_ACCOUNT_SID
                and settings.TWILIO_AUTH_TOKEN
                and settings.TWILIO_FROM_NUMBER
            ),
            "recipients": settings.sms_recipients,
        },
    }


@router.get("/channels")
def alert_channels(user: User = Depends(get_current_user)) -> dict[str, object]:
    """Report the delivery configuration of the alerting system."""
    return _channel_status()


@router.post("/test-email")
def send_test_email(admin: User = Depends(require_admin)) -> dict[str, object]:
    """Verify the configured SMTP connection without creating a theft alert."""
    recipients = settings.email_recipients or [admin.email]
    sent, error = send_email_alert(
        recipients=recipients,
        subject=f"[{settings.APP_NAME}] SMTP test",
        threat_level="low",
        threat_score=0,
        reason="This is a configuration test from the vehicle theft detection dashboard.",
        vehicle_description=None,
        plate_number=None,
        owner_name=admin.full_name,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        site_name=settings.SITE_NAME,
        latitude=None,
        longitude=None,
        image_paths=None,
    )
    if not sent:
        raise HTTPException(status_code=503, detail=error or "Email delivery failed")
    log.info("SMTP test email sent by admin %s to %s", admin.id, ", ".join(recipients))
    return {"sent": True, "recipients": recipients, "message": "Test email sent"}


@router.post("/test-sms")
def send_test_sms(admin: User = Depends(require_admin)) -> dict[str, object]:
    """
    Verify the Twilio connection without creating a theft alert.

    Mirrors `/test-email`: the fallback recipient is the administrator's own
    phone number when one is registered, otherwise the configured desk number.
    """
    recipients = settings.sms_recipients or ([admin.phone_number] if admin.phone_number else [])
    if not recipients:
        raise HTTPException(
            status_code=503,
            detail="no SMS recipients configured - set ALERT_SMS_TO or add a "
            "phone number to your profile",
        )
    sent, error = send_sms_alert(
        recipients=recipients,
        threat_level="low",
        threat_score=0,
        plate_number=None,
        vehicle_description=None,
        site_name=settings.SITE_NAME,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        reason="This is a configuration test from the vehicle theft detection dashboard.",
        maps_url=None,
    )
    if not sent:
        raise HTTPException(status_code=503, detail=error or "SMS delivery failed")
    log.info("test SMS sent by admin %s to %s", admin.id, ", ".join(recipients))
    return {"sent": True, "recipients": recipients, "message": "Test SMS sent"}


@router.get("", response_model=list[schemas.AlertRead])
def list_alerts(
    alert_status: AlertStatus | None = Query(None, alias="status"),
    threat_level: ThreatLevel | None = None,
    hours: int | None = Query(None, ge=1, le=24 * 365, description="only the last N hours"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Alert history.

    Owners see only alerts concerning their own vehicles; admins see all. Note
    that alerts with no matched vehicle (an unregistered plate, say) are visible
    only to admins - there is no owner to scope them to.
    """
    since = (
        datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    )
    alerts = crud.list_alerts(
        db,
        status=alert_status,
        threat_level=threat_level,
        since=since,
        skip=skip,
        limit=limit,
    )

    if user.role is not UserRole.ADMIN:
        own_vehicle_ids = {v.id for v in crud.list_vehicles(db, owner_id=user.id, limit=500)}
        alerts = [a for a in alerts if a.vehicle_id in own_vehicle_ids]

    return alerts


@router.get("/{alert_id}", response_model=schemas.AlertRead)
def get_alert(
    alert_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    alert = crud.get_alert(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    if user.role is not UserRole.ADMIN:
        vehicle = crud.get_vehicle(db, alert.vehicle_id) if alert.vehicle_id else None
        if vehicle is None or vehicle.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This alert concerns another owner's vehicle",
            )
    return alert


@router.patch("/{alert_id}", response_model=schemas.AlertRead)
def update_alert(
    alert_id: int,
    payload: schemas.AlertUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Acknowledge, resolve, or flag an alert as a false positive."""
    alert = get_alert(alert_id, user, db)      # reuses the ownership check
    updated = crud.update_alert_status(
        db,
        alert,
        status=payload.status,
        user_id=user.id,
        notes=payload.resolution_notes,
    )
    log.info(
        "alert %s marked %s by user %s", alert_id, payload.status.value, user.id
    )
    return updated
