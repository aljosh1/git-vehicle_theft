"""
SMS alerts via the Twilio REST API.

SMS is the fallback channel that matters: it arrives when the recipient has no
data connection, and it wakes a phone that has email notifications muted.  The
trade-off is a hard 1600-character limit (and billing per 160-character
segment), so the message is deliberately terse - the full evidence lives in the
email and the dashboard.
"""

from __future__ import annotations

from backend.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)

# Twilio segments messages every 160 chars (70 for unicode). Staying inside one
# or two segments keeps the cost predictable during a long demonstration.
_MAX_SMS_LENGTH = 320


def send_sms_alert(
    *,
    recipients: list[str],
    threat_level: str,
    threat_score: int,
    plate_number: str | None,
    vehicle_description: str | None,
    site_name: str | None,
    timestamp: str,
    reason: str,
    maps_url: str | None = None,
) -> tuple[bool, str | None]:
    """
    Send a theft alert SMS to each recipient.

    Returns:
        `(success, error_message)`. `success` is True if *at least one* message
        was accepted by Twilio - a single bad number must not mask delivery to
        the rest. Never raises.
    """
    if not settings.ALERTS_SMS_ENABLED:
        return False, "SMS alerts are disabled (ALERTS_SMS_ENABLED=false)"
    if not recipients:
        return False, "no SMS recipients configured"
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN):
        return False, "Twilio credentials are not configured"
    if not settings.TWILIO_FROM_NUMBER:
        return False, "TWILIO_FROM_NUMBER is not configured"

    try:
        from twilio.rest import Client
    except ImportError:
        error = "twilio is not installed; run: pip install twilio"
        log.error(error)
        return False, error

    body = _compose(
        threat_level=threat_level,
        threat_score=threat_score,
        plate_number=plate_number,
        vehicle_description=vehicle_description,
        site_name=site_name,
        timestamp=timestamp,
        reason=reason,
        maps_url=maps_url,
    )

    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    except Exception as exc:
        log.exception("could not create the Twilio client")
        return False, f"{type(exc).__name__}: {exc}"

    sent, errors = 0, []
    for number in recipients:
        try:
            message = client.messages.create(
                body=body, from_=settings.TWILIO_FROM_NUMBER, to=number
            )
            log.info("alert SMS queued to %s (sid=%s)", number, message.sid)
            sent += 1
        except Exception as exc:
            # Common causes: unverified number on a trial account, bad country
            # code, or insufficient balance. Record and keep going.
            log.exception("failed to send SMS to %s", number)
            errors.append(f"{number}: {type(exc).__name__}")

    if sent:
        return True, "; ".join(errors) if errors else None
    return False, "; ".join(errors) or "no messages were sent"


def _compose(
    *,
    threat_level: str,
    threat_score: int,
    plate_number: str | None,
    vehicle_description: str | None,
    site_name: str | None,
    timestamp: str,
    reason: str,
    maps_url: str | None,
) -> str:
    """
    Build the message body, truncating to fit the SMS budget.

    Field order is by operational value: what happened, which vehicle, where,
    when. The reason is truncated first because it is the longest and the least
    actionable on a phone screen.
    """
    parts = [
        f"[{threat_level.upper()} {threat_score}/100] Vehicle theft alert",
        f"Plate: {plate_number or 'not read'}",
    ]
    if vehicle_description:
        parts.append(f"Vehicle: {vehicle_description}")
    parts.append(f"Site: {site_name or 'unknown'}")
    parts.append(f"Time: {timestamp}")

    body = "\n".join(parts)

    remaining = _MAX_SMS_LENGTH - len(body) - len(maps_url or "") - 12
    if remaining > 30 and reason:
        trimmed = reason if len(reason) <= remaining else reason[: remaining - 3] + "..."
        body += f"\n{trimmed}"

    if maps_url:
        body += f"\n{maps_url}"

    if len(body) > _MAX_SMS_LENGTH:
        body = body[: _MAX_SMS_LENGTH - 3] + "..."
    return body
