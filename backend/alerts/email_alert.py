"""
Email alerts over SMTP.

Uses Python's stdlib `smtplib` + `email.message` - no third-party dependency.
Alerts are sent as multipart HTML with the evidence images **inline** (via
`cid:` references) rather than as attachments, because a security guard reading
the alert on a phone should see the thief's face without tapping anything.

Configuration lives in `.env` (`SMTP_*`, `ALERTS_EMAIL_ENABLED`).  With Gmail
you must use a 16-character App Password, not your account password - normal
passwords are rejected for SMTP.
"""

from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

from backend.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


def send_email_alert(
    *,
    recipients: list[str],
    subject: str,
    threat_level: str,
    threat_score: int,
    reason: str,
    vehicle_description: str | None,
    plate_number: str | None,
    owner_name: str | None,
    timestamp: str,
    site_name: str | None,
    latitude: float | None,
    longitude: float | None,
    image_paths: list[Path] | None = None,
) -> tuple[bool, str | None]:
    """
    Send one theft alert email.

    Returns:
        `(success, error_message)`. Never raises - a failed notification must
        not crash the surveillance pipeline, so the error is returned for
        recording on the `alerts` row instead.
    """
    if not settings.ALERTS_EMAIL_ENABLED:
        return False, "email alerts are disabled (ALERTS_EMAIL_ENABLED=false)"
    cleaned_recipients = _clean_recipients(recipients)
    if not cleaned_recipients:
        return False, "no email recipients configured"
    if not settings.SMTP_HOST or not settings.SMTP_PORT:
        return False, "SMTP host/port are not configured"

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = settings.ALERT_EMAIL_FROM or settings.SMTP_USER
        message["To"] = ", ".join(cleaned_recipients)
        # Mail clients surface this as a priority flag.
        if threat_level.lower() in {"high", "critical"}:
            message["X-Priority"] = "1"
            message["Importance"] = "high"

        maps_url = (
            f"https://maps.google.com/?q={latitude},{longitude}"
            if latitude is not None and longitude is not None
            else None
        )

        # Plain-text part: for clients that cannot render HTML, and for SMS
        # gateways that forward email as text.
        text_lines = [
            f"VEHICLE THEFT ALERT - {threat_level.upper()}",
            f"Threat score: {threat_score}/100",
            "",
            f"Reason: {reason}",
            "",
            f"Vehicle : {vehicle_description or 'Unidentified'}",
            f"Plate   : {plate_number or 'Not read'}",
            f"Owner   : {owner_name or 'Unregistered / unknown'}",
            f"Time    : {timestamp}",
            f"Location: {site_name or 'Unknown site'}",
        ]
        if maps_url:
            text_lines.append(f"Map     : {maps_url}")
        message.set_content("\n".join(text_lines))

        # Inline images get a Content-ID so the HTML can reference them.
        images = [p for p in (image_paths or []) if p and Path(p).exists()]
        image_cids: list[tuple[str, Path]] = [
            (f"evidence{index}", Path(path)) for index, path in enumerate(images)
        ]

        message.add_alternative(
            _build_html(
                threat_level=threat_level,
                threat_score=threat_score,
                reason=reason,
                vehicle_description=vehicle_description,
                plate_number=plate_number,
                owner_name=owner_name,
                timestamp=timestamp,
                site_name=site_name,
                maps_url=maps_url,
                image_cids=[cid for cid, _ in image_cids],
            ),
            subtype="html",
        )

        # Attach into the HTML part so the cid: references resolve.
        html_part = message.get_payload()[-1]
        for cid, path in image_cids:
            mime_type, _ = mimetypes.guess_type(path.name)
            maintype, _, subtype = (mime_type or "image/jpeg").partition("/")
            try:
                html_part.add_related(
                    path.read_bytes(),
                    maintype=maintype,
                    subtype=subtype,
                    cid=f"<{cid}>",
                    filename=path.name,
                )
            except OSError:
                log.exception("could not attach evidence file %s", path)

        _deliver(message)
        log.info("alert email sent to %s", ", ".join(cleaned_recipients))
        return True, None

    except smtplib.SMTPAuthenticationError:
        error = (
            "SMTP authentication failed - for Gmail you must use a 16-character "
            "App Password, not the account password"
        )
        log.error(error)
        return False, error
    except TimeoutError:
        error = "SMTP connection timed out"
        log.error(error)
        return False, error
    except smtplib.SMTPConnectError as exc:
        error = f"SMTP connect error: {exc.smtp_code} {exc.smtp_error!r}"
        log.error(error)
        return False, error
    except smtplib.SMTPRecipientsRefused:
        error = "all recipients were refused by the SMTP server"
        log.error(error)
        return False, error
    except Exception as exc:
        log.exception("failed to send alert email")
        return False, f"{type(exc).__name__}: {exc}"


def _clean_recipients(recipients: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in recipients:
        candidate = (item or "").strip()
        if not candidate:
            continue
        _, addr = parseaddr(candidate)
        addr = addr.strip()
        if not addr or "@" not in addr:
            continue
        key = addr.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(addr)
    return cleaned


def _deliver(message: EmailMessage) -> None:
    """Open the SMTP connection and send. Port 465 implies implicit TLS."""
    use_auth = bool(settings.SMTP_USER and settings.SMTP_PASSWORD)

    if settings.SMTP_PORT == 465:
        with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
            if use_auth:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            smtp.send_message(message)
        return

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
        smtp.ehlo()
        if settings.SMTP_USE_TLS:
            smtp.starttls()
            smtp.ehlo()
        if use_auth:
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        smtp.send_message(message)


def _build_html(
    *,
    threat_level: str,
    threat_score: int,
    reason: str,
    vehicle_description: str | None,
    plate_number: str | None,
    owner_name: str | None,
    timestamp: str,
    site_name: str | None,
    maps_url: str | None,
    image_cids: list[str],
) -> str:
    """Inline-styled HTML - email clients strip <style> blocks."""
    colours = {
        "critical": "#b91c1c",
        "high": "#ea580c",
        "medium": "#ca8a04",
        "low": "#0891b2",
        "none": "#64748b",
    }
    accent = colours.get(threat_level.lower(), "#b91c1c")

    rows = [
        ("Threat level", f"{threat_level.upper()} ({threat_score}/100)"),
        ("Reason", reason),
        ("Vehicle", vehicle_description or "Unidentified"),
        ("License plate", plate_number or "Not read"),
        ("Registered owner", owner_name or "Unregistered / unknown"),
        ("Date &amp; time", timestamp),
        ("Location", site_name or "Unknown site"),
    ]
    if maps_url:
        rows.append(("GPS", f'<a href="{maps_url}">View on Google Maps</a>'))

    table_html = "".join(
        f'<tr><td style="padding:8px 12px;background:#f8fafc;font-weight:600;'
        f'border-bottom:1px solid #e2e8f0;width:150px">{label}</td>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #e2e8f0">{value}</td></tr>'
        for label, value in rows
    )

    images_html = "".join(
        f'<div style="margin-top:12px"><img src="cid:{cid}" '
        f'style="max-width:100%;border-radius:6px;border:1px solid #e2e8f0" /></div>'
        for cid in image_cids
    )
    if images_html:
        images_html = (
            '<h3 style="margin:20px 0 4px;font-size:15px;color:#0f172a">'
            "Captured evidence</h3>" + images_html
        )

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:20px;background:#f1f5f9;
font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a">
  <div style="max-width:620px;margin:0 auto;background:#fff;border-radius:10px;
overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1)">
    <div style="background:{accent};color:#fff;padding:18px 22px">
      <div style="font-size:12px;letter-spacing:.08em;opacity:.85">
        VEHICLE THEFT DETECTION SYSTEM</div>
      <div style="font-size:21px;font-weight:700;margin-top:2px">
        {threat_level.upper()} THREAT DETECTED</div>
    </div>
    <div style="padding:22px">
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        {table_html}
      </table>
      {images_html}
      <p style="margin:22px 0 0;font-size:12px;color:#64748b;line-height:1.5">
        This is an automated alert. If this activity was authorised, mark the
        incident as a false positive in the dashboard so the detection
        thresholds can be reviewed.
      </p>
    </div>
  </div>
</body></html>"""
