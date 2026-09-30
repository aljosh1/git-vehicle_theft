"""
Alerts package - notification dispatch.  *Interfaces only.*

Planned modules (Step 5):

    email_alert.py   SMTP (stdlib smtplib) with inline evidence attachments
    sms_alert.py     Twilio REST API
    dispatcher.py    fan-out to the owner + security desk, records per-channel
                     delivery on the `alerts` row, honours the cooldown window
"""
