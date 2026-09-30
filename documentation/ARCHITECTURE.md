# Architecture

## Data flow

```
Camera / video file
        │
        ▼
  VideoCapture (background thread, drops stale frames)
        │
        ▼
  YOLOv8 ──► vehicles (car/motorcycle/bus/truck) + persons
        │
        ├──► Plate detector ──► crop ──► grayscale ──► denoise ──►
        │                        threshold ──► EasyOCR ──► normalise
        │                                                     │
        │                                          get_vehicle_by_plate()
        │                                                     │
        ├──► Face detector ──► FaceNet embedding ──► cosine similarity
        │                       against gallery ──► Authorized/Unauthorized
        │                                                     │
        ▼                                                     ▼
  Theft engine ◄─────────────────────────────────────────────┘
        │  threat score 0-100
        ├──► detection_logs   (always)
        └──► alerts + dispatcher (score >= THREAT_ALERT_THRESHOLD, respecting
                                  the per-camera cooldown)
                    │
                    ├──► SMTP email  (evidence attached)
                    └──► Twilio SMS
```

## Threat scoring model (Step 5)

The brief specifies three trigger conditions. Rather than treating them as
independent booleans, they contribute additively to a 0–100 score, because a
single condition alone is a weak signal and the combination is a strong one:

| Condition | Contribution |
|-----------|--------------|
| Plate not found in the vehicle database | +35 |
| Unknown person near a registered vehicle | +40 |
| Face detected but unmatched (unauthorised) | +45 |
| Person loitering beyond `LOITER_SECONDS` | +15 |
| Vehicle flagged stolen by its owner | +100 (forces CRITICAL) |
| Recognised owner present | −50 (suppresses false positives) |

Bands: 0–19 none · 20–39 low · 40–59 medium · 60–79 high · 80–100 critical.
Alerts dispatch at `THREAT_ALERT_THRESHOLD` (default 60).

The owner-present subtraction is the important one. Without it, an authorised
owner whose plate happens to OCR poorly at night would trigger an alert every
time they collect their own car — and an alerting system that cries wolf gets
switched off, which is the real failure mode of systems like this.

## Performance strategy

Frame budget at 30 FPS is 33 ms. Measured rough costs on CPU:

| Stage | Cost |
|-------|------|
| YOLOv8n detection (640px) | 25–40 ms |
| Plate OCR (EasyOCR, one crop) | 60–120 ms |
| Face embedding (Facenet512) | 80–200 ms |

Running all three every frame is impossible on CPU. Mitigations, all
configurable in `.env`:

1. **Stage striding** — OCR and face recognition run every Nth frame.
2. **Result caching by track id** — once a plate is read for track 7, it is not
   re-read while track 7 remains in frame.
3. **Threaded capture** — the grabber thread keeps only the newest frame, so
   inference latency never turns into growing input lag.
4. **Resolution control** — inference at 640px regardless of capture size.

Reported FPS is measured over a rolling window and written to `detection_logs`,
so the throughput claims in the project report come from instrumentation rather
than estimation.
