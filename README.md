# A Real-Time Deep Learning Framework for Vehicle Theft Detection Using Object Detection

An intelligent surveillance system that detects vehicles, reads license plates,
recognises authorised owners, flags unauthorised persons attempting vehicle
access, and dispatches instant alerts with photographic evidence.

**Stack:** Python · FastAPI · SQLite · OpenCV · PyTorch · Ultralytics YOLOv8 ·
EasyOCR · YuNet/SFace face recognition · React + Tailwind CSS

---

## Status

| # | Module | Status |
|---|--------|--------|
| 1 | User authentication (bcrypt + JWT, role-based) | **Complete** |
| 2 | Vehicle registration database (5 tables) | **Complete** |
| 3 | Dataset image, uploaded video, and live-camera analysis | **Complete** |
| 4 | Vehicle detection (pretrained YOLOv8) | **Implemented; dataset evaluation pending** |
| 5 | License plate recognition (YOLOv8 + EasyOCR) | **Implemented; training/evaluation pending** |
| 6 | Face detection & recognition | **Complete** |
| 7 | Theft detection logic (0–100 threat score) | **Complete** |
| 8 | Alert notifications (SMTP + Twilio) | **Complete** |
| 9 | Web dashboard (React + Tailwind) | **Complete** |
| 10 | Training & evaluation scripts | **Complete; labelled dataset/results pending** |

The software pipeline is implemented, but the repository does not yet contain a
labelled image dataset or trained `models/license_plate.pt`. Do not present
vehicle accuracy, plate accuracy, OCR accuracy, mAP, or theft-classification
accuracy as project results until the experiments in the Training section have
been run on a fixed test split.

---

## Two things to know before you demo this

### 1. Measured throughput is 3.1 FPS on CPU, not 30

Benchmarked with `python training/evaluate.py --weights models/yolov8n.pt --fps-only`:

| Model | Mean latency | Throughput |
|-------|-------------:|-----------:|
| `yolov8n.pt` at 640 px, CPU | 320 ms | **3.1 FPS** |

The brief targets 25–30 FPS. Detection alone consumes ten times the 33 ms frame
budget on this hardware, before OCR or face recognition run. This is a hardware
limit, not a code defect — the same model runs 60+ FPS on a modest CUDA GPU.

Mitigations are built in and configurable in `.env` (`DETECT_EVERY_N_FRAMES`,
`OCR_EVERY_N_FRAMES`, `FACE_EVERY_N_FRAMES`, plus per-track result caching), so
the system degrades gracefully rather than stalling.

**In your report, quote the measured number and the reason.** "3.1 FPS on CPU;
30+ FPS requires GPU acceleration" is a defensible finding about a
hardware/accuracy trade-off. An unmeasured claim of 30 FPS is what gets
challenged in a viva. See `documentation/TRAINING.md` for the full analysis.

### 2. The face backend here is YuNet + SFace, not DeepFace

The brief specifies DeepFace/FaceNet. On this machine (Python 3.14) neither
installs — TensorFlow publishes no 3.14 wheels, and `facenet-pytorch` pins a
Pillow version that will not build.

The recogniser therefore falls through to **OpenCV YuNet (detection) + SFace
(128-d embeddings)**, which is a genuine research-grade path — SFace reports
~99.6% on LFW — not a degraded stand-in. Self-match similarity measures 1.000
and an unrelated face 0.134, verified in `tests/smoke_recognition.py`.

The backend chain is DeepFace → facenet-pytorch → YuNet+SFace → histogram, and
the active one is reported by `/api/info` so a demonstration never
misrepresents which model produced a result. Install Python 3.12 if you need
DeepFace specifically.

---

## Dataset image analysis

The dashboard accepts individual `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`
files in addition to videos and camera sources. Select **Analyse image** on the
Vehicle Analysis page. The backend forces vehicle detection, plate detection,
OCR, face recognition, database matching, and theft scoring once, then returns:

- the annotated image;
- vehicle/person detections and confidence scores;
- recognised plate text and database status;
- whether the matched vehicle was reported stolen;
- face status, threat score, verdict reason, and total processing time.

The corresponding authenticated API route is `POST /api/stream/analyse-image`.
Images are limited to 20 MB and annotated results are stored under
`data/analyses/`. Alert dispatch is disabled for ordinary dataset analysis; API
clients can explicitly set `dispatch_alert=true` for an operational check.

---

## Setup

### 1. Python

Use **Python 3.10–3.12**. The pinned PyTorch, Ultralytics, and DeepFace stack is
not supported by Python 3.14.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For an NVIDIA GPU (strongly recommended — see the FPS note above):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
then set `DEVICE=cuda` in `.env`.

### 2. Configure

```powershell
Copy-Item .env.example .env
```

Set `SECRET_KEY` and `ADMIN_PASSWORD` at minimum. Email and SMS stay disabled
until you fill in the SMTP / Twilio blocks.

### 3. Model weights

| File | Needed for | How |
|------|-----------|-----|
| `models/yolov8n.pt` | vehicles + persons | auto-downloads on first run |
| `models/face_detection_yunet.onnx` | face detection | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/face_recognition_sface.onnx` | face embeddings | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| `models/license_plate.pt` | plate localisation | optional — train it, see below |

Without `license_plate.pt` the system uses a classical OpenCV fallback
(edge-density + aspect-ratio filtering). Lower accuracy, but the pipeline runs
end to end — which is what makes the project demonstrable before the plate model
has finished training.

### Optional: Roboflow hosted plate detector

You can use a hosted Roboflow model instead of local YOLO plate weights.

1. Install dependency:

```powershell
pip install inference-sdk
```

2. Set backend and credentials in `.env`:

```env
PLATE_DETECTOR_BACKEND=roboflow
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_MODEL_ID=vehicle-license-plate-1hdcy/1
ROBOFLOW_API_KEY=your_private_api_key
```

`ROBOFLOW_API_KEY` must stay server-side only. Never commit it, and never expose
it in frontend code.

### 4. Initialise and run

```powershell
python -m backend.database.seed --demo
uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

- Dashboard: http://localhost:5173
- API docs: http://127.0.0.1:8000/docs
- Login: `admin@vtds.example.com` / `Admin@12345`

### 5. Test

```powershell
pytest tests/ -v
python tests/smoke_detector.py        # YOLOv8 + FPS
python tests/smoke_recognition.py     # face backend + OCR post-processing
```

---

## Training

A labelled dataset must be added before these commands can produce accuracy
results. Keep images or frames from the same source sequence in only one split
to prevent test leakage.

```powershell
python training/prepare_dataset.py --source C:\path\to\plate_images --split 70,20,10
python training/train_plate_detector.py --epochs 100 --install
python training/evaluate.py --weights models/license_plate.pt
```

`evaluate.py` writes `metrics.csv`, `metrics.json`, `metrics_bar.png`,
`fps_benchmark.png` and `summary.txt` to `training/outputs/evaluation_<timestamp>/`
— formatted for direct inclusion in a report. Full guide:
`documentation/TRAINING.md`.

---

## Threat scoring

The brief lists three trigger conditions. Treating them as independent booleans
would alarm on every weak signal, so each contributes additively to a 0–100
score:

| Condition | Points |
|-----------|-------:|
| Plate not in the database | +35 |
| Unknown person near a registered vehicle | +40 |
| Unauthorised face | +45 |
| Loitering beyond `LOITER_SECONDS` | +15 |
| Vehicle plate unreadable | +10 |
| Three or more people gathered | +10 |
| Night-time (only if already suspicious) | +5 |
| **Recognised owner present** | **−50** |
| Vehicle reported stolen | forces 100 |

Bands: 0–19 none · 20–39 low · 40–59 medium · 60–79 high · 80–100 critical.
Alerts fire at 60.

**No single condition alerts on its own** — that is a design invariant with a
regression test (`test_single_conditions_all_stay_below_the_alert_threshold`).
A stranger walking past a car park is Tuesday; a stranger *at* an unregistered
vehicle (75) is worth waking someone for.

The −50 owner suppression is the most important rule in the system. Without it,
an owner whose plate OCRs badly at night triggers an alert every time they
collect their own car — and an alerting system that cries wolf gets switched
off, which is the real failure mode of systems like this.

---

## Architecture

```
backend/
├── config.py          every tunable, loaded from .env
├── main.py            FastAPI app, CORS, lifespan, router mounting
├── api/               HTTP layer — 8 routers, no SQL, no model code
├── core/
│   ├── security.py    bcrypt + JWT
│   ├── theft_engine.py  scoring rules (I/O-free, so fast to test)
│   └── pipeline.py    threaded capture + per-frame orchestration
├── database/          engine, 5 ORM tables, schemas, CRUD, seed
├── detection/         YOLOv8 wrappers (vehicle, plate)
├── recognition/       OCR + face embedding/matching
├── alerts/            SMTP, Twilio, dispatcher with cooldown
└── utils/logger.py    rotating file + coloured console

frontend/src/
├── api/client.js      one place that knows about tokens
├── context/           auth state
├── components/        Layout, shared UI primitives
└── pages/             Login, Dashboard, LiveMonitoring, Vehicles, Owners, Alerts
```

Imports point strictly downwards: `api → core → detection/recognition/alerts →
database → config`. Nothing in `database/` imports a detector; nothing in
`detection/` imports a router. That is what makes each module independently
testable and lets YOLOv8 be swapped without touching anything above it.

More detail: `documentation/ARCHITECTURE.md`.

---

## Security and ethics

This is a research prototype. Before exposing it beyond localhost:

- Change `SECRET_KEY` and `ADMIN_PASSWORD`; never commit the real `.env`.
- Serve over HTTPS — JWTs are bearer credentials.
- **`/media` is currently unauthenticated** for demonstration convenience. It
  serves captured evidence containing identifiable faces and plate numbers. Put
  it behind the auth dependency before any real deployment.
- The MJPEG endpoint authenticates by query-string token, because a browser will
  not attach headers to an `<img src>` request. That token appears in server
  logs. It is used nowhere else for this reason.

**Surveillance footage of identifiable people is personal data.** Define a
retention period, and obtain whatever consent your institution's ethics process
requires before recording anyone who has not registered. A system that captures
and emails photographs of strangers has obligations that a lab exercise does
not, and an examiner is entitled to ask how you handled them.

Access control is enforced and regression-tested: owners cannot list, view or
edit another owner's vehicles, `?owner_id=` cannot be used to leak another
fleet, self-registration cannot grant an admin role, and login responses are
identical for unknown-email and wrong-password so the endpoint is not an
account-enumeration oracle.



All training data is stored in plate_detector_2000_416_5epochs.

Numerical data

Open results.csv in Excel. It contains one row per epoch with:

training box, classification, and DFL losses;
precision and recall;
mAP@0.5 and mAP@0.5:0.95;
validation losses;
learning rates;
elapsed training time.
In Excel, use File > Open, select the CSV, and then save it as an .xlsx workbook. You can also import it through Data > From Text/CSV.

Training configuration and summary

training_summary.json contains the model, five epochs, image size 416, batch size 4, CPU device, dataset, duration, completion time, and best-weights path.
args.yaml contains the complete YOLO training configuration and augmentation settings.
Graphs and report images

results.png shows losses and metrics across all epochs.
confusion_matrix.png shows prediction errors.
BoxPR_curve.png shows the precision-recall curve.
BoxF1_curve.png shows F1 score by confidence threshold.
BoxP_curve.png and BoxR_curve.png show precision and recall curves.
The val_batch*_pred.jpg files show visual detection predictions compared with the corresponding val_batch*_labels.jpg ground truth.
Trained model