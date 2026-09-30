# Training and Evaluation Guide

Everything in this guide is driven by reproducible scripts:

| Script | Purpose |
|--------|---------|
| `training/prepare_dataset.py` | source images + metadata -> leakage-safe YOLO layout, manifest, and YAML |
| `training/train_plate_detector.py` | fine-tune one detector configuration |
| `training/run_plate_experiment.py` | train YOLOv8n and YOLO11n identically, compare them, and install the selected model |
| `training/evaluate.py` | detector precision/recall/F1/mAP/FPS plus OCR character and exact-plate metrics |
| `training/compare_detectors.py` | compare arbitrary vehicle or plate weights on the same untouched test split |
| `training/evaluate_theft.py` | end-to-end stolen/not-stolen precision, recall, F1, false-alert rate, and latency |

---

## What actually needs training

**Only the license plate detector.**

| Model | Status | Why |
|-------|--------|-----|
| Vehicle + person detection | pretrained, no training | `yolov8n.pt` already detects COCO classes 2/3/5/7 (car, motorcycle, bus, truck) and 0 (person) |
| License plate localisation | **must be trained** | COCO contains no license-plate class, so YOLOv8 cannot locate plates without fine-tuning |
| Face detection + recognition | pretrained, no training | YuNet + SFace ONNX (or DeepFace/FaceNet) ship as trained weights |
| OCR | pretrained, no training | EasyOCR ships with an English recognition model |

Until `models/license_plate.pt` exists the system uses the classical OpenCV
fallback in `backend/detection/plate_detector.py`. It works end to end, so the
project is demonstrable immediately — just with noticeably lower plate accuracy.

---

## Step 1 - Build a traceable dataset

Roughly 300 images is the floor for a usable model; 500–1000 is comfortable.

- **Roboflow Universe** — search "license plate"; many datasets are CC-licensed and already in YOLO format.
- **Your own footage** — extract frames from CCTV and annotate with [LabelImg](https://github.com/HumanSignal/labelImg) (save as *YOLO*, not Pascal VOC).
- **Augmentation** — `train_plate_detector.py` already applies HSV jitter, mild rotation, scaling and mosaic during training, so you do not need to pre-augment on disk.

If you use your own footage, capture the hard cases deliberately: night, rain,
oblique angles, dirty plates. A model trained only on clean daylight shots
scores well in validation and then fails in the exact conditions a theft
actually happens in.

Record every image in a metadata CSV based on `dataset/metadata_template.csv`:

| Column | Meaning |
|--------|---------|
| `image` | path relative to the source directory |
| `group_id` | source video, burst, or vehicle identity; one group is never split |
| `condition` | `day`, `night`, `rain`, `blur`, `dirt`, `angle`, `occlusion`, or `unspecified` |
| `plate_text` | ground-truth plate text for OCR evaluation |
| `source_name` | dataset/provenance identifier |

For Nigerian data, obtain owner/site permission, avoid publishing faces or plate
identities without a lawful basis, and retain a private collection register with
capture date, general location, device, consent, and annotator. Deliberately
sample all seven listed conditions and multiple cars, motorcycles, buses, and
trucks. Frames from one recording or the same vehicle must share a `group_id`.

### Annotation format

One `.txt` per image, one line per plate, normalised 0–1:

```
0 0.523 0.612 0.084 0.042
│ │     │     │     └─ height
│ │     │     └─────── width
│ │     └───────────── y_center
│ └─────────────────── x_center
└───────────────────── class_id (0 = license_plate)
```

---

## Step 2 - Prepare the fixed split

```powershell
python training/prepare_dataset.py --source C:\path\to\raw_images --metadata dataset\metadata.csv --split 70,20,10
```

Produces:

```
dataset/
|-- images/{train,val,test}/
|-- labels/{train,val,test}/
|-- manifest.csv
`-- plate_data.yaml
```

The group assignment is seeded (`--seed 42`), so re-running reproduces the
identical split. Whole groups are assigned together, and SHA-256 duplicate
content across splits is rejected. `manifest.csv` records the final path, split,
group, condition, plate text, source, and content hash. Keep this manifest fixed
for every compared model; do not tune hyperparameters on the test results.

Images with no matching label file are **skipped and listed**. That is
deliberate — an unlabelled image in the training set teaches the model that
plates do not exist in it, which actively harms recall.

---

## Step 3 - Train and select a plate detector

Run the controlled two-model experiment:

```powershell
python training/run_plate_experiment.py --data dataset/plate_data.yaml --models yolov8n.pt,yolo11n.pt --epochs 100
```

Both configurations use the same split, seed, image size, batch size, epochs,
patience, and augmentation. Selection is declared before evaluation: highest
mAP@0.5:0.95, then mAP@0.5, then FPS. The winner is copied to
`models/license_plate.pt`, and `installed_plate_model.json` records its source
weights, dataset, rule, and measured metrics.

Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 100 | 50 is often enough with early stopping |
| `--batch` | 16 | drop to 4–8 if you hit out-of-memory |
| `--imgsz` | 640 | 960 helps on small/distant plates, costs speed |
| `--device` | from `.env` | `cpu`, `cuda`, or `0` |
| `--patience` | 25 | early-stop after N epochs with no improvement |

**Expect CPU training to take hours.** 100 epochs on ~500 images is a few hours
on CPU versus roughly 20 minutes on a CUDA GPU. The script warns you about this
at startup. [Google Colab](https://colab.research.google.com) gives free GPU
time and is the pragmatic choice if you have no local GPU.

Artefacts land in `training/outputs/plate_detector/`:

```
weights/best.pt          ← the one you want
weights/last.pt
results.csv              per-epoch loss + metric history
results.png              training curves
confusion_matrix.png
PR_curve.png  F1_curve.png
training_summary.json
```

---

## Step 4 - Evaluate each component

```powershell
python training/evaluate.py --weights models/license_plate.pt --data dataset/plate_data.yaml --ocr-manifest dataset/manifest.csv
```

OCR output includes micro character accuracy, character error rate (Levenshtein
edits divided by ground-truth characters), exact full-plate accuracy, failed
reads counted as empty predictions, per-image predictions, and a breakdown by
capture condition.

Evaluate vehicle and plate detectors separately. For example:

```powershell
python training/compare_detectors.py --task vehicle --data dataset/vehicle_data.yaml --model yolov8n=models/yolov8n.pt --model yolo11n=models/yolo11n.pt
python training/compare_detectors.py --task plate --data dataset/plate_data.yaml --model yolov8n=training/outputs/plate_yolov8n/weights/best.pt --model yolo11n=training/outputs/plate_yolo11n/weights/best.pt
```

Speed-only benchmark, no labelled dataset needed:

```powershell
python training/evaluate.py --weights models/yolov8n.pt --fps-only
```

Writes to `training/outputs/evaluation_<timestamp>/`:

| File | Contents |
|------|----------|
| `metrics.csv` | the headline table — paste straight into the report |
| `metrics.json` | machine-readable, for comparing runs |
| `metrics_bar.png` | precision / recall / F1 / mAP bar chart |
| `fps_benchmark.png` | latency distribution histogram |
| `summary.txt` | formatted report section |

### Interpreting the numbers

Report measured values with confidence intervals or repeated-run variation when
possible. A high aggregate mAP does not by itself establish production readiness:
test-set provenance, class balance, calibration, night/rain subsets, latency,
false alerts, drift, privacy, and operational validation all matter. Compare
models on the same frozen test split and discuss failure cases rather than
assigning universal quality labels to arbitrary mAP bands.

**On "accuracy".** The brief lists accuracy as a metric, and `evaluate.py`
reports it — but labelled honestly as `accuracy_proxy_f1`. Object detection is
scored over predicted boxes, not over a fixed sample set, so there is no
well-defined true-negative count and therefore no meaningful classification
accuracy. **Lead with mAP@0.5 in your report** and mention F1; presenting a
made-up "accuracy: 96%" for a detector is the kind of claim an examiner will
probe.

### End-to-end theft decision evaluation

Create a balanced scenario manifest from `dataset/theft_scenarios_template.csv`.
Prepare the application database independently so the expected label is never
used as an inference input, then run:

```powershell
python training/evaluate_theft.py --manifest dataset/theft_scenarios.csv --split test
```

The output contains TP, TN, FP, FN, accuracy, precision, recall, F1, false-alert
rate, mean latency, p95 latency, per-condition metrics, and every sample-level
prediction. Report class counts with the percentages. Inspect false positives
and false negatives individually; an aggregate score cannot explain whether OCR,
database matching, association, or threat logic caused the error.

---

## Measured baseline on this machine

Recorded with `evaluate.py --fps-only --runs 20`, CPU, 640 px:

| Model | Mean latency | Throughput |
|-------|-------------:|-----------:|
| `yolov8n.pt` (vehicle + person) | 320 ms | **3.1 FPS** |

That is **well below the 25–30 FPS target in the brief**, and it is the single
most important performance fact about this deployment. Detection alone consumes
ten times the 33 ms frame budget, before OCR or face recognition run at all.

### Why, and what to do about it

The gap is hardware, not code. Options, in order of effect:

1. **Use a CUDA GPU** — the single biggest win. YOLOv8n at 640 px runs 60+ FPS on a modest GPU, roughly a 20× speedup. Install CUDA Torch:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   then set `DEVICE=cuda` in `.env`.
2. **Stride the detector** — `DETECT_EVERY_N_FRAMES=3` in `.env` processes every third frame. The display stays live; temporal resolution drops.
3. **Shrink the input** — inference at 320 px instead of 640 px is roughly 4× faster, at a real cost to small/distant plate recall.
4. **Export to ONNX or OpenVINO** — 2–3× faster on Intel CPUs with no accuracy loss:
   ```powershell
   yolo export model=models/yolov8n.pt format=openvino
   ```

**Report the measured number, not the target.** "3.1 FPS on CPU; 30+ FPS
requires GPU acceleration" is a defensible research finding about the
hardware/accuracy trade-off. Claiming 30 FPS without having measured it on the
demonstration machine is the claim that gets challenged in a viva.

The pipeline's striding and caching (`OCR_EVERY_N_FRAMES`,
`FACE_EVERY_N_FRAMES`, per-track result caching) exist precisely so the system
degrades gracefully instead of stalling when the hardware cannot keep up — see
`documentation/ARCHITECTURE.md`.

---

## Training the vehicle detector (optional)

The COCO-pretrained model already covers all four required vehicle classes, so
this is only worth doing if you need something COCO lacks — recognising a
specific *model* ("Toyota Camry" rather than "car"), for instance, which needs a
make/model dataset such as Stanford Cars.

```powershell
python training/train_plate_detector.py `
  --data dataset/vehicle_data.yaml `
  --classes car,motorcycle,truck,bus `
  --name vehicle_detector
```

Then point `VEHICLE_MODEL_PATH` in `.env` at the resulting weights.
