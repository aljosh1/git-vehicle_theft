# Methodology

## 1. Research Design

This project adopted a design-and-experimental methodology to develop and assess a real-time vehicle theft detection system. The work combined computer vision, biometric recognition, database matching, rule-based decision support, and web-based monitoring. The central objective was not merely to detect a vehicle, but to combine multiple observations into an explainable theft-risk decision.

The methodology was organised into five connected phases:

1. system architecture and component selection;
2. dataset collection, annotation, auditing, and preprocessing;
3. license plate detector training and model selection;
4. integration and system implementation; and
5. component-level and end-to-end evaluation.

A modular approach was used so that each component could be evaluated independently before the complete image-to-alert pipeline was tested. This was important because an incorrect final decision may originate from vehicle detection, plate localisation, OCR, face matching, person-to-vehicle association, database records, or the threat-scoring rules.

## 2. System Architecture

### 2.1 Architectural style

The system uses a layered, modular client-server architecture. The React frontend provides authentication, live monitoring, vehicle and owner management, detection history, and alert views. The FastAPI backend exposes authenticated HTTP and streaming endpoints and coordinates the computer-vision pipeline. SQLite provides persistent storage for users, registered vehicles, face data, detection logs, and alerts.

The backend follows a downward dependency structure:

```text
Frontend dashboard
        |
        v
FastAPI routes and authentication
        |
        v
Real-time pipeline orchestrator
        |
        +--> vehicle/person detector
        +--> plate detector --> OCR
        +--> face detector --> face embedding and matching
        +--> database lookup and association
        +--> theft-risk engine
        |
        +--> detection logs and evidence storage
        +--> email/SMS alert dispatcher
```

HTTP routing, detection, recognition, persistence, decision logic, and notification are separated into dedicated modules. This makes individual components replaceable and testable. For example, the plate detector exposes one output format regardless of whether local YOLO weights, a hosted Roboflow model, or the OpenCV fallback is used.

### 2.2 Input and processing architecture

The system accepts live cameras, network streams, uploaded video, and individual dataset images. OpenCV `VideoCapture` acquires video frames. For live sources, a background capture thread keeps the newest frame and discards stale frames when inference cannot match the camera rate. For video files, frames are queued without intentional dropping so that the complete recording can be analysed.

Each processed frame passes through the following stages:

1. **Vehicle and person detection:** A COCO-pretrained YOLOv8n detector identifies persons and the four relevant vehicle classes: car, motorcycle, bus, and truck. ByteTrack identifiers are used when tracking is enabled.
2. **Plate localisation:** Plate detection is performed inside each vehicle crop to reduce background false positives. The preferred path is a locally fine-tuned YOLO detector. A hosted Roboflow backend is configurable, while an edge-density, contour, area, and aspect-ratio OpenCV method provides an offline fallback.
3. **Optical character recognition:** The selected plate crop is converted to grayscale, enhanced with CLAHE, denoised with a bilateral filter, and binarised using Otsu thresholding. EasyOCR is the primary OCR engine and PaddleOCR is the fallback. Output is upper-cased, stripped of punctuation, corrected for common contextual character confusions, and compared with registered plate numbers. A similarity threshold permits a minor OCR error during database matching.
4. **Face recognition:** Faces associated with detected persons are converted to embeddings and compared with the registered owner gallery using cosine similarity. The preferred backend chain is DeepFace, facenet-pytorch, and OpenCV YuNet with SFace. On the current environment, YuNet and SFace provide the practical detection and 128-dimensional embedding path.
5. **Spatial association:** Each nearby person is assigned to at most one vehicle using intersection-over-union and normalised centre distance. This prevents a person's identity or risk evidence from being incorrectly applied to every vehicle in a crowded frame.
6. **Threat assessment:** Plate status, stolen-vehicle status, face status, proximity, loitering, crowd size, and time of day are combined into an explainable score from 0 to 100.
7. **Persistence and alerting:** Relevant detections and their processing times are written to the database. Scores at or above the configured alert threshold generate an evidence record and may be dispatched through SMTP email or Twilio SMS, subject to a per-camera cooldown.

### 2.3 Theft decision model

The decision stage is rule-based rather than an opaque binary classifier because the available theft scenarios are limited and each alert must be explainable to an operator. The implemented contributions are:

| Evidence | Score contribution |
|---|---:|
| Plate not registered | +35 |
| Unknown person near a vehicle | +40 |
| Unauthorised face | +45 |
| Loitering beyond the configured duration | +15 |
| Plate could not be read | +10 |
| Three or more people near a vehicle | +10 |
| Night-time when another suspicious condition exists | +5 |
| Recognised owner present | -50 |
| Vehicle previously reported stolen | score forced to 100 |

The score is bounded to 0-100 and mapped to none (0-19), low (20-39), medium (40-59), high (60-79), or critical (80-100). The default alert threshold is 60. The weights were configured so that a single weak observation remains below the alert threshold, while corroborating evidence triggers escalation. Owner recognition suppresses false alarms, whereas a sighting of a vehicle already reported stolen overrides all other evidence.

## 3. Dataset Collection and Preprocessing

### 3.1 Data sources

The license plate localisation dataset was obtained as a YOLOv8-format export of the Roboflow License Plate v3 dataset. The provider's original train, validation, and test directories were retained rather than randomly re-splitting the exported data. The dataset contains one class, `license_plate`, with each object represented by a normalised YOLO bounding box.

To improve local relevance, the methodology also supports supplementary images extracted from authorised CCTV footage and locally captured road or parking-area scenes. Additional sampling should deliberately include:

- day and night scenes;
- rain, shadows, glare, and low contrast;
- motion blur and compression artefacts;
- frontal and oblique viewing angles;
- dirty, partially occluded, and distant plates; and
- cars, motorcycles, buses, and trucks.

Collection must be conducted with site or owner permission. Capture provenance, date, general location, device, condition, source, vehicle or sequence group, and plate transcription should be recorded in the metadata template. Faces and registration identifiers are personally identifiable data; therefore, access restrictions, a retention schedule, and institutional ethical approval or another lawful basis are required before operational collection.

### 3.2 Annotation

Each plate is annotated with one class identifier and four normalised coordinates:

```text
class_id x_center y_center width height
```

Class `0` denotes a license plate. Labels are paired with image files using matching stems. For OCR evaluation, the verified plate transcription is stored separately in the metadata manifest. Theft scenario images use a second manifest containing a sample identifier, image path, expected stolen/not-stolen outcome, split, and capture condition.

### 3.3 Dataset audit and quality control

Before training, the Roboflow export is audited programmatically. The audit performs the following checks:

1. confirms that train, validation, and test image and label directories exist;
2. verifies one-to-one image-label pairing;
3. confirms that images can be decoded and have valid dimensions;
4. validates that every annotation contains five numeric YOLO fields;
5. rejects non-integer or unexpected class identifiers;
6. checks that centres and box dimensions are normalised and remain inside the image;
7. computes SHA-256 hashes for images and labels;
8. detects exact image duplication across splits; and
9. derives source group identifiers by removing Roboflow augmentation hashes so related variants can be identified.

The audit creates a traceable manifest containing split, paths, group identifier, dimensions, annotation count, hashes, source URL, and licence. Exact cross-split duplicates cause validation to fail. Group overlap is also reported because augmented or near-duplicate versions of one source image across different splits can inflate evaluation results even when their binary hashes differ.

For newly collected local data, all frames from the same video, image burst, or vehicle identity should share one `group_id`. A seeded group-wise 70:20:10 train-validation-test allocation is recommended. Whole groups must remain in one split, and the final test set must be frozen before model tuning.

### 3.4 Image preprocessing and augmentation

The detector receives images resized by Ultralytics to a configured input size of 640 pixels while preserving the expected model preprocessing. Augmentation is applied online to the training split only. The configured transformations are:

- hue 0.015, saturation 0.7, and value 0.4 jitter;
- rotation up to 10 degrees;
- translation 0.1 and scale 0.5;
- shear 2.0 and perspective 0.0005;
- horizontal flipping probability 0.5;
- no vertical flipping; and
- mosaic augmentation enabled.

These settings model plausible lighting, scale, and camera-angle variation without producing unrealistic upside-down plates. Validation and test images are not augmented. OCR preprocessing is evaluated independently on ground-truth plate crops to separate recognition errors from plate localisation errors.

## 4. Model Training and Configuration

### 4.1 Model selection strategy

Only the license plate localiser requires project-specific training. Vehicle and person detection use pretrained COCO weights because the required classes already exist in COCO. EasyOCR/PaddleOCR and the face-recognition backends also use pretrained models. This transfer-learning strategy reduces data and computation requirements while focusing training on the domain component absent from COCO.

Two lightweight detector configurations, YOLOv8n and YOLO11n, are trained as a controlled comparison. Nano models were selected because the system is intended for real-time use and must balance localisation accuracy against latency.

### 4.2 Controlled training configuration

Both candidate plate detectors are fine-tuned from pretrained weights using identical settings:

| Parameter | Configuration |
|---|---:|
| Task | Single-class plate object detection |
| Input size | 640 x 640 |
| Maximum epochs | 100 |
| Batch size | 16, reduced if memory requires |
| Random seed | 42 |
| Deterministic mode | Enabled |
| Early-stopping patience | 25 epochs |
| Optimisation and loss | Ultralytics defaults for the selected detector |
| Training augmentation | Identical settings described in Section 3.4 |
| Device | CPU, CUDA, or GPU index from configuration |

During training, bounding-box regression, classification, and distribution focal losses are monitored alongside validation precision, recall, mAP@0.5, and mAP@0.5:0.95. Early stopping limits overfitting and unnecessary computation. The best checkpoint rather than the final epoch is retained.

### 4.3 Model selection and test isolation

Candidate models are compared on the same validation split. The selection rule is declared before test evaluation:

1. highest mAP@0.5:0.95;
2. highest mAP@0.5 as the first tie-breaker; and
3. highest throughput as the second tie-breaker.

Only the selected model is evaluated once on the untouched test split. Its checkpoint is installed as `models/license_plate.pt`, and a JSON receipt records the model source, dataset, selection rule, validation metrics, and final test metrics. This prevents choosing a model based on test-set performance.

Training produces the best and last weights, per-epoch results, loss and metric plots, precision-recall and F1 curves, a confusion matrix, and a machine-readable training summary. These artefacts provide reproducibility and evidence for subsequent analysis.

## 5. Integration and System Implementation

### 5.1 Backend integration

The trained model is loaded through the plate-detector abstraction. If the checkpoint is unavailable, the system can use an authenticated server-side Roboflow inference client or the OpenCV fallback. API credentials remain in environment configuration and are never exposed to the frontend.

For every detected vehicle, the pipeline crops the vehicle region, detects its plate, reads the most confident plate crop, and queries the database. Exact matching is  attempted first, followed by an 80% string-similarity check to tolerate a single OCR error. The database result provides the vehicle identifier, registered status, owner relationship, and stolen flag.

Face recognition is run within detected person regions. Registered owner embeddings are loaded into an in-memory gallery and L2-normalised. A face embedding is classified as authorised only when its cosine similarity reaches the configured threshold; otherwise, it contributes unauthorised or unknown evidence according to the recognition outcome.

### 5.2 Real-time performance controls

The following controls reduce repeated computation and latency:

- threaded camera capture with a queue of two frames;
- newest-frame retention for live sources;
- ordered processing for video files;
- configurable OCR and face-recognition frame strides;
- plate and face result caching by tracking identifier;
- model warm-up before operational inference;
- inference-device configuration; and
- rolling-window measurement of actual frames per second.

A synchronous image-analysis path forces all recognition stages once and disables alerts by default, making it appropriate for dataset experiments. The live path maintains recent frames and sacrifices temporal resolution when hardware is slower than the camera rate.

### 5.3 Application and operational workflow

FastAPI provides JWT-authenticated routes for users, vehicles, owners, detections, alerts, statistics, image analysis, and live streaming. Role-based access control prevents owners from viewing or modifying another owner's records. The React dashboard consumes these APIs and displays the annotated stream, current risk score, detection history, registered assets, and alert records.

When the threat threshold is reached, the backend stores the assessment, trigger explanation, camera identifier, plate and face results, processing time, and evidence image. The dispatcher can then send email and SMS notifications. A cooldown prevents repeated notifications from adjacent frames of the same event.

## 6. Evaluation Procedure and Metrics

Evaluation is conducted at four levels: dataset integrity, individual perception components, integrated theft decisions, and operational performance.

### 6.1 Object detection metrics

The vehicle and license plate detectors are evaluated independently on labelled test data using:

- **Precision:** `TP / (TP + FP)`, measuring the proportion of predicted objects that are correct.
- **Recall:** `TP / (TP + FN)`, measuring the proportion of ground-truth objects detected.
- **F1 score:** `2PR / (P + R)`, balancing precision and recall.
- **mAP@0.5:** mean average precision at an intersection-over-union threshold of 0.50.
- **mAP@0.5:0.95:** mean average precision averaged over IoU thresholds from 0.50 to 0.95 in increments of 0.05.

mAP@0.5:0.95 is the primary detector-selection metric because it rewards accurate localisation across strict overlap thresholds. Conventional classification accuracy is not treated as a principal detector metric because object detection does not have a meaningful fixed true-negative count. Where an accuracy proxy is requested, it is explicitly labelled as the F1 score rather than presented as classification accuracy.

### 6.2 OCR metrics

OCR is evaluated on ground-truth plate crops from the test split. Predictions and labels are normalised to uppercase alphanumeric strings before comparison. Failed reads are treated as empty predictions. Metrics include:

- **Character error rate (CER):** total Levenshtein insertions, deletions, and substitutions divided by the total number of ground-truth characters.
- **Character accuracy:** `max(0, 1 - CER)`.
- **Exact plate accuracy:** the proportion of complete predicted strings exactly matching ground truth.
- **Mean OCR processing time:** average OCR latency per plate crop.

Results are also grouped by capture condition to reveal performance differences for night, rain, blur, dirt, angle, and occlusion.

### 6.3 Face recognition evaluation

Face recognition should be assessed on authorised-owner and unauthorised-person pairs that are disjoint from gallery-enrolment images. The following measures are appropriate:

- verification accuracy at the configured cosine-similarity threshold;
- true acceptance rate for registered owners;
- false acceptance rate for unauthorised persons;
- false rejection rate for registered owners; and
- receiver operating characteristic or threshold-sweep results.

The active backend and threshold must be reported with the results. Smoke-test self-similarity is useful for implementation verification but is not a substitute for evaluation on an independent labelled face set.

### 6.4 End-to-end theft decision metrics

A balanced, labelled theft-scenario manifest is processed through the complete synchronous pipeline. The expected label is withheld from inference and used only after prediction. The database is prepared independently with the correct registered vehicles, owners, stolen flags, and face gallery.

A scenario is predicted as theft when its threat score is at least the configured alert threshold. From true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), the system reports:

- **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`;
- **Precision:** `TP / (TP + FP)`;
- **Recall or sensitivity:** `TP / (TP + FN)`;
- **F1 score:** harmonic mean of precision and recall;
- **False-alert rate:** `FP / (FP + TN)`;
- **Mean processing latency** per scenario; and
- **95th-percentile latency**.

Per-condition results and sample-level predictions are retained. False positives and false negatives are manually traced through plate detection, OCR, face recognition, spatial association, database matching, and threat triggers to identify the actual failure source.

### 6.5 Runtime and system metrics

Operational performance is measured after model warm-up using repeated inference runs. Reported measures include mean, median, minimum, maximum, standard deviation, and 95th-percentile latency, plus throughput calculated as `1000 / mean_latency_ms`. Live pipeline FPS is independently measured over a rolling frame window and stored with detections.

The measured baseline currently available for the COCO YOLOv8n detector on the project CPU is approximately 320 ms per frame, or 3.1 FPS, at 640-pixel input. This is a hardware baseline, not an end-to-end accuracy result. OCR, face recognition, database operations, and alerting add further latency. Consequently, claims of 25-30 FPS require measurement on GPU-accelerated deployment hardware.

### 6.6 Validation controls

To strengthen internal and external validity:

1. all compared models use the same frozen splits and random seed;
2. related images remain in a single group and exact duplicate leakage is rejected;
3. the validation set is used for selection and threshold tuning, while the test set is used once for final reporting;
4. model, dataset, device, input size, threshold, and backend versions are recorded;
5. component metrics are reported separately from end-to-end metrics;
6. results are stratified by environmental condition;
7. class counts accompany percentages, particularly for theft scenarios;
8. false-negative cases receive priority because missed thefts carry higher operational cost;
9. false-alert rate is reported because excessive alarms reduce operator trust; and
10. no unmeasured accuracy or real-time performance value is presented as an achieved result.

## 7. Reproducibility and Reporting

The project provides scripts for dataset preparation and auditing, controlled model training, detector comparison, OCR evaluation, runtime benchmarking, and end-to-end theft evaluation. Configuration is externalised through environment variables, and generated outputs include CSV, JSON, text summaries, plots, checkpoints, and sample-level predictions.

At the current stage, the complete software pipeline and evaluation procedures are implemented. The Roboflow dataset is configured for a fixed split, but final plate-detection, OCR, face-recognition, and theft-classification values must be populated only after the corresponding controlled experiments have completed. This distinction ensures that the methodology describes both the implemented system and a defensible procedure for obtaining results without fabricating performance claims.
