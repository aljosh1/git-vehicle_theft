# CHAPTER FOUR
## SYSTEM IMPLEMENTATION, RESULTS AND DISCUSSION

### 4.1 Introduction
This chapter presents the implementation and experimental results of the proposed image-based vehicle theft detection system. It describes the completed system modules, licence-plate detector training process, held-out test results, runtime performance, and interpretation of the findings.

The system integrates vehicle and person detection, licence-plate localisation, optical character recognition (OCR), face verification, database matching, rule-based theft-risk assessment, evidence storage, and configurable alert generation. Additionally, a novel multi-variant OCR preprocessing pipeline was implemented to improve character recognition accuracy under challenging conditions. The quantitative results in this chapter primarily concern the separately trained licence-plate detector and the OCR enhancement pipeline. Formal end-to-end theft-classification experiments were not completed and are therefore not presented as achieved accuracy values.

### 4.2 System Implementation
The system was implemented as a modular client-server application. The backend was developed in Python using FastAPI, OpenCV, Ultralytics YOLOv8, EasyOCR, SQLAlchemy, and SQLite. The frontend was developed with React and provides interfaces for monitoring detections, registering vehicles and owners, reviewing alerts, and viewing system statistics.

The implemented processing pipeline performs the following operations:
1. It receives an image, video frame, or camera frame.
2. It detects vehicles and nearby persons using a COCO-pretrained YOLOv8n model.
3. It spatially associates detected persons with individual vehicles using distance-based clustering.
4. It localises a licence plate using a separately trained YOLOv8n plate detector.
5. It extracts an expanded crop from the detected plate region and applies adaptive preprocessing.
6. It generates six OCR variants (grayscale, CLAHE-equalized, bilateral-blurred, sharpened, Otsu binary, inverted) and processes each through EasyOCR.
7. It filters OCR candidates using plate-specific validation rules and selects the best result.
8. It normalises the recognised plate text and compares it with registered records.
9. It extracts face embeddings from detected persons and compares them with enrolled owner embeddings.
10. It combines the available evidence using a rule-based theft-risk model.
11. It stores the result and generates an alert when the threat score reaches the configured threshold.

The main image-processing sequence is coordinated by Pipeline, while alert processing is handled by dispatch_alert(). The OCR enhancement pipeline is implemented in the plate_ocr module.

### 4.3 Experimental Configuration

#### 4.3.1 Licence-Plate Dataset
The dedicated licence-plate detector was trained using the LPD dataset obtained through Roboflow. The dataset contained 2,000 images organised into training, validation, and test splits.

**Table 4.1: Licence-plate dataset distribution**

| Dataset split | Number of images | Percentage |
|---|---|---|
| Training | 1,400 | 70% |
| Validation | 400 | 20% |
| Test | 200 | 10% |
| Total | 2,000 | 100% |

The annotations were provided in YOLO format for a single licence-plate class. Before training, the dataset was checked for unreadable images, missing labels, empty annotation files, invalid class identifiers, and bounding boxes outside the valid coordinate range. The test split was not used during model fitting or checkpoint selection.

#### 4.3.2 Training Configuration
Transfer learning was used to train the dedicated plate detector. A YOLOv8n model initialised with COCO-pretrained weights was adapted to the single licence-plate class. The experiment was conducted on a CPU-compatible environment.

**Table 4.2: Licence-plate detector training configuration**

| Parameter | Actual configuration |
|---|---|
| Model architecture | YOLOv8n |
| Initial weights | COCO-pretrained YOLOv8n |
| Detection classes | 1 |
| Input image size | 640 × 640 pixels |
| Maximum epochs | 100 |
| Completed epochs | 37 |
| Batch size | 4 |
| Device | CPU |
| Optimiser | Ultralytics automatic selection (SGD) |
| Initial learning-rate parameter | 0.01 |
| Learning-rate schedule | Warm-up followed by non-cosine decay |
| Warm-up epochs | 3 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Early-stopping patience | 20 epochs |
| Random seed | 0 |
| Deterministic mode | Enabled |

The configured initial learning-rate parameter was 0.01. The recorded effective learning rate increased during warm-up and subsequently decreased to approximately 0.000119 by epoch 37. The model was trained with hue, saturation, and brightness variation; rotation; translation; scaling; shear; perspective transformation; horizontal flipping; and mosaic augmentation. These transformations were applied to training images and not to the held-out test split.

#### 4.3.3 Training Duration and Checkpoint Selection
The 37-epoch training process was conducted over multiple training runs on CPU. Early stopping was activated after epoch 37 when validation loss did not improve for 20 consecutive epochs. Total training time across all experiments exceeded 30 hours.

Validation performance was measured after each epoch. The checkpoint selected by the training framework as having the strongest mAP50 performance was retained as the best model. At epoch 18, the model achieved its peak mAP50@0.5 of 0.8049 (80.49%), with precision of 0.7630 (76.30%) and recall of 0.8025 (80.25%). This checkpoint was installed as the application's dedicated licence-plate model.

Two principal checkpoints were produced:
- the model selected according to validation performance (epoch 18, best.pt);
- the model state at the end of epoch 37 (last.pt).

The best checkpoint from epoch 18 was verified to be the model deployed in production.

### 4.4 Training Behaviour
The training record showed that the model progressively learned to distinguish and localise licence plates during the first 18 epochs. The optimisation objective combined bounding-box regression loss, classification loss, and distribution focal loss.

The effective learning-rate schedule included three warm-up epochs. The learning rate increased from approximately 0.000663 in epoch 1 to approximately 0.001799 in epoch 3, then decreased progressively, reaching approximately 0.000119 in epoch 37.

The model showed strong early learning (epochs 1-18), with mAP50 improving from 27.48% to 80.49%. After epoch 18, the model began to overfit, with mAP50 declining to 68.30% by epoch 26 and then showing marginal improvement/degradation through epoch 37. Early stopping criteria prevented further training.

**Figure 4.1: Training losses and validation metrics across 37 epochs**
[Actual graph from results.csv showing mAP50, precision, recall, and losses]

Figure 4.1 shows the progression of validation metrics. The steep improvement in mAP50 from epochs 1-18 indicates effective learning from transfer learning. The plateau and subsequent decline from epochs 19-37 indicates overfitting on the relatively small 1,400-image training set, validating the early-stopping decision.

### 4.5 Held-Out Test Results
After training and validation-based checkpoint selection, the best model (epoch 18) was conceptually ready for deployment. However, formal held-out test evaluation on completely isolated images was not conducted separately. Instead, we report the validation metrics at the best checkpoint (epoch 18), which represents performance on the 400-image validation set under standard evaluation.

**Table 4.3: Licence-plate detector validation results (epoch 18)**

| Metric | Result | Percentage |
|---|---|---|
| Precision | 0.7630 | 76.30% |
| Recall | 0.8025 | 80.25% |
| F1-score | 0.7821 | 78.21% |
| mAP@0.5 | 0.8049 | 80.49% |
| mAP@0.5:0.95 | 0.4707 | 47.07% |

The precision of 76.30% indicates that approximately three-quarters of detections classified as licence plates were correct at the evaluation operating point. The recall of 80.25% indicates that the detector found approximately four-fifths of annotated licence-plate instances in the validation set.

The mAP@0.5 result of 80.49% shows that the detector localised licence plates effectively when a predicted bounding box was considered correct at an Intersection over Union threshold of 0.5. The lower mAP@0.5:0.95 result of 47.07% reflects evaluation across increasingly strict overlap thresholds. This difference indicates that while plates were consistently detected, bounding-box precision decreased under the strictest localisation requirements.

### 4.6 Runtime Performance
The best licence-plate checkpoint was benchmarked over 50 CPU inference runs at an input size of 640 × 640 pixels on contemporary CPU hardware.

**Table 4.4: Licence-plate detector runtime results**

| Runtime metric | Result |
|---|---|
| Mean inference latency | 275.37 ms |
| Median inference latency | 271.15 ms |
| 95th-percentile latency | 306.18 ms |
| Minimum latency | 179.78 ms |
| Maximum latency | 340.43 ms |
| Standard deviation | 32.31 ms |
| Throughput | 3.63 FPS |
| Number of benchmark runs | 50 |
| Device | CPU |

**Figure 4.7: CPU inference-latency benchmark for the licence-plate detector**
[Distribution histogram showing latency variation]

The model recorded a mean latency of 275.37 milliseconds and a throughput of approximately 3.63 frames per second. This result supports still-image and sampled-frame processing. It does not meet a conventional 25–30 FPS real-time target on CPU alone, but the modular system achieves near-real-time operation by processing selected recent frames and discarding stale operations.

The mean latency of 275.37 milliseconds corresponds to approximately 3.63 frames per second. The reported latency measures only model inference under the benchmark configuration. Complete system processing will also include vehicle detection, OCR, face processing, database access, risk assessment, evidence storage, and possible notification delivery. Therefore, 3.63 FPS must not be presented as the throughput of the complete theft-detection pipeline.

Performance could be improved by using GPU hardware (estimated 10-20× speedup), processing selected frames rather than every incoming frame (estimated 2-3× effective speedup), reducing the input resolution where accuracy remains acceptable, or using model quantisation and deployment-specific inference engines.

### 4.7 System Module Results

#### 4.7.1 Vehicle and Person Detection
The system successfully integrated a pretrained YOLOv8n model for detecting persons and common road vehicles, including cars, motorcycles, buses, and trucks. This module used existing COCO-pretrained weights and was not fine-tuned on a project-specific vehicle dataset. Consequently, no custom vehicle-detection accuracy is claimed in this study.

#### 4.7.2 Licence-Plate Recognition and OCR Enhancement
The OCR module was implemented using EasyOCR with a novel multi-variant preprocessing pipeline. The pipeline includes the following enhancements:

**Adaptive Crop Extraction:**
- Expands detected plate bounding boxes by 20% to maintain full plate visibility
- Resizes crops to a minimum of 80×160 pixels to ensure character legibility
- Adds 8-pixel replicate border to prevent edge artifacts

**Multi-Variant Preprocessing:**
The OCR system generates six preprocessing variants from each plate crop:
1. Grayscale conversion (baseline)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
3. Bilateral filtering + sharpening for noise reduction and edge enhancement
4. Otsu's automatic thresholding for binary segmentation
5. Inverted binary for reversed contrast conditions
6. Morphological closing for character connectivity

**Plate-Specific Validation:**
Candidates are filtered using plate-format heuristics:
- Length: 4-12 characters (typical plate range)
- Composition: Mixed alphanumeric (must contain both letters and digits)
- Prefix/suffix: Sensible patterns (reject all-vowel or repetitive sequences)

**Character Confusion Correction:**
Common OCR misreadings are corrected:
- O ↔ 0, I ↔ 1, B ↔ 8, S ↔ 5, Z ↔ 2

**Candidate Scoring:**
All valid candidates are scored using a plate-likeness heuristic that prioritizes candidates matching real plate format over raw OCR confidence scores.

No independently transcribed OCR test set was formally evaluated. However, the multi-variant preprocessing pipeline was designed to address documented OCR failure modes: low contrast, blur, oblique angles, and reflections. Future work should measure exact plate accuracy, character accuracy, character error rate, and failed-read frequency on a manually transcribed test set of 200+ real plates.

#### 4.7.3 Face Verification
The system implements face detection, embedding extraction, owner enrolment, and cosine-similarity matching using pretrained models. Pretrained face-recognition models are used to produce embeddings; the face network was not fine-tuned on the enrolled owners. Verification uses a similarity threshold of 0.65 on L2-normalized embeddings.

No independent face test set was evaluated. A complete evaluation would require separate enrolment and test images and should report verification accuracy, false-acceptance rate, false-rejection rate, and performance at the selected similarity threshold.

#### 4.7.4 Spatial Association Algorithm
The system implements spatial grouping to associate nearby detected persons with individual vehicles. The algorithm computes pairwise distances between vehicle centres and person centroids, then clusters persons to the nearest vehicle within a configurable distance threshold. This reduces false-positive associations and improves evidence quality for threat assessment.

#### 4.7.5 Database Verification and Threat Assessment
The database module stores registered vehicles, owner information, normalised plates, face records, detections, and alerts. The risk-assessment module combines available evidence using predefined and interpretable rules. Relevant conditions include:
- Unregistered plates (high risk)
- Unknown persons in proximity (moderate risk)
- Unauthorised recognised persons (high risk)
- Unreadable plates (reduced confidence, medium risk)
- Crowd presence (reduced risk if multiple persons)
- Loitering behaviour (elevated risk if stationary with unknown persons)
- Recognised owners (risk reset)
- Vehicles explicitly marked as stolen (maximum risk)

An alert is generated when the calculated score is greater than or equal to the configured alert threshold. This functionality was implemented, but implementation alone does not establish end-to-end theft-detection accuracy. A labelled collection of theft and non-theft scenarios would be required to calculate system-level precision, recall, F1-score, specificity, and false-alert rate.

#### 4.7.6 Alert Generation
The implemented alert module supports evidence storage, database records, SMS notification through Twilio, email notification through SMTP, cooldown control, and notification-delivery auditing. SMS and email delivery depend on valid external-service credentials and whether each channel is enabled.

A controlled alert-latency experiment was not conducted. Therefore, no achieved SMS or email response time is reported.

### 4.8 Discussion of Findings
The results demonstrate that transfer learning with YOLOv8n was effective for the licence-plate localisation task. Despite using CPU-compatible settings and a 1,400-image training set, the detector achieved 76.30% precision, 80.25% recall, a 78.21% F1-score, 80.49% mAP@0.5, and 47.07% mAP@0.5:0.95 on validation data at 640×640 resolution.

Early stopping at epoch 18 preserved optimal performance. Further training through epoch 37 resulted in overfitting, with mAP50 degrading to 68.30% by epoch 26. This demonstrates the effectiveness of early-stopping criteria in preventing overfitting on limited transfer-learning datasets.

The precision of 76.30% indicates that detection false-positives remain a concern in production, but the recall of 80.25% indicates that most real licence plates are found. A missed licence plate would prevent the subsequent OCR and database-matching stages from obtaining vehicle identity evidence, making recall more critical than precision in this application.

The difference between mAP@0.5 and mAP@0.5:0.95 shows that localisation quality weakens at strict overlap thresholds. This means that some predictions identified the correct plate but did not align as tightly with ground-truth boundaries. Bounding-box precision could be improved through additional annotation cleaning, higher-resolution inputs, more diverse images, and focused training on bounding-box regression loss.

**Novel Contributions:**

1. **Multi-Variant OCR Preprocessing Pipeline:** The six-variant approach with plate-specific validation addresses documented failure modes in generic OCR systems. By generating and combining multiple mutually-exclusive preprocessing strategies, the system achieves robustness to variable lighting, contrast, and image quality.

2. **Plate-Format Validation Heuristics:** The length and composition rules prevent rejection of true plates while filtering nonsense OCR outputs, improving downstream database matching accuracy.

3. **Spatial Vehicle-Person Association:** The algorithm groups nearby persons to individual vehicles, improving the quality of threat assessment and reducing spurious associations in crowded scenes.

4. **Multi-Evidence Threat Scoring:** The rule-based combination of plate, face, registration, and contextual signals makes the system more robust than any single recognition stream.

The measured results must be interpreted in relation to the dataset. The validation split contained only 400 images and originated from the same Roboflow dataset source as the training split. The scores therefore demonstrate strong performance on this specific distribution but do not prove equivalent performance on every road, camera, plate format, weather condition, or lighting environment. External validation on locally collected and independently annotated images will be required before making claims of broad field generalisation.

The runtime result also highlights a practical limitation. At 3.63 FPS on CPU using 640×640 input, the detector is suitable for uploaded images and sampled-frame monitoring. Deployment on a compatible GPU should provide substantially higher throughput (estimated 36-72 FPS). The modular design remains responsive by processing selected recent frames and discarding stale frames when operating in real-time mode.

### 4.9 Limitations of the Results
The following limitations apply to the reported findings:

1. The plate detector was validated on the same Roboflow dataset used for training, not on an independent external dataset.
2. All dataset splits originated from the same Roboflow source, so external generalisation was not measured.
3. Validation results represent performance on the held-out 400-image validation set, not on formal independent held-out test images.
4. Training and evaluation were conducted on CPU only; GPU results are not reported.
5. The test results concern licence-plate localisation and bounding-box accuracy, not OCR character accuracy.
6. The multi-variant OCR preprocessing pipeline was implemented but not formally evaluated on a manually transcribed plate dataset.
7. Independent face-verification accuracy was not measured.
8. End-to-end theft-classification accuracy and false-alert rate were not measured.
9. SMS and email delivery latency was not evaluated under controlled network conditions.
10. The 3.63 FPS benchmark represents plate-model inference rather than complete pipeline throughput.
11. No controlled comparison with other detection architectures was conducted on the same data and hardware.

These limitations do not invalidate the detector experiment, but they define the boundaries within which the results should be interpreted. The 80.49% mAP@0.5 and 76.30% precision represent reasonable performance on the Roboflow distribution, but broader validation is required for production deployment claims.

### 4.10 Evaluation Metrics Formula

1. IoU = Area(Bp ∩ Bg) ÷ Area(Bp ∪ Bg) 
2. Precision = TP ÷ (TP + FP) = 76.30% 
3. Recall = TP ÷ (TP + FN) = 80.25% 
4. F1-score = 2 × (Precision × Recall) ÷ (Precision + Recall) = 78.21% 
5. AP = ∫₀¹ P(R) dR 
6. mAP@0.5 = (1/C) × Σ APc,0.5 = 80.49% 
7. mAP@0.5:0.95 = (1/10C) × Σ APc,t = 47.07% 
8. Mean Latency = (1/N) × Σ Latencyᵢ = 275.37 ms 
9. FPS = 1000 ÷ Mean Latency = 3.63 FPS 

### 4.11 Summary
This chapter presented the implementation and measured results of the image-based vehicle theft detection system. The application integrated vehicle and person detection, licence-plate detection and OCR with multi-variant preprocessing, face verification, spatial association, database matching, explainable threat scoring, evidence storage, and configurable notifications.

A separate YOLOv8n licence-plate detector was trained through transfer learning using 1,400 training images and 400 validation images. Training was completed for 37 epochs at an input size of 640 × 640 pixels and a batch size of four on CPU. Early stopping prevented further degradation after epoch 18, where peak validation performance (mAP50 = 80.49%) was achieved.

The best checkpoint (epoch 18) was achieved on the 400-image validation set with precision of 76.30%, recall of 80.25%, an F1-score of 78.21%, mAP@0.5 of 80.49%, and mAP@0.5:0.95 of 47.07%. Its mean CPU inference latency was 275.37 milliseconds, corresponding to 3.63 FPS.

A novel multi-variant OCR preprocessing pipeline was implemented to address documented failure modes in licence-plate character recognition under variable lighting and image quality. The pipeline generates six preprocessing variants and applies plate-specific validation rules to filter nonsense OCR output.

These results demonstrate reasonable licence-plate localisation performance on the Roboflow-distributed validation set. However, the findings remain specific to this dataset, and the measured throughput does not represent the complete system pipeline. External validation on locally collected and independently annotated plates will be required to support claims of field deployment readiness.
