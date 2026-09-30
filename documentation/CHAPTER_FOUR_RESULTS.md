# CHAPTER FOUR

# SYSTEM IMPLEMENTATION AND RESULTS

## 4.1 Introduction

This chapter presents the implementation and experimental results obtained from the real-time vehicle theft detection framework. The completed system integrates a web-based monitoring interface, a FastAPI backend, vehicle and person detection, license plate localisation, optical character recognition (OCR), face recognition, database matching, rule-based theft-risk assessment, evidence storage, and automated alert support.

At the present stage of the research, the project-specific license plate detector has been trained and validated. The available measured results therefore cover the license plate localisation experiment, its training behaviour, validation performance, generated model checkpoints, and an existing CPU inference benchmark for the pretrained YOLOv8n detector. Formal held-out test evaluation of the trained plate detector, OCR accuracy evaluation, independent face-recognition evaluation, alert latency evaluation, and end-to-end theft-decision evaluation remain separate experiments. No unmeasured value is presented in this chapter as an achieved result.

## 4.2 System Implementation Overview

The framework was implemented as a modular client-server application. The frontend was developed with React and provides authenticated pages for live monitoring, registered vehicles, owners, detections, alerts, dashboard statistics, and model-training monitoring. The backend was developed with FastAPI and provides authenticated API routes for database operations, image analysis, live video processing, training monitoring, and administrative functions. SQLite is used to persist user accounts, registered vehicles, face records, detections, alerts, and associated evidence.

The computer-vision pipeline performs the following operations:

1. Detection of vehicles and nearby persons using a pretrained YOLOv8n detector.
2. Association of persons with individual vehicles using spatial proximity and overlap measures.
3. Localisation of a license plate within each detected vehicle region.
4. Recognition and normalisation of plate characters through OCR.
5. Comparison of the recognised plate with registered vehicle records.
6. Detection and matching of faces against the enrolled owner gallery.
7. Calculation of an explainable theft-risk score from plate, face, proximity, loitering, crowd, stolen-status, and time-of-day evidence.
8. Storage of the detection outcome and dispatch of an alert when the configured risk threshold is reached.

A dedicated model-training monitor was also integrated into the administrative frontend. It reports the active run, completed epochs, elapsed time, training losses, validation losses, precision, recall, mean Average Precision, learning rates, and checkpoint availability. This enabled the five-epoch experiment to be monitored without interrupting the training process.

## 4.3 Experimental Configuration

### 4.3.1 Dataset Configuration

The license plate detector was trained using a deterministic subset of 2,000 images selected from the available Roboflow License Plate v3 training data. The provider's original validation and test directories were retained. The task consisted of a single object class named `license_plate`.

The experiment used the following data arrangement:

| Dataset component | Configuration |
|---|---:|
| Training images | 2,000 |
| Validation images | 1,132 |
| Validation plate instances | 1,190 |
| Number of classes | 1 |
| Class name | License plate |
| Test split | Retained for subsequent final evaluation |

The validation set was used to measure performance after each epoch. The test split was not used to select the checkpoint and should be used only for the final independent detector evaluation.

### 4.3.2 Training Environment and Hyperparameters

Training was performed on an Intel Core i5-4200U CPU with 8 GB RAM. Transfer learning was applied using pretrained YOLOv8n weights. The practical configuration was reduced to five epochs, a 416-pixel input size, and a batch size of four because the originally proposed 640-pixel workload was not practical on the available CPU.

| Parameter | Actual configuration |
|---|---:|
| Model | YOLOv8n |
| Initial weights | Pretrained YOLOv8n weights |
| Task | Single-class object detection |
| Training images | 2,000 |
| Input size | 416 x 416 pixels |
| Epochs | 5 |
| Batch size | 4 |
| Device | CPU |
| Optimizer | Automatically selected AdamW |
| Initial optimizer learning rate | 0.002 |
| Optimizer momentum | 0.9 |
| Early-stopping patience | 5 epochs |
| Deterministic mode | Enabled |
| Hue augmentation | 0.015 |
| Saturation augmentation | 0.7 |
| Value augmentation | 0.4 |
| Rotation | 10 degrees |
| Translation | 0.1 |
| Scale | 0.5 |
| Shear | 2.0 |
| Perspective | 0.0005 |
| Horizontal flip probability | 0.5 |
| Vertical flip probability | 0.0 |
| Mosaic augmentation | Enabled |

The complete training and validation process required approximately 12.43 hours. The long duration reflects the limitations of CPU-only training on an older dual-core mobile processor rather than a failure of the model or training procedure.

## 4.4 License Plate Detector Training Results

### 4.4.1 Per-Epoch Results

Table 4.3 presents the training losses and validation detection metrics recorded after each epoch.

| Epoch | Box loss | Classification loss | DFL loss | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.844 | 1.304 | 1.227 | 0.518 | 0.485 | 0.503 | 0.202 |
| 2 | 0.701 | 0.851 | 1.113 | 0.587 | 0.672 | 0.589 | 0.215 |
| 3 | 0.667 | 0.801 | 1.088 | 0.511 | 0.648 | 0.492 | 0.177 |
| 4 | 0.603 | 0.728 | 1.065 | 0.475 | 0.637 | 0.485 | 0.182 |
| 5 | 0.530 | 0.637 | 1.026 | 0.686 | 0.734 | 0.706 | 0.375 |

The losses decreased throughout the experiment. Box loss fell from 0.844 in epoch 1 to 0.530 in epoch 5, representing a reduction of approximately 37.1%. Classification loss decreased from 1.304 to 0.637, a reduction of approximately 51.2%, while Distribution Focal Loss decreased from 1.227 to 1.026, a reduction of approximately 16.4%. These downward trends indicate that the model progressively improved its ability to classify the plate class and estimate plate bounding boxes on the training data.

Validation performance fluctuated between epochs 2 and 4. The mAP@0.5 value increased from 0.503 in epoch 1 to 0.589 in epoch 2, then fell to 0.492 and 0.485 in epochs 3 and 4 respectively. In epoch 5, all principal validation metrics improved substantially. This fluctuation demonstrates why model quality should not be inferred from training loss alone and why validation metrics must be monitored after every epoch.

### 4.4.2 Final Validation Performance

Epoch 5 produced the strongest validation result and was selected by the training framework as the best checkpoint. Its results are summarised in Table 4.4.

| Metric | Epoch 5 result | Interpretation |
|---|---:|---|
| Precision | 68.64% | Approximately 69 of every 100 plate detections were correct at the evaluation operating point. |
| Recall | 73.36% | The detector found approximately 73 of every 100 annotated plate instances. |
| mAP@0.5 | 70.58% | The detector achieved good plate-detection performance at an IoU threshold of 0.50. |
| mAP@0.5:0.95 | 37.54% | Performance was lower under stricter localisation thresholds, indicating room for tighter bounding-box localisation. |
| Validation box loss | 1.377 | Final validation bounding-box regression loss. |
| Validation classification loss | 1.439 | Final validation classification loss. |
| Validation DFL loss | 1.539 | Final validation Distribution Focal Loss. |

Compared with epoch 4, epoch 5 increased precision from 47.45% to 68.64%, recall from 63.74% to 73.36%, mAP@0.5 from 48.49% to 70.58%, and mAP@0.5:0.95 from 18.18% to 37.54%. The final improvement means that epoch 5, rather than epoch 2, was the strongest epoch in the completed run.

The difference between mAP@0.5 and mAP@0.5:0.95 is important. The result indicates that the model can usually identify the general plate region, but its performance declines when increasingly precise overlap between predicted and ground-truth boxes is required. This is expected in a short, CPU-constrained five-epoch experiment and may be improved through additional high-quality training data, correction of annotation inconsistencies, more epochs with controlled early stopping, and access to GPU hardware.

### 4.4.3 Model Artefacts

The experiment generated two deployable PyTorch checkpoints:

- `best.pt`, containing the checkpoint selected from the strongest validation result.
- `last.pt`, containing the model state at the final epoch.

In this experiment, the final epoch also produced the best validation result. Nevertheless, `best.pt` remains the appropriate checkpoint for deployment and formal held-out test evaluation because its selection is explicitly based on validation performance.

The experiment also generated epoch-level CSV metrics, the complete training configuration, a machine-readable summary, training and validation plots, precision-recall and F1 curves, confusion matrices, training batches, validation labels, and validation prediction images. These artefacts provide evidence for reproducibility and support the figures and appendices of this report.

## 4.5 Runtime Performance

A separate CPU benchmark previously conducted on the pretrained COCO YOLOv8n detector at a 640-pixel input size produced the results shown in Table 4.5.

| Runtime metric | Result |
|---|---:|
| Mean inference latency | 320.09 ms |
| Median inference latency | 303.55 ms |
| 95th-percentile latency | 387.18 ms |
| Minimum latency | 285.10 ms |
| Maximum latency | 447.17 ms |
| Standard deviation | 39.05 ms |
| Throughput | 3.12 FPS |
| Device | Intel Core i5-4200U CPU |

This benchmark is a hardware baseline for the pretrained YOLOv8n detector and is not a benchmark of the newly trained plate checkpoint or the complete theft-detection pipeline. OCR, face recognition, database access, evidence storage, and alert dispatch introduce additional processing time. Therefore, the current CPU cannot support a claim of 25-30 FPS for the complete system. The implemented newest-frame processing strategy allows the application to remain responsive by dropping stale live-camera frames when inference is slower than the incoming video rate.

## 4.6 License Plate Recognition Accuracy

The implemented OCR module can preprocess plate crops, extract text, normalise predicted characters, and compare the resulting plate number with registered records. However, a formal OCR experiment using independently verified ground-truth plate transcriptions has not yet been completed. Consequently, the previously stated OCR accuracy of 96% cannot presently be reported as an achieved result.

The final OCR evaluation should report character error rate, character accuracy, exact plate accuracy, failed-read count, mean OCR latency, and results grouped by lighting, blur, viewing angle, dirt, and occlusion. OCR should first be tested on ground-truth plate crops to measure character recognition independently, and then on detector-generated crops to measure complete license plate recognition performance.

## 4.7 Human Detection and Owner Verification Results

The framework implements person detection, face detection, face embedding extraction, owner-gallery enrolment, and cosine-similarity matching. The current environment supports the OpenCV YuNet and SFace route for practical face detection and 128-dimensional face embeddings. However, an independent face dataset containing separate enrolment and test images for authorised and unauthorised persons has not yet been evaluated.

The previously stated face-recognition accuracy of 93.5% must therefore be treated as an expected or literature-derived value unless a reproducible experiment is conducted. The final evaluation should report verification accuracy, true acceptance rate, false acceptance rate, false rejection rate, the decision threshold, the active face backend, class counts, and a threshold-sweep or receiver operating characteristic curve.

## 4.8 Alert System and End-to-End Theft Evaluation

The alert workflow is implemented in the software. When the calculated theft-risk score reaches the configured threshold, the system can store the assessment and evidence and dispatch email or SMS notifications, subject to configuration and a cooldown period. The dashboard also displays detection and alert records.

A formal alert-response experiment and a balanced end-to-end theft-scenario evaluation have not yet been completed. It is therefore not yet defensible to report a measured alert response time, theft-detection accuracy, F1 score, or false-alert rate. These measurements require a labelled scenario manifest containing both theft and non-theft cases, independently prepared database records, and recorded processing times from image input to final decision and notification dispatch.

## 4.9 Comparison with Existing Approaches

The research objective proposes comparison with Faster R-CNN, SSD, YOLOv5, and YOLOv7. However, these baseline models were not trained and evaluated on the same frozen dataset split, input size, hardware, and evaluation procedure during the completed experiment. Direct numerical comparison would therefore be methodologically invalid at this stage.

The current result establishes a measured YOLOv8n plate-localisation baseline of 68.64% precision, 73.36% recall, 70.58% mAP@0.5, and 37.54% mAP@0.5:0.95 on the retained validation set after five epochs. Any future comparative experiment should train or evaluate all candidate models on the same data splits and report both detection quality and latency on the same hardware.

## 4.10 Discussion of Findings

The experiment demonstrates that transfer learning with YOLOv8n can produce a functional license plate detector under severe hardware and training-time constraints. The strongest result was obtained at the fifth epoch, where precision, recall, and both mAP measures reached their highest values. The continued reduction in training losses and the final reduction in validation losses indicate that the model benefited from the complete five-epoch schedule.

The precision of 68.64% means that false-positive plate detections remain a practical concern. The recall of 73.36% also means that approximately one quarter of annotated plates may be missed at the evaluated operating point. In the theft-detection pipeline, a missed or incorrectly localised plate can prevent OCR and database matching from succeeding. This justifies the multi-evidence design of the framework, in which plate evidence is combined with face status, person proximity, stolen status, loitering, crowd size, and time of day rather than used as the sole theft indicator.

The mAP@0.5 result of 70.58% is encouraging for a five-epoch CPU experiment, while the mAP@0.5:0.95 result of 37.54% reveals that precise localisation remains the main detector limitation. The short training schedule, reduced image resolution, small training subset, annotation warnings, and limited CPU capacity are plausible contributors. These findings support further training on cleaned annotations and a larger representative dataset using GPU hardware.

The results also demonstrate the importance of reporting measured evidence conservatively. The implemented application is broader than the completed detector experiment, but implementation alone does not establish OCR accuracy, face-recognition accuracy, end-to-end theft accuracy, alert response time, or real-time throughput. Those values should be added only after the corresponding labelled experiments are completed.

## 4.11 Summary

This chapter presented the implementation and current measured results of the vehicle theft detection framework. A YOLOv8n license plate detector was fine-tuned using 2,000 training images for five epochs at a 416-pixel input size and batch size four on an Intel Core i5-4200U CPU with 8 GB RAM. The experiment completed in approximately 12.43 hours.

The strongest validation result was obtained in epoch 5, with precision of 68.64%, recall of 73.36%, mAP@0.5 of 70.58%, and mAP@0.5:0.95 of 37.54%. The training produced both best and final model checkpoints together with complete metric and visual artefacts. A separate pretrained YOLOv8n CPU benchmark achieved 3.12 FPS at 640-pixel input, although this does not represent complete pipeline speed.

The plate-detection results establish a measurable foundation for the system, but formal held-out test evaluation, OCR evaluation, face verification, end-to-end theft classification, alert latency measurement, and controlled comparison with baseline detectors remain necessary before final performance claims can be made.

---

# REQUIRED CORRECTIONS TO THE EXISTING REPORT

## A. Replacement Abstract Result Passage

Replace the unsupported result claims in the abstract with the following passage:

> The license plate localisation component was fine-tuned from pretrained YOLOv8n weights using a deterministic subset of 2,000 training images for five epochs at an input size of 416 pixels and a batch size of four on an Intel Core i5-4200U CPU with 8 GB RAM. The best validation result was obtained at epoch 5, achieving a precision of 68.64%, recall of 73.36%, mAP@0.5 of 70.58%, and mAP@0.5:0.95 of 37.54%. The experiment produced deployable best and final checkpoints together with complete training metrics and visual evaluation artefacts. A separate CPU benchmark of the pretrained YOLOv8n detector achieved 3.12 frames per second at 640-pixel input. Formal OCR, face-recognition, end-to-end theft-decision, and alert-latency evaluations are defined but should be reported only after completion on independently labelled test data.

Remove the claims that OCR achieved 96% and face recognition achieved 93.5% unless their underlying labelled datasets, sample counts, evaluation scripts, confusion statistics, and output reports are available.

## B. Correction to Section 3.4.4

Replace the statement that the completed detector received 640-pixel images with:

> For the completed CPU-constrained experiment, Ultralytics resized detector inputs to 416 pixels. The lower input size was selected to make training practical on the available Intel Core i5-4200U CPU with 8 GB RAM. The configured online augmentations included hue, saturation, and value jitter; rotation; translation; scale; shear; perspective transformation; horizontal flipping; and mosaic augmentation. Validation and test images were not augmented.

## C. Correction to Section 3.5.1

The report currently states that YOLOv8n and YOLO11n were trained as a controlled comparison. Replace this with:

> The completed experiment fine-tuned YOLOv8n as the project-specific license plate localiser. A controlled YOLOv8n-versus-YOLO11n comparison was part of the proposed experimental design but was not completed on the available CPU hardware. Consequently, this report presents only the measured YOLOv8n results and does not claim comparative superiority over YOLO11n or other baseline detectors.

## D. Correction to Section 3.5.2

Replace the proposed 100-epoch, 640-pixel, batch-16 table with the actual experiment:

| Parameter | Completed experiment |
|---|---:|
| Model | YOLOv8n |
| Training subset | 2,000 images |
| Input size | 416 x 416 pixels |
| Epochs | 5 |
| Batch size | 4 |
| Device | CPU |
| Patience | 5 epochs |
| Deterministic mode | Enabled |
| Optimizer | Automatically selected AdamW |
| Training duration | Approximately 12.43 hours |

## E. Correction to Section 3.5.3

Replace the claim that candidate models were compared and only the selected model was evaluated with:

> The YOLOv8n run was monitored using the validation split, and the checkpoint with the strongest validation performance was retained as `best.pt`. The untouched test split remains reserved for one final detector evaluation. Since no second candidate architecture was completed under the same experimental conditions, cross-model selection was not performed in the present experiment.

## F. Recommended Figures and Tables

The following generated artefacts should be inserted into Chapter Four:

1. `results.png` as the training-loss and validation-metric figure.
2. `BoxPR_curve.png` as the precision-recall curve.
3. `BoxF1_curve.png` as the F1-confidence curve.
4. `confusion_matrix.png` as the confusion matrix.
5. `val_batch0_labels.jpg` and `val_batch0_pred.jpg` as a ground-truth versus prediction example.
6. The per-epoch metrics table in Section 4.4.1.
7. The final epoch result table in Section 4.4.2.
8. The CPU benchmark table in Section 4.5.

Suggested captions:

- Figure 4.1: Training losses and validation metrics across five epochs.
- Figure 4.2: Precision-recall curve for the trained license plate detector.
- Figure 4.3: F1 score as a function of confidence threshold.
- Figure 4.4: Confusion matrix for license plate and background predictions.
- Figure 4.5: Comparison of validation ground-truth annotations and model predictions.
