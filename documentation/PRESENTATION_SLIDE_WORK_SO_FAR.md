# Summary Table: Work Completed So Far

| Area | Summary | Key Metrics / Evidence | Status |
|---|---|---|---|
| System implementation | End-to-end vehicle theft detection pipeline implemented, covering vehicle/person detection, plate localization + OCR, face verification, threat scoring, alert dispatch, and evidence logging. | FastAPI backend, React dashboard, SQLite persistence, staged inference controls. | Complete |
| Plate detection performance | Dedicated YOLOv8n plate detector trained and evaluated on the validation split. | Precision: 76.30% · Recall: 80.25% · F1-score: 78.21% · mAP@0.5: 80.49% · mAP@0.5:0.95: 47.07% | Complete |
| Runtime performance | CPU benchmark performed for the plate detector at 640 × 640 input size. | Mean latency: 275.37 ms · Throughput: 3.63 FPS | Complete |
| OCR contribution | Multi-variant OCR preprocessing pipeline implemented to improve plate text recognition under difficult image conditions. | 6 variants: CLAHE, sharpening, thresholding, inversion, morphology, and baseline preprocessing. | Implemented |
| Deployment controls | Real-time stability features added to improve practical use on CPU. | Frame striding, track-level caching, cooldown-based alerts, stale-frame dropping. | Implemented |
| Evaluation boundaries | Current results are limited to plate detection validation and CPU benchmarking. | End-to-end theft accuracy, OCR character accuracy, and independent face-verification metrics still pending. | In progress |
| Next milestones | Remaining work focuses on broader validation and deployment-ready reporting. | Held-out/external validation, theft-scenario evaluation, GPU benchmark, final report figures. | Pending |

*Use this table in the report as the single consolidated summary of completed work and measured results. It is intentionally limited to verified outcomes only.*