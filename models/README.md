"""
Model weights directory.

Nothing here is committed to git (see `.gitignore`) - weights are large binary
files that do not belong in a source repository.

| File                | Purpose                          | How to obtain |
|---------------------|----------------------------------|---------------|
| `yolov8n.pt`        | vehicle + person detection       | auto-downloaded by Ultralytics on first run, or `yolo download model=yolov8n.pt` |
| `license_plate.pt`  | license-plate localisation       | train it yourself: `python training/train_plate_detector.py` (see `documentation/TRAINING.md`) |

If `license_plate.pt` is absent the plate module falls back to classical
OpenCV localisation (edge density + aspect-ratio filtering).  Accuracy is
noticeably lower, but the system still runs end to end - which is what makes
the project demonstrable before the plate model has finished training.

Face-recognition weights are handled by DeepFace itself and cached under
`~/.deepface/weights/` on first use.  That download is ~90 MB, so run the
system once with an internet connection before an offline demonstration.
