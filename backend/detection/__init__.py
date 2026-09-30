"""
Detection package - YOLOv8 wrappers.  *Interfaces only at this stage.*

Planned modules (Step 3):

    base.py             `Detection` dataclass + shared NMS/geometry helpers
    vehicle_detector.py YOLOv8 filtered to COCO classes 2/3/5/7 and person
    plate_detector.py   fine-tuned YOLOv8 plate localiser, with a classical
                        OpenCV fallback when no plate weights are present

All detectors will expose the same call signature:

    detector.detect(frame: np.ndarray) -> list[Detection]

so the pipeline can swap YOLOv8 for YOLO11 (or a TensorRT export) without any
change above this layer.
"""
