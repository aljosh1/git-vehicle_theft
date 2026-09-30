"""Generate confusion matrix images for a given YOLO weights file.

This script runs `YOLO.val(..., plots=True)` into a temporary run directory
and copies any `confusion_matrix*.png` outputs into the project's
`training/outputs/final_evaluation` directory (creating it if missing).

Usage:
    python -m training.generate_confusion_matrix --weights <path-to-weights>

If `--weights` is omitted the script will default to the likely plate
detector best weights used during training.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion matrix images")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Dataset YAML for validation",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size to validate at",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold to use when validating/plotting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run validation on (e.g. cpu or 0 for GPU)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    outputs = base / "training" / "outputs"
    final_dir = outputs / "final_evaluation"
    final_dir.mkdir(parents=True, exist_ok=True)

    # sensible default weights if none provided (common training output)
    default_weights = (
        outputs / "plate_detector_640_100e" / "weights" / "best.pt"
    )
    weights = args.weights or default_weights

    if not weights.exists():
        print(f"Weights not found: {weights}")
        sys.exit(1)

    data_yaml = args.data or (base / "dataset" / "plate_data.yaml")
    if not data_yaml.exists():
        print(f"Dataset YAML not found: {data_yaml}")
        sys.exit(1)

    # import ultralytics lazily so we can give a friendly error
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime environment
        print("ultralytics is required to run this script. Install with: pip install ultralytics")
        raise SystemExit(1) from exc

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("generate_confusion")

    tmp_name = "confusion_epoch18_tmp"
    tmp_dir = outputs / tmp_name
    if tmp_dir.exists():
        # avoid clobbering an old run
        shutil.rmtree(tmp_dir)

    log.info("Loading model from %s", weights)
    model = YOLO(str(weights))

    log.info(
        "Running validation (plots enabled) into %s/%s with conf=%s device=%s",
        outputs,
        tmp_name,
        args.conf,
        args.device,
    )
    # This writes plots to outputs/tmp_name
    model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        split="val",
        device=args.device,
        conf=float(args.conf),
        verbose=False,
        plots=True,
        project=str(outputs),
        name=tmp_name,
    )

    # collect produced confusion matrices
    found = list(tmp_dir.rglob("confusion_matrix*.png"))
    if not found:
        log.warning("No confusion matrix images found in %s", tmp_dir)
    else:
        for fp in found:
            dest = final_dir / fp.name
            shutil.copy2(fp, dest)
            log.info("Copied %s -> %s", fp, dest)

    # copy a few example validation images if present
    for name in ("val_batch0_pred.jpg", "val_batch0_labels.jpg"):
        src = tmp_dir / name
        if src.exists():
            shutil.copy2(src, final_dir / src.name)

    # cleanup tmp dir
    try:
        shutil.rmtree(tmp_dir)
        log.info("Removed temporary run directory %s", tmp_dir)
    except Exception:
        log.warning("Could not remove temporary directory %s", tmp_dir)

    print(f"Confusion matrix generation complete. Output: {final_dir}")


if __name__ == "__main__":
    main()
