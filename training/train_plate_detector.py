"""
Train a YOLOv8 license plate detector.

    python training/train_plate_detector.py --data dataset/plate_data.yaml --epochs 100

Fine-tunes COCO-pretrained YOLOv8 on a single `license_plate` class.  Transfer
learning from COCO rather than training from scratch is what makes this feasible
on a few hundred images: the backbone already knows edges, corners and text-like
texture, so only the detection head needs to learn what a plate looks like.

Outputs (under `training/outputs/<name>/`):

    weights/best.pt        copy this to models/license_plate.pt
    weights/last.pt
    results.csv            per-epoch loss and metric history
    results.png            training curves
    confusion_matrix.png
    PR_curve.png, F1_curve.png
    training_log.txt       this script's own log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR, settings  # noqa: E402
from backend.utils.logger import get_logger, setup_logging  # noqa: E402
from training.install_plate_model import install_plate_model  # noqa: E402

log = get_logger(__name__)

OUTPUT_ROOT = BASE_DIR / "training" / "outputs"


def check_dataset(data_yaml: Path) -> None:
    """Fail early and clearly if the dataset is missing or empty."""
    if not data_yaml.exists():
        log.error(
            "dataset descriptor not found: %s\n"
            "Run: python training/prepare_dataset.py --source <your images>",
            data_yaml,
        )
        raise SystemExit(1)

    try:
        import yaml

        spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except ImportError:
        log.warning("PyYAML not installed - skipping the dataset sanity check")
        return
    except Exception:
        log.exception("could not parse %s", data_yaml)
        raise SystemExit(1) from None

    root_value = Path(spec.get("path", data_yaml.parent))
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value).resolve()
    train_value = Path(spec.get("train", "images/train"))
    train_path = train_value if train_value.is_absolute() else root / train_value
    if not train_path.exists():
        log.error("training images path does not exist: %s", train_path)
        raise SystemExit(1)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if train_path.is_file():
        image_paths = [
            Path(line.strip())
            for line in train_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        missing = [path for path in image_paths if not path.exists()]
        if missing:
            log.error("%d listed training images do not exist; first: %s", len(missing), missing[0])
            raise SystemExit(1)
        count = sum(1 for path in image_paths if path.suffix.lower() in image_suffixes)
        location = f"image list {train_path}"
    else:
        count = sum(1 for p in train_path.iterdir() if p.suffix.lower() in image_suffixes)
        location = str(train_path)

    if count == 0:
        log.error("no training images in %s", train_path)
        raise SystemExit(1)

    log.info("dataset OK: %d training images in %s", count, location)
    if count < 100:
        log.warning(
            "only %d training images. Expect weak accuracy below ~300 images; "
            "consider augmentation (see documentation/TRAINING.md).",
            count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 plate detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "dataset" / "plate_data.yaml",
        help="Ultralytics dataset yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=settings.DEVICE, help="cpu / cuda / 0")
    parser.add_argument("--name", default="plate_detector")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="early-stopping patience in epochs (0 disables)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="copy best.pt to models/license_plate.pt when training finishes",
    )
    args = parser.parse_args()

    setup_logging("INFO", settings.logs_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is not installed; run: pip install ultralytics")
        raise SystemExit(1) from None

    check_dataset(args.data)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    log.info(
        "training %s for %d epochs at %dpx on %s (batch=%d)",
        args.model, args.epochs, args.imgsz, args.device, args.batch,
    )
    if args.device == "cpu":
        log.warning(
            "training on CPU. 100 epochs on ~500 images takes several hours; "
            "a CUDA GPU reduces that to roughly 20 minutes."
        )

    model = YOLO(args.model)
    try:
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(OUTPUT_ROOT),
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            # --- augmentation ------------------------------------------------
            # Plates are rigid, near-planar and always upright in traffic
            # footage, so the geometric augmentation is kept mild: heavy
            # rotation or vertical flipping would synthesise views that never
            # occur and waste capacity. Colour jitter is generous instead,
            # because lighting genuinely varies from noon to sodium-lamp night.
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0005,
            flipud=0.0,          # plates are never upside down
            fliplr=0.5,          # a mirrored plate is still a plate shape
            mosaic=1.0,
            plots=True,
        )
    except KeyboardInterrupt:
        log.warning("training interrupted - partial weights are in %s", OUTPUT_ROOT)
        raise SystemExit(130) from None
    except Exception:
        log.exception("training failed")
        raise SystemExit(1) from None

    duration = datetime.now() - started
    run_dir = OUTPUT_ROOT / args.name
    best = run_dir / "weights" / "best.pt"

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dataset": str(args.data),
        "duration_seconds": round(duration.total_seconds(), 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "best_weights": str(best) if best.exists() else None,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n  Training complete in {duration}")
    print(f"     run directory : {run_dir}")
    print(f"     best weights  : {best if best.exists() else 'not produced'}")

    if args.install and best.exists():
        target = settings.models_dir / "license_plate.pt"
        receipt = install_plate_model(
            best,
            installed_weights=target,
            trigger="train_plate_detector --install",
            metadata={
                "run_name": args.name,
                "dataset": str(args.data),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "best_weights": str(best),
            },
        )
        print(f"     installed to  : {target}")
        print(f"     receipt       : {settings.models_dir / 'license_plate.install_receipt.json'}")
        log.info("installed trained weights to %s", target)
    elif best.exists():
        print(
            f"\n  To use it:  copy \"{best}\" \"{settings.models_dir / 'license_plate.pt'}\""
        )

    print(f"\n  Evaluate: python training/evaluate.py --weights {best}\n")


if __name__ == "__main__":
    main()
