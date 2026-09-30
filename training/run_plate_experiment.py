"""Train, compare, and install two plate detector configurations reproducibly."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR, settings  # noqa: E402
from training.compare_detectors import compare_models  # noqa: E402
from training.evaluate import evaluate_accuracy  # noqa: E402
from training.install_plate_model import install_plate_model  # noqa: E402
from training.train_plate_detector import check_dataset  # noqa: E402

OUTPUT_ROOT = BASE_DIR / "training" / "outputs"


def train_model(
    model_name: str,
    data: Path,
    *,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
) -> Path:
    """Train one model with the experiment's fixed settings."""
    from ultralytics import YOLO

    run_name = f"plate_{Path(model_name).stem}"
    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=True,
        seed=seed,
        deterministic=True,
        patience=25,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        plots=True,
    )
    best = OUTPUT_ROOT / run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training did not produce {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n and YOLO11n plate detectors on one fixed split"
    )
    parser.add_argument(
        "--data", type=Path, default=BASE_DIR / "dataset" / "plate_data.yaml"
    )
    parser.add_argument(
        "--models", default="yolov8n.pt,yolo11n.pt", help="comma-separated bases"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    check_dataset(args.data)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if len(models) < 2:
        parser.error("--models must contain at least two configurations")

    trained: list[tuple[str, Path]] = []
    for model_name in models:
        best = train_model(
            model_name,
            args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
        )
        trained.append((Path(model_name).stem, best))

    comparison_dir = OUTPUT_ROOT / "comparisons"
    comparison = compare_models(
        trained,
        args.data,
        task="plate",
        imgsz=args.imgsz,
        device=args.device,
        runs=args.runs,
        output_dir=comparison_dir,
        split="val",
    )
    best_name = comparison["best_model"]
    best_weights = next(path for name, path in trained if name == best_name)
    final_test_metrics = evaluate_accuracy(
        best_weights, args.data, args.imgsz, args.device, split="test"
    )
    installed = settings.models_dir / "license_plate.pt"

    receipt = install_plate_model(
        best_weights,
        installed_weights=installed,
        trigger="run_plate_experiment",
        metadata={
            "selected_model": best_name,
            "dataset": str(args.data.resolve()),
            "selection_rule": comparison["selection_rule"],
            "validation_selection_metrics": next(
                result for result in comparison["results"] if result["model"] == best_name
            ),
            "final_test_metrics": final_test_metrics,
        },
    )

    comparison_receipt = {
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "installed_weights": str(installed),
        "selected_model": best_name,
        "source_weights": str(best_weights),
        "dataset": str(args.data.resolve()),
        "selection_rule": comparison["selection_rule"],
        "validation_selection_metrics": next(
            result for result in comparison["results"] if result["model"] == best_name
        ),
        "final_test_metrics": final_test_metrics,
    }
    (comparison_dir / "installed_plate_model.json").write_text(
        json.dumps(comparison_receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
