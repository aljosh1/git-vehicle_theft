"""Compare detector configurations on one fixed Ultralytics test split.

Example:
    python training/compare_detectors.py --task plate --data dataset/plate_data.yaml \
        --model yolov8n=training/outputs/plate_v8/weights/best.pt \
        --model yolo11n=training/outputs/plate_11/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR, settings  # noqa: E402
from training.evaluate import benchmark_fps, evaluate_accuracy  # noqa: E402


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must use name=weights.pt")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("models must use name=weights.pt")
    return name.strip(), Path(path.strip())


def compare_models(
    models: list[tuple[str, Path]],
    data: Path,
    *,
    task: str,
    imgsz: int,
    device: str,
    runs: int,
    output_dir: Path,
    split: str = "val",
) -> dict:
    """Evaluate all models with identical validation and benchmark settings."""
    rows: list[dict] = []
    for name, weights in models:
        if not weights.exists() and weights.parent != Path("."):
            raise FileNotFoundError(f"weights not found: {weights}")
        detection = evaluate_accuracy(weights, data, imgsz, device, split=split)
        speed = benchmark_fps(weights, imgsz, device, runs=runs)
        rows.append(
            {
                "task": task,
                "model": name,
                "weights": str(weights),
                **detection,
                "mean_ms": speed["mean_ms"],
                "p95_ms": speed["p95_ms"],
                "fps": speed["fps"],
            }
        )

    rows.sort(key=lambda row: (row["map50_95"], row["map50"], row["fps"]), reverse=True)
    record = {
        "task": task,
        "data": str(data.resolve()),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "imgsz": imgsz,
        "device": device,
        "split": split,
        "selection_rule": "highest mAP@0.5:0.95, then mAP@0.5, then FPS",
        "best_model": rows[0]["model"],
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{task}_comparison.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{task}_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare detector models on one fixed dataset split"
    )
    parser.add_argument("--task", choices=("vehicle", "plate"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", type=parse_model, required=True, help="name=weights.pt"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "training" / "outputs" / "comparisons",
    )
    args = parser.parse_args()
    if len(args.model) < 2:
        parser.error("provide at least two --model arguments")
    if not args.data.exists():
        parser.error(f"dataset descriptor not found: {args.data}")

    record = compare_models(
        args.model,
        args.data,
        task=args.task,
        imgsz=args.imgsz,
        device=args.device,
        runs=args.runs,
        output_dir=args.output,
        split=args.split,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
