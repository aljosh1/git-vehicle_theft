"""
Model evaluation and benchmarking.

    python training/evaluate.py --weights models/license_plate.pt
    python training/evaluate.py --weights models/yolov8n.pt --fps-only

Produces every metric the project brief asks for - precision, recall, F1,
mAP@0.5, mAP@0.5:0.95 and FPS - as a CSV table, a JSON record and matplotlib
graphs suitable for pasting straight into a project report.

Outputs land in `training/outputs/evaluation_<timestamp>/`:

    metrics.csv          the headline table
    metrics.json         machine-readable, for later comparison
    metrics_bar.png      precision / recall / F1 / mAP bar chart
    fps_benchmark.png    latency distribution
    summary.txt          human-readable report section

On accuracy vs F1
-----------------
"Accuracy" is reported for completeness because the brief lists it, but for
object detection it is the least meaningful of these numbers: a detector is
scored over boxes, not over a fixed set of labelled samples, so there is no
well-defined true-negative count. mAP is the metric that actually characterises
a detector, and it is the one to lead with in the report. The value printed here
is the F1-equivalent accuracy proxy, and it is labelled as such rather than
passed off as classification accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR, settings  # noqa: E402
from backend.utils.logger import get_logger, setup_logging  # noqa: E402
from training.metrics import (  # noqa: E402
    crop_from_yolo_label,
    normalise_plate,
    ocr_metrics,
    read_ocr_manifest,
)

log = get_logger(__name__)

OUTPUT_ROOT = BASE_DIR / "training" / "outputs"


def evaluate_accuracy(
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    device: str,
    *,
    split: str = "test",
) -> dict:
    """
    Run Ultralytics validation and extract the detection metrics.

    Returns a dict of floats; missing keys are returned as None rather than 0.0,
    so a failed measurement is never mistaken for a genuine zero.
    """
    from ultralytics import YOLO

    log.info("evaluating %s against %s split in %s", weights.name, split, data_yaml)
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=True,
    )

    box = results.box
    precision = float(box.mp)          # mean precision across classes
    recall = float(box.mr)             # mean recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy_proxy_f1": round(f1, 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "num_classes": len(getattr(box, "maps", []) or []) or 1,
    }


def benchmark_fps(
    weights: Path, imgsz: int, device: str, runs: int = 50, warmup: int = 5
) -> dict:
    """
    Measure inference latency on synthetic frames.

    Synthetic input is fine here: convolution cost depends on tensor shape, not
    on picture content. The first few runs are discarded because the initial
    inference includes graph construction and is several times slower - keeping
    them would understate steady-state throughput.
    """
    from ultralytics import YOLO

    log.info("benchmarking %s at %dpx on %s", weights.name, imgsz, device)
    model = YOLO(str(weights))
    model.to(device)

    frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(frame, verbose=False, device=device)

    timings_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        model.predict(frame, verbose=False, device=device)
        timings_ms.append((time.perf_counter() - started) * 1000)

    mean_ms = statistics.mean(timings_ms)
    return {
        "device": device,
        "imgsz": imgsz,
        "runs": runs,
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(statistics.median(timings_ms), 2),
        "p95_ms": round(sorted(timings_ms)[int(runs * 0.95) - 1], 2),
        "min_ms": round(min(timings_ms), 2),
        "max_ms": round(max(timings_ms), 2),
        "stdev_ms": round(statistics.stdev(timings_ms), 2) if runs > 1 else 0.0,
        "fps": round(1000 / mean_ms, 2) if mean_ms > 0 else 0.0,
        "_timings": timings_ms,
    }


def evaluate_ocr(manifest: Path, output_dir: Path, split: str = "test") -> dict:
    """Run the configured OCR backend on ground-truth plate crops."""
    import cv2

    from backend.recognition.plate_ocr import read_plate

    samples = read_ocr_manifest(manifest, split=split)
    if not samples:
        raise ValueError(f"no OCR samples for split {split!r} in {manifest}")

    predictions: list[dict[str, str | float]] = []
    grouped: dict[str, list[tuple[str, str]]] = {}
    for sample in samples:
        image = cv2.imread(str(sample.image))
        if image is None:
            raise ValueError(f"could not read OCR image: {sample.image}")
        crop = crop_from_yolo_label(image, sample.label)
        started = time.perf_counter()
        result = read_plate(crop)
        elapsed_ms = (time.perf_counter() - started) * 1000
        predicted, confidence = result if result else ("", 0.0)
        predicted = normalise_plate(predicted)
        predictions.append(
            {
                "image": str(sample.image),
                "split": sample.split,
                "group_id": sample.group_id,
                "condition": sample.condition,
                "ground_truth": sample.ground_truth,
                "prediction": predicted,
                "confidence": round(float(confidence), 6),
                "edit_distance": ocr_metrics([(sample.ground_truth, predicted)])["edit_errors"],
                "exact_match": int(sample.ground_truth == predicted),
                "processing_ms": round(elapsed_ms, 3),
            }
        )
        grouped.setdefault(sample.condition, []).append((sample.ground_truth, predicted))

    with (output_dir / "ocr_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    overall = ocr_metrics(
        (str(row["ground_truth"]), str(row["prediction"])) for row in predictions
    )
    overall["mean_processing_ms"] = round(
        statistics.mean(float(row["processing_ms"]) for row in predictions), 3
    )
    return {
        "manifest": str(manifest),
        "split": split,
        "overall": overall,
        "by_condition": {
            condition: ocr_metrics(rows) for condition, rows in sorted(grouped.items())
        },
    }


def write_graphs(output_dir: Path, accuracy: dict | None, fps: dict | None) -> None:
    """Save the bar chart and the latency histogram."""
    try:
        import matplotlib

        matplotlib.use("Agg")          # no display on a headless machine
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed - skipping graphs")
        return

    if accuracy:
        labels = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
        values = [
            accuracy["precision"],
            accuracy["recall"],
            accuracy["f1_score"],
            accuracy["map50"],
            accuracy["map50_95"],
        ]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        bars = axis.bar(labels, values, color="#2563eb")
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Score")
        axis.set_title("Detection performance")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
        figure.tight_layout()
        figure.savefig(output_dir / "metrics_bar.png", dpi=150)
        plt.close(figure)

    if fps and fps.get("_timings"):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(fps["_timings"], bins=20, color="#0891b2", edgecolor="white")
        axis.axvline(
            fps["mean_ms"],
            color="#b91c1c",
            linestyle="--",
            label=f"mean {fps['mean_ms']:.1f} ms ({fps['fps']:.1f} FPS)",
        )
        axis.set_xlabel("Inference latency (ms)")
        axis.set_ylabel("Frames")
        axis.set_title(f"Latency distribution on {fps['device']}")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "fps_benchmark.png", dpi=150)
        plt.close(figure)

    log.info("graphs written to %s", output_dir)


def write_reports(output_dir: Path, record: dict) -> None:
    """Write metrics.json, metrics.csv and a human-readable summary.txt."""
    (output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    accuracy = record.get("accuracy") or {}
    ocr = record.get("ocr") or {}
    fps = record.get("fps") or {}

    rows = [("Metric", "Value")]
    if accuracy:
        rows += [
            ("Precision", f"{accuracy['precision']:.4f}"),
            ("Recall", f"{accuracy['recall']:.4f}"),
            ("F1 score", f"{accuracy['f1_score']:.4f}"),
            ("mAP@0.5", f"{accuracy['map50']:.4f}"),
            ("mAP@0.5:0.95", f"{accuracy['map50_95']:.4f}"),
        ]
    if ocr:
        overall = ocr["overall"]
        rows += [
            ("OCR character accuracy", f"{overall['character_accuracy']:.4f}"),
            ("OCR character error rate", f"{overall['character_error_rate']:.4f}"),
            ("OCR exact plate accuracy", f"{overall['exact_plate_accuracy']:.4f}"),
            ("OCR samples", str(overall["samples"])),
        ]
    if fps:
        rows += [
            ("Mean latency (ms)", f"{fps['mean_ms']:.2f}"),
            ("Median latency (ms)", f"{fps['median_ms']:.2f}"),
            ("P95 latency (ms)", f"{fps['p95_ms']:.2f}"),
            ("Throughput (FPS)", f"{fps['fps']:.2f}"),
            ("Device", str(fps["device"])),
        ]

    (output_dir / "metrics.csv").write_text(
        "\n".join(f"{name},{value}" for name, value in rows), encoding="utf-8"
    )

    lines = [
        "=" * 58,
        "  MODEL EVALUATION REPORT",
        "=" * 58,
        f"  Model      : {record['weights']}",
        f"  Evaluated  : {record['evaluated_at']}",
        "",
    ]
    for name, value in rows[1:]:
        lines.append(f"  {name:<22}: {value}")

    if fps:
        target = settings.TARGET_FPS
        verdict = (
            f"meets the {target} FPS real-time target"
            if fps["fps"] >= target
            else (
                f"below the {target} FPS target - raise DETECT_EVERY_N_FRAMES, "
                f"reduce the input size, or use a GPU"
            )
        )
        lines += ["", f"  Real-time  : {verdict}"]

    lines += ["=" * 58, ""]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model")
    parser.add_argument(
        "--weights",
        type=Path,
        default=settings.models_dir / "license_plate.pt",
        help="model weights to evaluate",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "dataset" / "plate_data.yaml",
        help="dataset yaml (omit with --fps-only)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--ocr-manifest",
        type=Path,
        help="CSV with image,plate_text,split,group_id,condition columns",
    )
    parser.add_argument("--ocr-split", default="test")
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--runs", type=int, default=50, help="FPS benchmark iterations")
    parser.add_argument(
        "--fps-only",
        action="store_true",
        help="benchmark speed only - no labelled dataset required",
    )
    args = parser.parse_args()

    setup_logging("INFO", settings.logs_dir)

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        log.error("ultralytics is not installed; run: pip install ultralytics")
        raise SystemExit(1) from None

    weights = args.weights
    if not weights.exists():
        # Ultralytics resolves bare names like "yolov8n.pt" itself, downloading
        # if needed - so only a path-like argument is a hard error.
        if weights.parent != Path("."):
            log.error(
                "weights not found: %s\nTrain one first: "
                "python training/train_plate_detector.py",
                weights,
            )
            raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "weights": str(weights),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "imgsz": args.imgsz,
        "device": args.device,
        "accuracy": None,
        "ocr": None,
        "fps": None,
    }

    if not args.fps_only:
        if not args.data.exists():
            log.warning(
                "dataset %s not found - running the speed benchmark only. "
                "Pass --fps-only to silence this.",
                args.data,
            )
        else:
            try:
                record["accuracy"] = evaluate_accuracy(
                    weights, args.data, args.imgsz, args.device, split=args.split
                )
            except Exception:
                log.exception("accuracy evaluation failed - continuing with FPS only")

    if args.ocr_manifest:
        if not args.ocr_manifest.exists():
            log.error("OCR manifest not found: %s", args.ocr_manifest)
            raise SystemExit(1)
        try:
            record["ocr"] = evaluate_ocr(
                args.ocr_manifest, output_dir, split=args.ocr_split
            )
        except Exception:
            log.exception("OCR evaluation failed")

    try:
        fps = benchmark_fps(weights, args.imgsz, args.device, runs=args.runs)
        record["fps"] = fps
    except Exception:
        log.exception("FPS benchmark failed")

    write_graphs(output_dir, record["accuracy"], record["fps"])

    # Strip the raw timing list before serialising - it is large and already
    # summarised by the percentiles.
    if record["fps"]:
        record["fps"] = {k: v for k, v in record["fps"].items() if k != "_timings"}

    write_reports(output_dir, record)
    print(f"  Artefacts: {output_dir}\n")


if __name__ == "__main__":
    main()