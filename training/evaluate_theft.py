"""Evaluate integrated image-to-alert decisions on labelled theft scenarios.

Manifest columns:
    sample_id,image,expected_stolen,split,condition

The application database and model configuration must be prepared before this
script runs. The expected label is used only after inference to score the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR, settings  # noqa: E402
from backend.core.pipeline import Pipeline  # noqa: E402
from training.metrics import TheftSample, binary_metrics  # noqa: E402


def parse_bool(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "stolen"}:
        return True
    if normalised in {"0", "false", "no", "not_stolen", "not-stolen"}:
        return False
    raise ValueError(f"invalid boolean label: {value!r}")


def evaluate_scenarios(
    manifest: Path,
    output_dir: Path,
    *,
    split: str = "test",
    threshold: int | None = None,
) -> dict:
    """Run each labelled image through the complete synchronous pipeline."""
    threshold = threshold if threshold is not None else settings.THREAT_ALERT_THRESHOLD
    pipeline = Pipeline(
        camera_id="theft-evaluation",
        source="manifest",
        enable_tracking=False,
        enable_alerts=False,
    )
    predictions: list[dict] = []
    samples: list[TheftSample] = []

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            row_split = (row.get("split") or "test").strip().lower()
            if split and row_split != split.lower():
                continue
            if not row.get("image") or not row.get("expected_stolen"):
                raise ValueError(
                    f"{manifest}:{line}: image and expected_stolen are required"
                )
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"could not read scenario image: {image_path}")

            _, result = pipeline.analyse_image(frame, dispatch_alerts=False)
            expected = parse_bool(row["expected_stolen"])
            predicted = result.threat_score >= threshold
            sample_id = (row.get("sample_id") or image_path.stem).strip()
            condition = (row.get("condition") or "unspecified").strip().lower()
            samples.append(
                TheftSample(sample_id, expected, predicted, result.processing_ms, condition)
            )
            predictions.append(
                {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "condition": condition,
                    "expected_stolen": int(expected),
                    "predicted_stolen": int(predicted),
                    "correct": int(expected == predicted),
                    "plate_text": result.plate_text or "",
                    "plate_in_database": result.plate_in_database,
                    "flagged_stolen": result.vehicle_flagged_stolen,
                    "threat_score": result.threat_score,
                    "threat_level": result.threat_level_name,
                    "reason": result.threat_reason,
                    "processing_ms": round(result.processing_ms, 3),
                }
            )

    if not samples:
        raise ValueError(f"no scenario samples for split {split!r} in {manifest}")
    grouped = {
        condition: [sample for sample in samples if sample.condition == condition]
        for condition in sorted({sample.condition for sample in samples})
    }
    report = {
        "manifest": str(manifest.resolve()),
        "split": split,
        "threshold": threshold,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": binary_metrics(samples),
        "by_condition": {
            condition: binary_metrics(rows) for condition, rows in grouped.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "theft_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "theft_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end theft decisions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=int, default=settings.THREAT_ALERT_THRESHOLD)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "training" / "outputs" / "theft_evaluation",
    )
    args = parser.parse_args()
    if not args.manifest.exists():
        parser.error(f"manifest not found: {args.manifest}")
    print(
        json.dumps(
            evaluate_scenarios(
                args.manifest,
                args.output,
                split=args.split,
                threshold=args.threshold,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
