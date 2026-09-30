"""Reusable metrics for OCR and end-to-end theft evaluation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OcrSample:
    image: Path
    ground_truth: str
    label: Path | None = None
    split: str = "test"
    group_id: str = ""
    condition: str = "unspecified"


@dataclass(frozen=True)
class TheftSample:
    sample_id: str
    expected_stolen: bool
    predicted_stolen: bool
    processing_ms: float
    condition: str = "unspecified"


def normalise_plate(value: str | None) -> str:
    """Upper-case plate text and remove formatting/non-alphanumeric noise."""
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def edit_distance(expected: str, predicted: str) -> int:
    """Compute Levenshtein distance with O(min(n, m)) memory."""
    expected = normalise_plate(expected)
    predicted = normalise_plate(predicted)
    if len(expected) < len(predicted):
        expected, predicted = predicted, expected
    previous = list(range(len(predicted) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, predicted_char in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_char != predicted_char),
                )
            )
        previous = current
    return previous[-1]


def crop_from_yolo_label(image: np.ndarray, label_path: Path | None) -> np.ndarray:
    """Crop the largest class-0 bounding box from a YOLO annotation file."""
    if label_path is None or not label_path.exists():
        raise ValueError(f"OCR sample needs an existing YOLO label: {label_path}")
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for line_number, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO values")
        class_id, cx, cy, box_width, box_height = map(float, parts)
        if int(class_id) != 0:
            continue
        x1 = max(0, int((cx - box_width / 2) * width))
        y1 = max(0, int((cy - box_height / 2) * height))
        x2 = min(width, int((cx + box_width / 2) * width))
        y2 = min(height, int((cy + box_height / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        raise ValueError(f"no valid class-0 plate box in {label_path}")
    x1, y1, x2, y2 = max(
        boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
    )
    return image[y1:y2, x1:x2]


def ocr_metrics(rows: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    """Return micro character accuracy, CER, and exact plate accuracy."""
    samples = [(normalise_plate(gt), normalise_plate(pred)) for gt, pred in rows]
    if not samples:
        raise ValueError("OCR evaluation needs at least one sample")

    total_characters = sum(len(gt) for gt, _ in samples)
    total_edits = sum(edit_distance(gt, pred) for gt, pred in samples)
    exact_matches = sum(gt == pred for gt, pred in samples)
    cer = total_edits / total_characters if total_characters else 0.0
    return {
        "samples": len(samples),
        "total_characters": total_characters,
        "edit_errors": total_edits,
        "character_accuracy": max(0.0, 1.0 - cer),
        "character_error_rate": cer,
        "exact_plate_accuracy": exact_matches / len(samples),
        "exact_matches": exact_matches,
    }


def binary_metrics(samples: Iterable[TheftSample]) -> dict[str, float | int]:
    """Calculate theft classification metrics and latency statistics."""
    rows = list(samples)
    if not rows:
        raise ValueError("theft evaluation needs at least one sample")
    tp = sum(row.expected_stolen and row.predicted_stolen for row in rows)
    tn = sum(not row.expected_stolen and not row.predicted_stolen for row in rows)
    fp = sum(not row.expected_stolen and row.predicted_stolen for row in rows)
    fn = sum(row.expected_stolen and not row.predicted_stolen for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negatives = tn + fp
    timings = sorted(row.processing_ms for row in rows)
    p95_index = max(0, int(0.95 * len(timings) + 0.999999) - 1)
    return {
        "samples": len(rows),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alert_rate": fp / negatives if negatives else 0.0,
        "mean_processing_ms": sum(timings) / len(timings),
        "p95_processing_ms": timings[p95_index],
    }


def read_ocr_manifest(path: Path, *, split: str = "test") -> list[OcrSample]:
    """Load OCR labels from a CSV manifest with paths relative to the manifest."""
    samples: list[OcrSample] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if not row.get("image") or not row.get("plate_text"):
                raise ValueError(f"{path}:{row_number}: image and plate_text are required")
            row_split = (row.get("split") or "test").strip().lower()
            if split and row_split != split.lower():
                continue
            image = Path(row["image"])
            if not image.is_absolute():
                image = (path.parent / image).resolve()
            label_value = (row.get("label") or "").strip()
            label = Path(label_value) if label_value else None
            if label is not None and not label.is_absolute():
                label = (path.parent / label).resolve()
            samples.append(
                OcrSample(
                    image=image,
                    ground_truth=normalise_plate(row["plate_text"]),
                    label=label,
                    split=row_split,
                    group_id=(row.get("group_id") or image.stem).strip(),
                    condition=(row.get("condition") or "unspecified").strip().lower(),
                )
            )
    return samples
