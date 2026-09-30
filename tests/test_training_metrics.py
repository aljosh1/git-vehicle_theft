"""Focused tests for research metrics and leakage-safe dataset preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.metrics import (
    TheftSample,
    binary_metrics,
    crop_from_yolo_label,
    edit_distance,
    normalise_plate,
    ocr_metrics,
)
from training.prepare_dataset import DatasetItem, split_dataset


def test_plate_normalisation_and_edit_distance() -> None:
    assert normalise_plate("lag-123 xy") == "LAG123XY"
    assert edit_distance("ABC123", "ABC12B") == 1
    assert edit_distance("ABC123", "") == 6


def test_ocr_metrics_count_failed_reads_as_errors() -> None:
    metrics = ocr_metrics(
        [("ABC123", "ABC123"), ("XYZ789", "XYZ78B"), ("LAG456", "")]
    )
    assert metrics["samples"] == 3
    assert metrics["total_characters"] == 18
    assert metrics["edit_errors"] == 7
    assert metrics["character_accuracy"] == pytest.approx(11 / 18)
    assert metrics["character_error_rate"] == pytest.approx(7 / 18)
    assert metrics["exact_plate_accuracy"] == pytest.approx(1 / 3)


def test_binary_theft_metrics_include_false_alert_rate_and_latency() -> None:
    metrics = binary_metrics(
        [
            TheftSample("tp", True, True, 10),
            TheftSample("fp", False, True, 20),
            TheftSample("tn", False, False, 30),
            TheftSample("fn", True, False, 40),
        ]
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["false_alert_rate"] == 0.5
    assert metrics["mean_processing_ms"] == 25
    assert metrics["p95_processing_ms"] == 40


def test_group_aware_split_never_leaks_a_sequence() -> None:
    items = [
        DatasetItem(Path(f"sequence_{group}_{frame}.jpg"), None, f"sequence-{group}")
        for group in range(10)
        for frame in range(3)
    ]
    splits = split_dataset(items, (0.7, 0.2, 0.1), seed=42)
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 21,
        "val": 6,
        "test": 3,
    }
    for group in {item.group_id for item in items}:
        containing_splits = [
            name for name, rows in splits.items() if any(item.group_id == group for item in rows)
        ]
        assert len(containing_splits) == 1


def test_group_split_is_reproducible() -> None:
    items = [DatasetItem(Path(f"{index}.jpg"), None, str(index)) for index in range(20)]
    first = split_dataset(items, (0.7, 0.2, 0.1), seed=7)
    second = split_dataset(items, (0.7, 0.2, 0.1), seed=7)
    assert {
        split: [item.image.name for item in rows] for split, rows in first.items()
    } == {
        split: [item.image.name for item in rows] for split, rows in second.items()
    }


def test_yolo_annotation_is_converted_to_plate_crop(tmp_path: Path) -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[25:75, 50:150] = 255
    label = tmp_path / "plate.txt"
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    crop = crop_from_yolo_label(image, label)

    assert crop.shape == (50, 100, 3)
    assert crop.mean() == 255
