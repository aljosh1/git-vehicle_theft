"""Audit a fixed-split Roboflow YOLO export and generate project metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


@dataclass(frozen=True)
class SampleResult:
    split: str
    image: Path
    label: Path
    image_sha256: str
    label_sha256: str
    width: int
    height: int
    annotations: int
    class_counts: Counter[int]
    errors: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_group(stem: str) -> str:
    """Remove Roboflow's generated hash so augmented variants share a group."""
    return ROBOFLOW_SUFFIX.sub("", stem)


def audit_sample(split: str, image: Path, label: Path) -> SampleResult:
    errors: list[str] = []
    class_counts: Counter[int] = Counter()
    annotation_count = 0
    width = 0
    height = 0

    decoded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        errors.append("unreadable image")
    else:
        height, width = decoded.shape[:2]
        if width <= 0 or height <= 0:
            errors.append("invalid image dimensions")

    try:
        lines = label.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        lines = []
        errors.append(f"unreadable label: {exc}")

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        annotation_count += 1
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"label line {line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            raw_class, *raw_box = fields
            class_value = float(raw_class)
            class_id = int(class_value)
            box = [float(value) for value in raw_box]
        except ValueError:
            errors.append(f"label line {line_number}: non-numeric value")
            continue
        if not math.isfinite(class_value) or class_value != class_id:
            errors.append(f"label line {line_number}: class ID must be an integer")
            continue
        class_counts[class_id] += 1
        if class_id != 0:
            errors.append(f"label line {line_number}: unexpected class ID {class_id}")
        if not all(math.isfinite(value) for value in box):
            errors.append(f"label line {line_number}: non-finite box value")
            continue
        x_center, y_center, box_width, box_height = box
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            errors.append(f"label line {line_number}: center outside [0, 1]")
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            errors.append(f"label line {line_number}: dimensions outside (0, 1]")
        if x_center - box_width / 2 < -1e-6 or x_center + box_width / 2 > 1.0 + 1e-6:
            errors.append(f"label line {line_number}: horizontal box exceeds image")
        if y_center - box_height / 2 < -1e-6 or y_center + box_height / 2 > 1.0 + 1e-6:
            errors.append(f"label line {line_number}: vertical box exceeds image")

    return SampleResult(
        split=split,
        image=image,
        label=label,
        image_sha256=sha256(image),
        label_sha256=sha256(label),
        width=width,
        height=height,
        annotations=annotation_count,
        class_counts=class_counts,
        errors=tuple(errors),
    )


def discover(root: Path) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    samples: list[tuple[str, Path, Path]] = []
    errors: list[str] = []
    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images or labels directory")
            continue
        images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        for stem in sorted(images.keys() - labels.keys()):
            errors.append(f"{split}: image has no label: {images[stem].name}")
        for stem in sorted(labels.keys() - images.keys()):
            errors.append(f"{split}: label has no image: {labels[stem].name}")
        samples.extend((split, images[stem], labels[stem]) for stem in sorted(images.keys() & labels.keys()))
    return samples, errors


def write_outputs(root: Path, output: Path, results: list[SampleResult], source: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "split", "image", "label", "group_id", "width", "height",
            "annotations", "image_sha256", "label_sha256", "source_name",
            "source_url", "license",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.split, item.image.name)):
            writer.writerow({
                "split": result.split,
                "image": result.image.relative_to(root).as_posix(),
                "label": result.label.relative_to(root).as_posix(),
                "group_id": base_group(result.image.stem),
                "width": result.width,
                "height": result.height,
                "annotations": result.annotations,
                "image_sha256": result.image_sha256,
                "label_sha256": result.label_sha256,
                "source_name": "Roboflow License Plate v3",
                "source_url": source.get("url", ""),
                "license": source.get("license", ""),
            })

    descriptor = {
        "path": root.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "license-plate"},
        "roboflow": source,
    }
    (output / "plate_data.yaml").write_text(
        "# Generated by training/audit_roboflow_dataset.py\n"
        + yaml.safe_dump(descriptor, sort_keys=False),
        encoding="utf-8",
    )

    attribution = (
        "# Dataset Attribution\n\n"
        "License Plate v3 was exported from Roboflow Universe in YOLOv8 format.\n\n"
        f"- Source: {source.get('url', 'not provided')}\n"
        f"- License: {source.get('license', 'not provided')}\n"
        f"- Workspace: {source.get('workspace', 'not provided')}\n"
        f"- Project: {source.get('project', 'not provided')}\n"
        f"- Version: {source.get('version', 'not provided')}\n"
    )
    (output / "ROBOFLOW_DATASET.md").write_text(attribution, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Audit a fixed-split Roboflow YOLO dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "dataset" / "raw" / "roboflow" / "License Plate.v3i.yolov8",
    )
    parser.add_argument("--output", type=Path, default=project_root / "dataset")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    provider_yaml = source / "data.yaml"
    if not provider_yaml.is_file():
        raise SystemExit(f"provider descriptor not found: {provider_yaml}")
    provider_spec = yaml.safe_load(provider_yaml.read_text(encoding="utf-8")) or {}
    provider = provider_spec.get("roboflow", {})

    samples, discovery_errors = discover(source)
    print(f"Auditing {len(samples):,} paired samples with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(executor.map(lambda item: audit_sample(*item), samples))

    errors = discovery_errors + [
        f"{result.split}/{result.image.name}: {error}"
        for result in results
        for error in result.errors
    ]
    split_counts = Counter(result.split for result in results)
    annotation_counts: Counter[int] = Counter()
    for result in results:
        annotation_counts.update(result.class_counts)

    hashes: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    for result in results:
        hashes[result.image_sha256].add(result.split)
        groups[base_group(result.image.stem).casefold()].add(result.split)
    duplicate_leaks = {digest: splits for digest, splits in hashes.items() if len(splits) > 1}
    group_leaks = {group: splits for group, splits in groups.items() if len(splits) > 1}

    print("Split counts: " + ", ".join(f"{split}={split_counts[split]:,}" for split in SPLITS))
    print("Annotations: " + ", ".join(f"class {key}={value:,}" for key, value in sorted(annotation_counts.items())))
    print(f"Exact image hashes crossing splits: {len(duplicate_leaks):,}")
    print(f"Roboflow base groups crossing splits: {len(group_leaks):,}")

    if duplicate_leaks:
        errors.append(f"{len(duplicate_leaks)} exact image hashes occur across splits")
    if errors:
        print(f"Validation failed with {len(errors):,} errors:")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100:,} more")
        raise SystemExit(1)

    write_outputs(source, args.output.resolve(), results, provider)
    print(f"Wrote {(args.output / 'plate_data.yaml').resolve()}")
    print(f"Wrote {(args.output / 'manifest.csv').resolve()}")
    print(f"Wrote {(args.output / 'ROBOFLOW_DATASET.md').resolve()}")


if __name__ == "__main__":
    main()
