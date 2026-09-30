"""
Dataset preparation for YOLOv8 training.

    python training/prepare_dataset.py --source raw_images/ --split 70,20,10

Takes a flat folder of images (and their YOLO-format `.txt` labels, if already
annotated) and produces the train/val/test layout Ultralytics expects, plus the
`data.yaml` that points at it.

    dataset/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── plate_data.yaml

Why a deterministic split
-------------------------
The shuffle is seeded, so re-running the script reproduces exactly the same
split.  That matters for a project report: an evaluation number is meaningless
if the test set silently changes between runs.

Label pairing
-------------
An image is only usable for training if its label file exists.  Images without
one are reported and skipped rather than copied in silently - an unlabelled
image in the training set teaches the model that plates do not exist there,
which actively degrades recall.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.logger import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


VALID_CONDITIONS = {
    "day", "night", "rain", "blur", "dirt", "angle", "occlusion", "unspecified"
}


@dataclass(frozen=True)
class DatasetItem:
    image: Path
    label: Path | None
    group_id: str
    condition: str = "unspecified"
    plate_text: str = ""
    source_name: str = "local"

    @property
    def output_stem(self) -> str:
        """Collision-resistant stable name for files collected from many folders."""
        key = self.image.as_posix().encode("utf-8")
        return f"{hashlib.sha1(key).hexdigest()[:10]}_{self.image.stem}"


def load_metadata(path: Path | None, source: Path) -> dict[str, dict[str, str]]:
    """Load optional metadata keyed by source-relative image path."""
    if path is None:
        return {}
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            image = (row.get("image") or "").replace("\\", "/").strip()
            if not image:
                raise ValueError(f"{path}:{line}: image is required")
            condition = (row.get("condition") or "unspecified").strip().lower()
            if condition not in VALID_CONDITIONS:
                raise ValueError(
                    f"{path}:{line}: condition must be one of {sorted(VALID_CONDITIONS)}"
                )
            records[image.casefold()] = {
                "group_id": (row.get("group_id") or Path(image).stem).strip(),
                "condition": condition,
                "plate_text": (row.get("plate_text") or "").strip().upper(),
                "source_name": (row.get("source_name") or "local").strip(),
            }
    return records


def find_pairs(
    source: Path, metadata: dict[str, dict[str, str]] | None = None
) -> tuple[list[DatasetItem], list[Path]]:
    """
    Match each image to its YOLO label file.

    Looks for `<name>.txt` beside the image, then in a sibling `labels/` folder
    - the two layouts annotation tools produce.

    Returns:
        `(pairs, unlabelled)` where pairs is [(image, label_or_None), ...].
    """
    pairs: list[DatasetItem] = []
    unlabelled: list[Path] = []
    metadata = metadata or {}

    for image in sorted(source.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        candidates = [
            image.with_suffix(".txt"),
            image.parent / "labels" / f"{image.stem}.txt",
            image.parent.parent / "labels" / f"{image.stem}.txt",
        ]
        label = next((c for c in candidates if c.exists()), None)
        if label is None:
            unlabelled.append(image)
        relative = image.relative_to(source).as_posix()
        details = metadata.get(relative.casefold(), {})
        pairs.append(
            DatasetItem(
                image=image,
                label=label,
                group_id=details.get("group_id", image.stem),
                condition=details.get("condition", "unspecified"),
                plate_text=details.get("plate_text", ""),
                source_name=details.get("source_name", "local"),
            )
        )

    return pairs, unlabelled


def split_dataset(
    pairs: list[DatasetItem],
    ratios: tuple[float, float, float],
    seed: int = 42,
) -> dict[str, list[DatasetItem]]:
    """Assign whole source groups to splits to prevent sequence leakage."""
    groups: dict[str, list[DatasetItem]] = {}
    for item in pairs:
        groups.setdefault(item.group_id, []).append(item)
    shuffled = sorted(groups.items())
    random.Random(seed).shuffle(shuffled)

    targets = [len(pairs) * ratio for ratio in ratios]
    result: dict[str, list[DatasetItem]] = {split: [] for split in SPLITS}
    for _, items in shuffled:
        deficits = [targets[index] - len(result[split]) for index, split in enumerate(SPLITS)]
        eligible = [index for index, deficit in enumerate(deficits) if deficit > 0]
        index = max(eligible, key=lambda i: deficits[i]) if eligible else 2
        result[SPLITS[index]].extend(items)
    return result


def write_split(
    splits: dict[str, list[DatasetItem]],
    output: Path,
    *,
    move: bool = False,
    skip_unlabelled: bool = True,
) -> dict[str, int]:
    """Copy (or move) files into the YOLO directory layout."""
    counts: dict[str, int] = {}
    transfer = shutil.move if move else shutil.copy2

    for split in SPLITS:
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for item in splits[split]:
            if item.label is None and skip_unlabelled:
                continue
            image_name = f"{item.output_stem}{item.image.suffix.lower()}"
            transfer(str(item.image), str(image_dir / image_name))
            if item.label is not None:
                transfer(str(item.label), str(label_dir / f"{item.output_stem}.txt"))
            written += 1
        counts[split] = written

    return counts


def write_manifest(
    splits: dict[str, list[DatasetItem]],
    output: Path,
    *,
    skip_unlabelled: bool = True,
) -> Path:
    """Write an auditable sample manifest and reject duplicate image leakage."""
    manifest = output / "manifest.csv"
    hashes: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for split, items in splits.items():
        for item in items:
            if item.label is None and skip_unlabelled:
                continue
            digest = hashlib.sha256(item.image.read_bytes()).hexdigest()
            previous = hashes.get(digest)
            if previous and previous != split:
                raise ValueError(
                    f"duplicate image content occurs in both {previous} and {split}: {item.image}"
                )
            hashes[digest] = split
            rows.append(
                {
                    "image": f"images/{split}/{item.output_stem}{item.image.suffix.lower()}",
                    "label": f"labels/{split}/{item.output_stem}.txt" if item.label else "",
                    "split": split,
                    "group_id": item.group_id,
                    "condition": item.condition,
                    "plate_text": item.plate_text,
                    "source_name": item.source_name,
                    "sha256": digest,
                }
            )
    if not rows:
        raise ValueError("cannot write a manifest with no usable samples")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def write_data_yaml(output: Path, class_names: list[str]) -> Path:
    """
    Write the Ultralytics dataset descriptor.

    Paths are absolute: Ultralytics resolves relative paths against its own
    settings directory, not the working directory, which is a common and
    confusing source of "dataset not found" errors.
    """
    path = output / "plate_data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    path.write_text(
        f"# Generated by training/prepare_dataset.py\n"
        f"path: {output.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLOv8 dataset from a flat image folder"
    )
    parser.add_argument("--source", required=True, type=Path, help="folder of images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="destination dataset root",
    )
    parser.add_argument(
        "--split", default="70,20,10", help="train,val,test percentages"
    )
    parser.add_argument("--classes", default="license_plate", help="comma-separated")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="CSV columns: image,group_id,condition,plate_text,source_name",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--move", action="store_true", help="move files instead of copying"
    )
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="copy images that have no label file (normally a mistake)",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    if not args.source.exists():
        log.error("source folder does not exist: %s", args.source)
        raise SystemExit(1)

    try:
        parts = [float(p) for p in args.split.split(",")]
        if len(parts) != 3 or any(part <= 0 for part in parts):
            raise ValueError
        ratios = tuple(p / sum(parts) for p in parts)
    except ValueError:
        log.error("--split must be three numbers, e.g. 70,20,10")
        raise SystemExit(1) from None

    try:
        metadata = load_metadata(args.metadata, args.source)
    except (OSError, ValueError) as exc:
        log.error("invalid metadata: %s", exc)
        raise SystemExit(1) from None

    pairs, unlabelled = find_pairs(args.source, metadata)
    if not pairs:
        log.error("no images found under %s", args.source)
        raise SystemExit(1)

    log.info("found %d images (%d without labels)", len(pairs), len(unlabelled))
    if unlabelled and not args.include_unlabelled:
        log.warning(
            "%d unlabelled images will be SKIPPED. Annotate them with LabelImg, "
            "or pass --include-unlabelled to copy them anyway.",
            len(unlabelled),
        )
        for image in unlabelled[:10]:
            log.warning("  unlabelled: %s", image.name)
        if len(unlabelled) > 10:
            log.warning("  ... and %d more", len(unlabelled) - 10)

    usable_pairs = pairs if args.include_unlabelled else [
        item for item in pairs if item.label is not None
    ]
    if not usable_pairs:
        log.error("no labelled images are available for dataset preparation")
        raise SystemExit(1)
    splits = split_dataset(usable_pairs, ratios, seed=args.seed)
    counts = write_split(
        splits,
        args.output,
        move=args.move,
        skip_unlabelled=not args.include_unlabelled,
    )
    manifest_path = write_manifest(
        splits, args.output, skip_unlabelled=not args.include_unlabelled
    )
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    yaml_path = write_data_yaml(args.output, class_names)

    print("\n  Dataset prepared")
    for split in SPLITS:
        print(f"     {split:5} : {counts[split]:5d} images")
    print(f"\n  Descriptor : {yaml_path}")
    print(f"  Manifest   : {manifest_path}")
    print(f"  Next step  : python training/train_plate_detector.py --data {yaml_path}\n")


if __name__ == "__main__":
    main()
