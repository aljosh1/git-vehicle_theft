"""Install the active plate detector with backup and receipt logging."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_plate_model(
    source_weights: Path,
    *,
    installed_weights: Path | None = None,
    trigger: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a plate model into place, preserving the previous one and logging it."""
    source = source_weights.resolve()
    if not source.exists():
        raise FileNotFoundError(f"source weights not found: {source}")

    settings.ensure_directories()
    target = (installed_weights or (settings.models_dir / "license_plate.pt")).resolve()
    receipts_dir = settings.models_dir / "install_receipts"
    backups_dir = settings.models_dir / "backups"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path: Path | None = None
    backup_hash: str | None = None

    if target.exists():
        backup_path = backups_dir / f"{target.stem}_{timestamp}{target.suffix}"
        shutil.copy2(target, backup_path)
        backup_hash = _sha256(backup_path)

    shutil.copy2(source, target)

    receipt = {
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "trigger": trigger,
        "source_weights": str(source),
        "installed_weights": str(target),
        "backup_weights": str(backup_path) if backup_path else None,
        "source_hash": _sha256(source),
        "installed_hash": _sha256(target),
        "backup_hash": backup_hash,
        "metadata": metadata or {},
    }

    history_path = receipts_dir / f"{target.stem}_{timestamp}.json"
    latest_path = settings.models_dir / f"{target.stem}.install_receipt.json"
    history_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install a trained plate detector with backup and receipt logging"
    )
    parser.add_argument("source", type=Path, help="source weights to install")
    parser.add_argument(
        "--target",
        type=Path,
        default=settings.models_dir / "license_plate.pt",
        help="installed weights path",
    )
    parser.add_argument(
        "--reason",
        default="manual install",
        help="human-readable reason recorded in the receipt",
    )
    args = parser.parse_args()

    receipt = install_plate_model(
        args.source,
        installed_weights=args.target,
        trigger=args.reason,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()