"""Read-only status reporting for local Ultralytics training runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from backend.api.deps import require_admin
from backend.config import settings
from backend.database.models import User

router = APIRouter(dependencies=[Depends(require_admin)])

_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "training" / "outputs"
_PREFERRED_RUN_NAME = "car_dataset_reg_yolov8n_416"


def _as_number(value: str | None) -> float | int | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _read_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, ImportError):
        return {}


def _read_history(path: Path) -> list[dict[str, float | int | None]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                {key.strip(): _as_number(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error):
        return []


def _trainer_processes() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        return []

    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "train_plate_detector.py" not in command:
                continue
            matches.append(
                {
                    "pid": process.info["pid"],
                    "command": command,
                    "started_at": datetime.fromtimestamp(
                        process.info["create_time"], tz=timezone.utc
                    ).isoformat(),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return matches


def _run_directories() -> list[Path]:
    if not _OUTPUT_ROOT.exists():
        return []
    return sorted(
        (path for path in _OUTPUT_ROOT.iterdir() if path.is_dir() and (path / "args.yaml").exists()),
        key=lambda path: (path / "args.yaml").stat().st_mtime,
        reverse=True,
    )


def _select_run(runs: list[Path]) -> Path | None:
    if not runs:
        return None

    for run in runs:
        if run.name == _PREFERRED_RUN_NAME:
            return run
    return runs[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _latest_report(filename: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        _OUTPUT_ROOT.glob(f"**/{filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        report = _read_json(path)
        if report is not None:
            return path, report
    return None, None


def _evaluation_reports() -> dict[str, Any]:
    metrics_path, metrics = _latest_report("metrics.json")
    theft_path, theft = _latest_report("theft_metrics.json")
    return {
        "detector": (metrics or {}).get("accuracy"),
        "ocr": (metrics or {}).get("ocr"),
        "performance": (metrics or {}).get("fps"),
        "theft": theft,
        "model_evaluated_at": (metrics or {}).get("evaluated_at"),
        "theft_evaluated_at": (theft or {}).get("evaluated_at"),
        "sources": {
            "model": str(metrics_path.relative_to(_OUTPUT_ROOT)) if metrics_path else None,
            "theft": str(theft_path.relative_to(_OUTPUT_ROOT)) if theft_path else None,
        },
    }


def _deployed_model_info() -> dict[str, Any]:
    receipt_path = settings.models_dir / "license_plate.install_receipt.json"
    receipt = _read_json(receipt_path) if receipt_path.exists() else None
    return {
        "vehicle_model": str(settings.resolve_model(settings.VEHICLE_MODEL_PATH)),
        "plate_model": str(settings.resolve_model(settings.PLATE_MODEL_PATH)),
        "plate_backend": settings.PLATE_DETECTOR_BACKEND,
        "plate_install_receipt": receipt,
    }


@router.get("/status")
def training_status(admin: User = Depends(require_admin)) -> dict[str, Any]:
    """Return process state and epoch metrics for the configured training run."""
    del admin
    runs = _run_directories()
    processes = _trainer_processes()
    if not runs:
        return {
            "status": "not_started",
            "running": bool(processes),
            "run_name": None,
            "history": [],
            "message": "No local training run was found.",
            "evaluation": _evaluation_reports(),
            "deployed": _deployed_model_info(),
        }

    run = _select_run(runs)
    args = _read_args(run / "args.yaml")
    history = _read_history(run / "results.csv")
    total_epochs = int(args.get("epochs") or 0)
    completed_epochs = len(history)
    matching_processes = [item for item in processes if run.name in item["command"]]
    running = bool(matching_processes)
    last_checkpoint = run / "weights" / "last.pt"
    best_checkpoint = run / "weights" / "best.pt"

    if running and completed_epochs == 0:
        status = "initializing"
        message = "Training is active; waiting for epoch 1 to complete."
    elif running:
        status = "running"
        message = f"Epoch {completed_epochs + 1} is in progress."
    elif total_epochs and completed_epochs >= total_epochs:
        status = "completed"
        message = "Training completed."
    elif completed_epochs:
        status = "stopped"
        message = f"Training stopped after epoch {completed_epochs}."
    else:
        status = "not_running"
        message = "A run exists, but no active trainer or completed epoch was found."

    elapsed_seconds = None
    started_at = None
    if matching_processes:
        started_at = matching_processes[0]["started_at"]
        started = datetime.fromisoformat(started_at)
        elapsed_seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))

    progress = (completed_epochs / total_epochs * 100) if total_epochs else 0.0
    return {
        "status": status,
        "running": running,
        "message": message,
        "run_name": run.name,
        "model": args.get("model"),
        "device": args.get("device"),
        "image_size": args.get("imgsz"),
        "batch_size": args.get("batch"),
        "total_epochs": total_epochs,
        "completed_epochs": completed_epochs,
        "progress_percent": round(progress, 2),
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "last_updated_at": datetime.fromtimestamp(
            max((run / "args.yaml").stat().st_mtime, (run / "results.csv").stat().st_mtime if (run / "results.csv").exists() else 0),
            tz=timezone.utc,
        ).isoformat(),
        "checkpoints": {
            "last_available": last_checkpoint.exists(),
            "best_available": best_checkpoint.exists(),
        },
        "artifacts": {
            "confusion_matrix": f"/training-artifacts/{run.name}/confusion_matrix.png"
            if (run / "confusion_matrix.png").exists()
            else None,
            "confusion_matrix_normalized": f"/training-artifacts/{run.name}/confusion_matrix_normalized.png"
            if (run / "confusion_matrix_normalized.png").exists()
            else None,
        },
        "latest": history[-1] if history else None,
        "history": history,
        "evaluation": _evaluation_reports(),
        "deployed": _deployed_model_info(),
    }
