"""
Application-wide logging.

Design notes
------------
* One rotating file handler (`logs/vtds.log`, 5 x 2 MB) plus a coloured console
  handler, installed exactly once per process by `setup_logging()`.
* Every module calls `get_logger(__name__)` and never configures handlers
  itself - that avoids the classic "log lines printed five times" bug.
* `log_exception()` is the helper the rest of the codebase uses to record an
  error together with its traceback.

How to log an error
-------------------
    from backend.utils.logger import get_logger
    log = get_logger(__name__)

    try:
        risky_call()
    except Exception:
        # exc_info=True attaches the full traceback. Inside an `except`
        # block, `log.exception(...)` is the shorter equivalent.
        log.exception("Plate OCR failed for camera %s", camera_id)

    # Non-fatal problem, no exception object:
    log.error("Model weights missing at %s", path)

Always pass values as `%s` arguments rather than f-strings: formatting is then
deferred until the record is actually emitted, and identical messages stay
groupable in log analysis.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI colours for the console handler.
_COLOURS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[41m",  # red background
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Console formatter that tints the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        original = record.levelname
        if colour:
            record.levelname = f"{colour}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original  # keep the record reusable


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """
    Install the console + rotating-file handlers on the root logger.

    Idempotent: calling it twice is a no-op, so it is safe to invoke from both
    the FastAPI startup hook and standalone scripts.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- console ---------------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ColourFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # --- rotating file ---------------------------------------------------
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "vtds.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )
        root.addHandler(file_handler)

    # Third-party libraries are extremely chatty at INFO - quieten them so the
    # detection log stays readable.
    for noisy in (
        "urllib3",
        "PIL",
        "matplotlib",
        "watchfiles",
        "python_multipart",
        "httpx",
        "twilio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger. Use `get_logger(__name__)` at import time."""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, *args: Any) -> None:
    """
    Record an error with its traceback.

    Thin wrapper over `logger.error(..., exc_info=True)` that exists so error
    reporting looks identical everywhere in the codebase:

        except Exception:
            log_exception(log, "could not read frame from camera %s", cam_id)
    """
    logger.error(message, *args, exc_info=True)
