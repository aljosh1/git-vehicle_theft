"""
License plate OCR using EasyOCR.

Takes a plate crop (from `PlateDetector.detect()`) and extracts the alphanumeric
text.  Preprocessing + EasyOCR + post-correction for common OCR confusions.

Fallback chain:
    EasyOCR (GPU/CPU) → PaddleOCR (if easyocr unavailable) → None

The module loads gracefully when no OCR engine is installed, so the server still
boots, and `/api/info` reports the capability as missing.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from backend.config import settings
from backend.database.models import Vehicle
from backend.utils.logger import get_logger

log = get_logger(__name__)

_OCR_ENGINE: str | None = None
_easyocr_reader = None
_paddleocr_reader = None
_PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NIGERIA_STATE_NOISE = {
    "ABUJA",
    "LAGOS",
    "KANO",
    "KADUNA",
    "RIVERS",
    "OGUN",
    "OYO",
    "ENUGU",
    "ANAMBRA",
    "DELTA",
    "EDO",
    "AKWAIBOM",
    "IMO",
}


def _expand_crop(crop: np.ndarray, margin_ratio: float = 0.18) -> np.ndarray:
    """Add a small border around the detected plate so OCR sees the full plate."""
    if crop is None or crop.size == 0:
        return crop

    height, width = crop.shape[:2]
    pad_x = max(8, int(width * margin_ratio))
    pad_y = max(8, int(height * margin_ratio))
    bordered = cv2.copyMakeBorder(
        crop,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REPLICATE,
    )

    # A smaller crop can be unreadably tiny at inference time, so enlarge it
    # without distorting the plate area too aggressively.
    scale = max(2.0, 512.0 / max(width, height))
    target_w = max(128, int(round(width * scale)))
    target_h = max(128, int(round(height * scale)))
    return cv2.resize(bordered, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Generate a small bank of OCR-friendly plate variants."""
    if crop is None or crop.size == 0:
        return []

    crop = _expand_crop(crop)
    height, width = crop.shape[:2]
    if width == 0 or height == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 24 or gray.shape[1] < 24:
        gray = cv2.resize(gray, (max(24, gray.shape[1] * 2), max(24, gray.shape[0] * 2)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Strong contrast enhancement while keeping the character strokes.
    blur = cv2.medianBlur(eq, 3)
    sharpen = cv2.GaussianBlur(blur, (0, 0), 1.5)
    sharpen = cv2.addWeighted(blur, 1.6, sharpen, -0.5, 0)

    _, otsu = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)

    # If the background is mostly bright, invert the threshold for better text
    # separation. This matters on reflective plates and dark car bodies.
    variants = [gray, eq, blur, sharpen, otsu, inv]

    # Keep only valid 2D arrays to avoid invalid OCR input.
    cleaned = []
    seen = set()
    for variant in variants:
        if variant is None or variant.ndim != 2:
            continue
        key = (variant.shape[0], variant.shape[1], variant.dtype, variant.mean())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(variant)
    return cleaned


def _candidate_texts_from_easyocr(variant: np.ndarray) -> list[tuple[str, float]]:
    if _easyocr_reader is None:
        return []
    try:
        results = _easyocr_reader.readtext(
            variant,
            detail=1,
            allowlist=_PLATE_CHARS,
            paragraph=False,
        )
    except TypeError:
        try:
            results = _easyocr_reader.readtext(
                variant,
                detail=1,
                allowlist=_PLATE_CHARS,
            )
        except Exception:
            return []
    except Exception:
        return []

    texts: list[tuple[str, float]] = []
    for item in results or []:
        try:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            conf = float(item[2])
        except (TypeError, ValueError):
            continue
        cleaned = _normalise_plate_text(text)
        if cleaned:
            texts.append((cleaned, conf))
    return texts


def _candidate_texts_from_paddleocr(variant: np.ndarray) -> list[tuple[str, float]]:
    if _paddleocr_reader is None:
        return []
    try:
        results = _paddleocr_reader.ocr(variant, cls=True)
    except Exception:
        return []

    texts: list[tuple[str, float]] = []
    if not results or not results[0]:
        return texts

    for item in results[0]:
        if not item or len(item) < 2:
            continue
        try:
            box, (text, conf) = item
        except (ValueError, TypeError):
            continue
        cleaned = _normalise_plate_text(str(text))
        if cleaned:
            texts.append((cleaned, float(conf)))
    return texts


def _score_plate_candidate(text: str, conf: float) -> float:
    """Prefer OCR results that look like real plate strings, not random text."""
    candidate = text.strip().upper()
    if not candidate:
        return -1.0
    if len(candidate) < 3:
        return -1.0
    if re.search(r"[^A-Z0-9]", candidate):
        return -1.0

    letters = sum(ch.isalpha() for ch in candidate)
    digits = sum(ch.isdigit() for ch in candidate)
    if letters == 0 or digits == 0:
        return conf * 0.4

    # Typical plates contain a letter cluster and a numeric cluster. Penalize
    # purely alphabetic or purely numeric OCR outputs.
    if candidate[0].isdigit() and candidate[-1].isalpha():
        return conf * 0.7

    return conf


def _best_plate_result(candidates: list[tuple[str, float]]) -> tuple[str, float] | None:
    if not candidates:
        return None

    valid = []
    for text, conf in candidates:
        validated = _validated_candidate(text, conf)
        if validated is not None:
            valid.append(validated)

    if not valid:
        return None

    best = None
    best_score = -1.0
    for text, conf in valid:
        score = _score_plate_candidate(text, conf)
        if score > best_score:
            best = (text, conf)
            best_score = score

    return best


def _init_easyocr():
    global _OCR_ENGINE, _easyocr_reader
    try:
        import easyocr

        _easyocr_reader = easyocr.Reader(
            ["en"], gpu=(settings.DEVICE != "cpu"), verbose=False
        )
        _OCR_ENGINE = "easyocr"
        log.info("plate OCR ready: EasyOCR on %s", settings.DEVICE)
    except ImportError:
        log.debug("easyocr not installed")
    except Exception:
        log.exception("easyocr initialisation failed")


def _init_paddleocr():
    global _OCR_ENGINE, _paddleocr_reader
    try:
        from paddleocr import PaddleOCR

        _paddleocr_reader = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False, use_gpu=(settings.DEVICE != "cpu")
        )
        _OCR_ENGINE = "paddleocr"
        log.info("plate OCR ready: PaddleOCR on %s", settings.DEVICE)
    except ImportError:
        log.debug("paddleocr not installed")
    except Exception:
        log.exception("paddleocr initialisation failed")


def _lazy_init() -> bool:
    """Load the OCR engine on first use. Returns True if one is available."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return True
    _init_easyocr()
    if _OCR_ENGINE is not None:
        return True
    _init_paddleocr()
    if _OCR_ENGINE is None:
        log.warning(
            "no OCR engine available - install easyocr or paddleocr to read plates"
        )
        return False
    return True


def read_plate(crop: np.ndarray) -> tuple[str, float] | None:
    """
    Extract text from a license plate crop.

    Args:
        crop: BGR image, ideally from `PlateDetector.detect()`.

    Returns:
        `(text, confidence)` or `None` if OCR failed or found nothing readable.
        Text is normalised (upper case, no spaces) for database matching.
    """
    if crop is None or crop.size == 0:
        return None
    if not _lazy_init():
        return None

    try:
        candidates: list[tuple[str, float]] = []
        for variant in _ocr_variants(crop):
            if _OCR_ENGINE == "easyocr":
                candidates.extend(_candidate_texts_from_easyocr(variant))
            elif _OCR_ENGINE == "paddleocr":
                candidates.extend(_candidate_texts_from_paddleocr(variant))

        if not candidates:
            return None

        best = _best_plate_result(candidates)
        if best is None:
            return None

        text, conf = best
        if len(text) < 3:
            return None
        return text, float(conf)

    except Exception:
        log.exception("plate OCR failed on crop shape %s", crop.shape)
        return None


def _normalise_plate_text(raw: str) -> str:
    """
    Upper case + alphanumeric only + common OCR error correction.

    Real plates are alphanumeric with optional dashes, but OCR often returns
    punctuation noise.
    """
    text = str(raw or "").upper()
    text = "".join(ch for ch in text if ch.isalnum() or ch in "-")
    text = text.replace(" ", "").replace("_", "").replace("-", "")

    if not text:
        return ""

    # OCR sometimes includes the state banner text or a leading "M" from the
    # Nigeria flag strip. Remove obvious state words first, then try to recover
    # the plate core from the remaining alphanumeric stream.
    for token in _NIGERIA_STATE_NOISE:
        text = text.replace(token, "")
    text = _extract_plate_core(text)

    # Keep OCR output as close to the actual plate string as possible.
    # Over-aggressive character substitution (e.g. converting 1 -> I) makes
    # later exact/fuzzy matching against the registered plate much worse.
    # Minor OCR confusion is handled by fuzzy matching against the database.
    return text


def _extract_plate_core(text: str) -> str:
    """Extract the most plate-like core from noisy OCR output."""
    if not text:
        return ""

    patterns = (
        r"[A-Z]{3}\d{3}[A-Z]{2,3}",
        r"[A-Z]{2}\d{3}[A-Z]{2,3}",
        r"[A-Z]{1,3}\d{3,4}[A-Z]{1,3}",
    )

    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    if not matches:
        return text

    # Prefer candidates closest to the common Nigerian private format length
    # (8 chars: ABC123DE), then longer exact-format matches.
    return sorted(
        matches,
        key=lambda value: (abs(len(value) - 8), -len(value)),
    )[0]


def fuzzy_match_plate(ocr_text: str, registered: str, threshold: float = 0.80) -> bool:
    """
    Tolerate minor OCR errors when matching against the database.
    """
    import difflib

    ocr_norm = Vehicle.normalise_plate(ocr_text)
    reg_norm = Vehicle.normalise_plate(registered)
    if ocr_norm == reg_norm:
        return True
    ratio = difflib.SequenceMatcher(None, ocr_norm, reg_norm).ratio()
    return ratio >= threshold


def _is_plate_like(text: str) -> bool:
    """Heuristic validation tuned for real-world plate strings.

    Local plates vary by country, but the usual shape is still a short mixed
    alphanumeric string with at least one letter and one digit, and a length in
    the range typical of vehicle plates. This gate is intentionally conservative:
    it rejects random OCR garbage before the database lookup stage.
    """
    if not text:
        return False

    plate = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(plate) < 3 or len(plate) > 12:
        return False
    if plate.isdigit() or plate.isalpha():
        return False
    if not re.search(r"[A-Z0-9]", plate):
        return False

    letters = sum(ch.isalpha() for ch in plate)
    digits = sum(ch.isdigit() for ch in plate)
    if letters == 0 or digits == 0:
        return False

    return True


def _validated_candidate(text: str, conf: float) -> tuple[str, float] | None:
    cleaned = _normalise_plate_text(text)
    if not cleaned:
        return None
    if not _is_plate_like(cleaned):
        return None
    return cleaned, float(conf)
