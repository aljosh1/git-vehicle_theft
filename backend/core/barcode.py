"""
Owner identity codes and their barcodes.

Every registered user carries a **unique owner code** - a short, human-readable
identifier such as `VTD-4F9A-2C71`. The code is the join key between the three
ways a person can be identified at a gate:

* their face (enrolled embeddings, matched by cosine similarity),
* their membership card (the barcode printed on it),
* and the account itself.

Why a code at all
-----------------
Face recognition is probabilistic: a bad angle, a helmet, low light and the
matcher returns UNKNOWN. A barcode is a deterministic read - the guard either
scans the card or does not. Pairing the two is what makes the system usable in
practice, and the code is what lets an incident be attributed to a named person
even when the face was never readable.

Encoding
--------
Both symbologies are produced by OpenCV itself, so there is no extra dependency
and no native library to install:

* **Code 128** - the linear barcode on the card. Dense, scannable by any phone
  camera or hand-held reader, and it encodes the full alphanumeric code.
* **QR** - the same payload as a matrix code, for scanners that prefer it and
  for printing at small sizes.

Collisions are resolved by re-drawing with a counter, not by locking the table,
so a code is never issued twice even under concurrent registration.
"""

from __future__ import annotations

import secrets

import cv2
import numpy as np

from backend.utils.logger import get_logger

log = get_logger(__name__)

# Deliberately excludes look-alike glyphs (0/O, 1/I/L) so a code read off a
# scratched card is still unambiguous when an operator types it in.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_GROUP_SIZE = 4
_CODE_GROUPS = 2
_PREFIX = "VTD"

# A Code 128 symbol is drawn narrow (1 px) and widened to ~3 px per module at
# print time; a card is ~85 mm wide, so 60 modules is comfortably inside it.
_QR_SIZE = 360
_QUIET_ZONE = 24

# =============================================================================
#  Code 128 symbology (Code Set B)
# =============================================================================
# Each row is the bar/space module widths of one of the 107 Code 128 values,
# alternating bar-then-space. Set B maps printable ASCII 32-126 onto values
# 0-94; 103-106 are the start codes and 106 is the stop.
_CODE128_PATTERNS = (
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312",
    "132212", "221213", "221312", "231212", "112232", "122132", "122231", "113222",
    "123122", "123221", "223211", "221132", "221231", "213212", "223112", "312131",
    "311222", "321122", "321221", "312212", "322112", "322211", "212123", "212321",
    "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121",
    "313121", "211331", "231131", "213113", "213311", "213131", "311123", "311321",
    "331121", "312113", "312311", "332111", "314111", "221411", "431111", "111224",
    "111422", "121124", "121421", "141122", "141221", "112214", "112412", "122114",
    "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112",
    "421211", "212141", "214121", "412121", "111143", "111341", "131141", "114113",
    "114311", "411113", "411311", "113141", "114131", "311141", "411131", "211412",
    "211214", "211232", "2331112",
)
_CODE128_STOP = "2331112"
_START_B = 104
_CODE_B_FIRST = 32
_CODE_B_LAST = 126
# How far a measured run may deviate from a pattern's ideal width, in modules.
# Half a module of slack absorbs a resample or a slight rotation without letting
# genuinely different symbols collide.
_PATTERN_TOLERANCE = 0.5


class BarcodeError(RuntimeError):
    """Raised when a payload cannot be encoded - a malformed owner code."""


# =============================================================================
#  Code generation
# =============================================================================
def generate_owner_code() -> str:
    """
    Produce a new, unguessable owner code.

    Uses `secrets` rather than `random`: these codes are an access token for a
    physical gate, so the sequence must not be predictable from the previously
    issued code.
    """
    groups = [
        "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_GROUP_SIZE))
        for _ in range(_CODE_GROUPS)
    ]
    return f"{_PREFIX}-" + "-".join(groups)


def normalise_owner_code(value: str | None) -> str | None:
    """
    Canonicalise a scanned or typed code for comparison.

    Scanners are inconsistent about case and separators, and some strip the
    `VTD` prefix as a "type" indicator. Both are folded away here so
    `vtd 4f9a 2c71`, `VTD-4F9A-2C71` and `4F9A2C71` all resolve to the same
    owner. Returns None for anything that is not a well-formed code.
    """
    if not value:
        return None
    body = value.strip().upper()
    if body.startswith(_PREFIX):
        body = body[len(_PREFIX) :]
    body = body.replace("-", "").replace(" ", "").replace("_", "")
    expected = _CODE_GROUPS * _CODE_GROUP_SIZE
    if len(body) != expected or not all(c in _CODE_ALPHABET for c in body):
        return None
    return f"{_PREFIX}-" + "-".join(
        body[i : i + _CODE_GROUP_SIZE] for i in range(0, expected, _CODE_GROUP_SIZE)
    )


def is_valid_owner_code(value: str | None) -> bool:
    return normalise_owner_code(value) is not None


# =============================================================================
#  Rendering
# =============================================================================
def encode_code128(payload: str) -> np.ndarray:
    """
    Render `payload` as a Code 128 (Code Set B) barcode.

    OpenCV 4.10's Python bindings expose a QR *encoder* but no Code 128 one, so
    the symbol is emitted directly from the symbology table: each character is
    a value in the range 32..126, Code Set B encodes it as `ord(c) - 32`, and
    the six-digit row for that value gives the bar/space widths. 11 modules of
    quiet space and the standard stop pattern complete the symbol.

    Returns a BGR image. Raises `BarcodeError` if the payload contains a
    character Code Set B cannot represent.
    """
    if not payload:
        raise BarcodeError("cannot encode an empty payload")

    values: list[int] = [_START_B]
    for char in payload:
        code = ord(char)
        if not (_CODE_B_FIRST <= code <= _CODE_B_LAST):
            raise BarcodeError(
                f"character {char!r} (U+{code:04X}) is not representable in "
                "Code 128 set B; owner codes are alphanumeric by design"
            )
        values.append(code - 32)

    # Mod-103 weighted checksum: the start value plus every character weighted
    # by its 1-based position.
    checksum = (_START_B + sum(i * v for i, v in enumerate(values[1:], start=1))) % 103
    values.append(checksum)

    widths: list[int] = []
    for value in values:
        widths.extend(int(digit) for digit in _CODE128_PATTERNS[value])
    widths.extend(int(digit) for digit in _CODE128_STOP)

    # One module = 3 px, which survives printing and downscaling.
    module = 3
    bar_width = sum(widths) * module
    canvas_width = bar_width + 2 * _QUIET_ZONE * module
    canvas_height = 160

    image = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    x = _QUIET_ZONE * module
    is_bar = True
    for width in widths:
        if is_bar:
            image[:, x : x + width * module] = 0
        x += width * module
        is_bar = not is_bar
    return image



def encode_qr(payload: str) -> np.ndarray:
    """Render `payload` as a QR code."""
    if not hasattr(cv2, "QRCodeEncoder"):
        raise BarcodeError(
            "this OpenCV build has no QR encoder; upgrade opencv-python"
        )
    try:
        params = cv2.QRCodeEncoder.Params()
        # `version = 0` means "smallest that fits". The correction-level and
        # mode enums are not exposed by every OpenCV Python binding, so they are
        # left at their defaults rather than referenced by name.
        params.version = 0
        image = cv2.QRCodeEncoder.create(params).encode(payload)
    except cv2.error as exc:
        raise BarcodeError(f"QR encoding failed: {exc}") from exc
    except (AttributeError, TypeError) as exc:
        raise BarcodeError(f"this OpenCV build cannot encode QR codes: {exc}") from exc

    if image is None or image.size == 0:
        raise BarcodeError("QR encoding produced an empty symbol")
    return _fit_to_canvas(image, (_QR_SIZE, _QR_SIZE))


def render_card(
    owner_code: str,
    *,
    name: str | None = None,
    size: tuple[int, int] = (520, 620),
) -> np.ndarray:
    """
    Build a printable membership card: header, QR, Code 128 and the code in text.

    The human-readable code is printed *and* encoded, because the barcode is the
    fast path and the text is the fallback when the card is too scratched to
    scan.
    """
    normalised = normalise_owner_code(owner_code)
    if normalised is None:
        raise BarcodeError(f"'{owner_code}' is not a valid owner code")

    width, height = size
    card = np.full((height, width, 3), 245, dtype=np.uint8)
    card = cv2.rectangle(card, (0, 0), (width - 1, height - 1), (210, 214, 220), 3)

    # OpenCV colours are BGR, not RGB. Getting this backwards produces an
    # orange header where a blue one was intended.
    accent = (235, 99, 37)      # BGR blue
    header_h = 96
    card[:header_h] = accent
    label = (255, 255, 255)
    cv2.putText(
        card, "VEHICLE THEFT DETECTION", (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, label, 2, cv2.LINE_AA,
    )
    subtitle = name or "OWNER CREDENTIAL"
    cv2.putText(
        card, subtitle[:34], (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (222, 234, 250), 1, cv2.LINE_AA,
    )

    # QR, left-aligned beside a stacked identity block on the right.
    qr_size = 210
    qr = _resize_keeping_aspect(encode_qr(normalised), qr_size, qr_size)
    qx, qy = 26, header_h + 26
    card[qy : qy + qr.shape[0], qx : qx + qr.shape[1]] = qr
    cv2.rectangle(
        card, (qx - 1, qy - 1), (qx + qr.shape[1], qy + qr.shape[0]), (200, 205, 212), 1
    )

    # Identity block to the right of the QR.
    info_x = qx + qr.shape[1] + 28
    cv2.putText(
        card, "MEMBER", (info_x, qy + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 138, 150), 1, cv2.LINE_AA,
    )
    cv2.putText(
        card, subtitle[:22], (info_x, qy + 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 65, 81), 1, cv2.LINE_AA,
    )
    cv2.putText(
        card, "Owner credential", (info_x, qy + 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (156, 163, 175), 1, cv2.LINE_AA,
    )

    # Code 128 across the lower half.
    barcode = _resize_keeping_aspect(encode_code128(normalised), width - 48, 132)
    bx = 24
    by = qy + qr.shape[0] + 34
    card[by : by + barcode.shape[0], bx : bx + barcode.shape[1]] = barcode

    # The code in text: the fallback when a card is too damaged to scan.
    text_scale = 0.95
    (tw, _), _ = cv2.getTextSize(normalised, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
    cv2.putText(
        card, normalised, ((width - tw) // 2, by + barcode.shape[0] + 46),
        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (17, 24, 39), 2, cv2.LINE_AA,
    )
    footer = "Scan at the gate to verify ownership"
    (fw, _), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        card, footer, ((width - fw) // 2, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (107, 114, 128), 1, cv2.LINE_AA,
    )
    return card


# =============================================================================
#  Reading
# =============================================================================
def decode_barcode(image: np.ndarray) -> str | None:
    """
    Read the payload from an image containing a Code 128 or QR symbol.

    Tries, in order:

    1. the built-in Code 128 reader (works on the linear symbol and needs no
       OpenCV decoder support);
    2. OpenCV's multi-format detector;
    3. OpenCV's QR detector.

    OpenCV 4.10's Python bindings decode QR reliably but do not decode Code 128,
    which is why step 1 exists rather than being an optimisation. Returns the raw
    text, or None when nothing decodes.
    """
    if image is None or image.size == 0:
        return None

    for attempt in (
        _decode_code128,
        _decode_with_barcode_detector,
        _decode_with_qr_detector,
    ):
        try:
            value = attempt(image)
        except cv2.error:
            log.debug("barcode decoder raised; trying the next strategy")
            continue
        except Exception:
            log.exception("unexpected barcode decoding failure")
            continue
        if value:
            return value.strip()
    return None


# =============================================================================
#  Code 128 reader
# =============================================================================
# Each of the 107 symbols is 11 modules wide (the stop is 13) and always has
# exactly three bars and three spaces, alternating. Indexing the table by the
# six module widths and reversing the lookup is all the matching needed.
_CODE128_LOOKUP: dict[tuple[int, ...], int] = {
    tuple(int(d) for d in pattern): index
    for index, pattern in enumerate(_CODE128_PATTERNS)
}


def _binarise(image: np.ndarray) -> np.ndarray:
    """
    Reduce an arbitrary photo to a single row of black/white module samples.

    A camera image of a card has lighting gradients, colour cast and perspective
    skew, so a fixed threshold would fail on half the scans. Instead every row is
    averaged (killing vertical noise from a photo of a printed surface), the row
    with the best local contrast is chosen - that is the row crossing the
    barcode - and it is thresholded at its own midpoint, which is invariant to
    overall brightness.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)

    row_mean = gray.mean(axis=1)
    # Local contrast proxy: how much each row deviates from the mean of its
    # neighbourhood. A row crossing bars and spaces has the highest variance.
    half = max(1, gray.shape[0] // 32)
    best_row, best_score = None, -1.0
    for y in range(0, gray.shape[0] - half, max(1, half // 2)):
        window = row_mean[y : y + 2 * half]
        score = float(np.abs(window - window.mean()).max())
        if score > best_score:
            best_score, best_row = score, y + half

    if best_row is None:
        profile = row_mean
    else:
        # Average a thin band around the best row rather than reading a single
        # line. Sensor noise makes any one row unreliable - a single flipped
        # pixel splits a bar into two runs and destroys the decode - while a
        # band of several rows is immune to it.
        top = max(0, best_row - half // 2)
        bottom = min(gray.shape[0], best_row + max(1, half // 2))
        profile = gray[top:bottom].mean(axis=0)

    low, high = float(profile.min()), float(profile.max())
    if high - low < 24.0:      # no usable contrast - treat as "no symbol"
        return np.zeros(profile.shape[0], dtype=bool)
    threshold = (low + high) / 2.0
    return profile < threshold



def _decode_code128(image: np.ndarray) -> str | None:
    """
    Decode a Code 128 Code Set B symbol by run-length analysis.

    Steps: binarise, measure run lengths, convert pixel runs to module units
    using the known 11-module symbol width as the scale reference, then look up
    each 6-run group in the symbology table and verify the checksum.
    """
    binary = _binarise(image)
    if binary.size == 0 or not binary.any():
        return None

    runs: list[int] = []
    current = bool(binary[0])
    length = 0
    for pixel in binary:
        if bool(pixel) is current:
            length += 1
        else:
            runs.append(length)
            current = bool(pixel)
            length = 1
    runs.append(length)

    # The symbol starts and ends with a bar, so an even run count is a strong
    # hint that we sampled the whole thing; trim to it if not.
    if len(runs) % 2 == 1:
        runs = runs[:-1]
    if len(runs) < 8:
        return None

    # Module width in pixels. The GCD of the run lengths is exact for a pristine
    # render but collapses on a resampled or photographed card, where a 3-px
    # module lands on 1 or 2 px depending on where the boundary falls and no
    # integer divisor fits.
    #
    # Rather than guessing, use the symbology itself: the start pattern is 11
    # modules, so the start run alone pins the scale to a narrow range. Each
    # candidate is then tested and the first checksum-valid symbol wins - mod-103
    # is what makes a wrong candidate harmless, because it fails to decode and
    # the sweep moves on rather than returning a plausible-looking misread.
    # A clean render decodes at the estimated scale. A photographed card has
    # speckle in the quiet zones and pinholes in the bars, which shatters a run
    # length into pieces; a single closing pass re-bridges gaps shorter than the
    # speckle while leaving a real bar edge untouched. Morphology is therefore
    # a *fallback* for a failed attempt, never the default, because on a clean
    # image it erodes the one-module bars and costs more than it gains.
    #
    # The scale is re-estimated for the despeckled sequence: speckle inflates
    # the run count, which drags the median down toward sub-module values that
    # no candidate can satisfy.
    for smoothing in (False, True):
        candidate_runs = _despeckle(runs) if smoothing else runs
        if len(candidate_runs) < 8:
            continue
        scale = _estimate_module_width(candidate_runs)
        candidates: set[float] = {scale}
        for delta in (-1, -0.5, 0.5, 1, -1.5, 1.5, -2, 2):
            if scale + delta >= 0.5:
                candidates.add(scale + delta)
        for divisor in sorted(candidates):
            decoded = _decode_modules(
                [width / divisor for width in candidate_runs]
            )
            if decoded is not None:
                return decoded
    return None


def _despeckle(runs: list[int]) -> list[int]:
    """
    Merge neighbouring runs that are implausibly short for the symbol's scale.

    A 1-3 px speckle splits a run into `2, 1, 1, 1, 9` instead of `2, 12`. Runs
    shorter than a module and shorter than half their neighbour are absorbed
    into the longer side, which restores the original boundaries without moving
    a genuine edge.
    """
    if not runs:
        return runs
    scale = _estimate_module_width(runs)
    merged = list(runs)
    changed = True
    while changed and len(merged) >= 8:
        changed = False
        for index in range(1, len(merged) - 1):
            width = merged[index]
            if width >= scale or width * 2 >= max(merged[index - 1], merged[index + 1]):
                continue
            if merged[index - 1] >= merged[index + 1]:
                merged[index - 1] += width
            else:
                merged[index + 1] += width
            del merged[index]
            changed = True
            break
    return merged



def _estimate_module_width(runs: list[int]) -> float:
    """
    Estimate the pixel width of one barcode module from the run lengths.

    Every Code 128 data symbol is 11 modules spanning exactly 6 runs, so the
    *sum* of those 6 runs is the module width times 11. Taking the median of
    `run[i] + ... + run[i+5]` over all windows therefore gives a scale that
    survives a few mis-measured runs, which a GCD or a mode does not.
    """
    spans = [
        sum(runs[i : i + 6]) / 11.0
        for i in range(0, max(1, len(runs) - 5))
    ]
    if not spans:
        return float(min(runs)) if runs else 1.0
    spans.sort()
    median = spans[len(spans) // 2]
    # Reject an implausible estimate so a degenerate image cannot trigger a
    # pointless sweep of hundreds of candidate scales.
    if median <= 0.05:
        return 1.0
    return median



def _decode_modules(modules: list[float]) -> str | None:
    """
    Turn a run-length sequence in module units into the decoded text.

    The symbol is bracketed by its start pattern (211214) and stop pattern
    (2331112). Locating those two landmarks is far more reliable than guessing
    where the symbol begins, because the quiet zone is an arbitrary number of
    modules wide and trimming by a fixed rule eats real bars.

    Returns None unless a well-formed, checksum-valid Code Set B symbol is found,
    so a wrong module-width candidate is simply rejected.
    """
    # The symbol is bracketed by its start pattern (211214) and stop pattern
    # (2331112).
    start_at = _find_pattern(modules, _CODE128_PATTERNS[_START_B])
    if start_at is None:
        return None
    end_at = _find_pattern(modules, _CODE128_STOP, from_end=True)
    if end_at is None:
        return None

    # Every data symbol is 11 modules / 6 runs, but the stop pattern is 13
    # modules / 7 runs. Walking in 6-run strides to `end_at` therefore lands
    # exactly on the stop pattern's first module - decode only the symbols
    # before it, then validate the stop itself.
    values: list[int] = []
    for index in range(start_at, end_at, 6):
        if index + 6 > len(modules):
            return None
        value = _CODE128_LOOKUP.get(
            tuple(max(1, int(round(width))) for width in modules[index : index + 6])
        )
        if value is None:
            return None
        values.append(value)
    if not values or values[0] != _START_B:
        return None
    if len(values) < 4:      # start + at least 2 characters + checksum
        return None

    payload_values, checksum = values[1:-1], values[-1]
    computed = (_START_B + sum(i * v for i, v in enumerate(payload_values, start=1))) % 103
    if checksum != computed:
        log.debug("Code 128 checksum mismatch: read %s, expected %s", checksum, computed)
        return None

    try:
        return "".join(chr(v + 32) for v in payload_values)
    except ValueError:
        return None



def _find_pattern(
    modules: list[float], pattern: str, *, from_end: bool = False
) -> int | None:
    """
    Locate a known module-width run sequence inside the scanned runs.

    `from_end` searches right-to-left, which is what the stop pattern needs: it
    is guaranteed to be the last symbol in the image, whereas the start pattern
    must be found on a bar boundary scanning forwards.
    """
    target = [int(digit) for digit in pattern]
    span = len(target)

    def matches(start: int) -> bool:
        return all(
            abs(modules[start + i] - target[i]) <= _PATTERN_TOLERANCE
            for i in range(span)
            if start + i < len(modules)
        )

    if from_end:
        for start in range(len(modules) - span, -1, -1):
            if matches(start):
                return start
        return None
    for start in range(0, len(modules) - span + 1):
        if matches(start):
            return start
    return None




def _decode_with_barcode_detector(image: np.ndarray) -> str | None:
    """Multi-format detector: understands Code 128, EAN, and QR."""
    detector = cv2.barcode.BarcodeDetector()
    found = detector.detectAndDecode(image)
    # OpenCV returns (retval, points, payload) on some builds and
    # (points, payload) on others; normalise both shapes.
    if len(found) >= 3:
        return found[2] or None
    if len(found) == 2:
        return found[1] or None
    return None


def _decode_with_qr_detector(image: np.ndarray) -> str | None:
    """QR-only fallback for builds without the multi-format detector."""
    detector = cv2.QRCodeDetector()
    payload, _, _ = detector.detectAndDecode(image)
    return payload or None


# =============================================================================
#  Helpers
# =============================================================================
def _fit_to_canvas(symbol: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Scale a 1-bit symbol up and centre it on a white canvas.

    Encoders return a tiny bitmap; scaling with nearest-neighbour keeps the
    module edges hard, which is what barcode readers need. A quiet zone is added
    because OpenCV's decoder refuses symbols that run to the image border.
    """
    width, height = size
    scale = max(1, min((width - _QUIET_ZONE) // max(symbol.shape[1], 1),
                       (height - _QUIET_ZONE) // max(symbol.shape[0], 1)))
    enlarged = cv2.resize(
        symbol,
        (symbol.shape[1] * scale, symbol.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    if enlarged.ndim == 2:
        enlarged = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    oy = (height - enlarged.shape[0]) // 2
    ox = (width - enlarged.shape[1]) // 2
    canvas[oy : oy + enlarged.shape[0], ox : ox + enlarged.shape[1]] = enlarged
    return canvas


def _resize_keeping_aspect(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Downscale to fit a box without distorting the aspect ratio."""
    scale = min(max_w / image.shape[1], max_h / image.shape[0], 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
