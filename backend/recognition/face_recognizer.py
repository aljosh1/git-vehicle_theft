"""
Face detection and recognition.

Decides whether a person in frame is an **Authorized** vehicle owner or an
**Unauthorized** stranger, which is the single most important input to the
theft-detection logic.

Backend chain
-------------
Four interchangeable backends are tried in order of preference.  Whichever
loads first is used; the rest of the system never learns which one it was:

1. **DeepFace** (`Facenet512`/`ArcFace` + RetinaFace) - the specification's
   first choice.  Requires TensorFlow.
2. **facenet-pytorch** (`InceptionResnetV1` pretrained on VGGFace2 + MTCNN) -
   same FaceNet architecture, PyTorch instead of TensorFlow.
3. **OpenCV YuNet + SFace** (ONNX, via `cv2.FaceDetectorYN` /
   `cv2.FaceRecognizerSF`) - YuNet detects, SFace produces a 128-d embedding.
   SFace reports ~99.6% on LFW, so this is a genuine research-grade path, not a
   consolation prize.  It needs no TensorFlow and no PyTorch, which is what
   makes it the working backend on Python 3.14.
4. **Histogram descriptor** - detection via YuNet, matching via a coarse 4x4
   tonal histogram.  Last resort when the SFace weights are absent; **not**
   research-grade, and the module says so loudly.

`ACTIVE_BACKEND` is reported through `/api/info` so a demonstration never
misrepresents which model produced a result.

Enrolment and inference must agree
----------------------------------
Every embedding - whether it is being written into the gallery at registration
time or compared against it during surveillance - goes through the single
entry point `embed_from_image()`.  That function detects the face itself and
performs landmark alignment, so a registered owner and a live query are
preprocessed identically.  Mixing an unaligned gallery vector with an aligned
query vector pushes genuine same-person pairs below any sane threshold, which
looks exactly like "recognition is broken".

Matching
--------
Embeddings are L2-normalised, so cosine similarity is a plain dot product and
the comparison against the whole gallery is one matrix multiply - fast enough
to run per frame even with hundreds of registered owners.

Thresholds are per-backend.  A cosine cutoff tuned for Facenet512 rejects every
real owner when SFace is the active model; OpenCV's own recommended SFace
cutoff is 0.363, not 0.6+.

Gallery vectors are tagged with the backend that produced them.  Vectors from a
different backend are refused rather than silently compared, because a 512-d
DeepFace gallery and a 128-d SFace query can only ever return UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.config import settings
from backend.database.models import PersonStatus
from backend.utils.logger import get_logger

log = get_logger(__name__)

ACTIVE_BACKEND: str | None = None
_model = None
_detector = None
_EMBEDDING_DIM = 512

# Cosine-similarity cutoffs, per backend. These are not interchangeable.
#   - deepface / facenet: conventional FaceNet-family operating point.
#   - sface: OpenCV's published recommendation for FaceRecognizerSF.
#   - histogram: deliberately high; this path should almost never authorise.
_BACKEND_THRESHOLDS: dict[str, float] = {
    "deepface": 0.60,
    "facenet-pytorch": 0.55,
    "opencv-yunet-sface": 0.363,
    "opencv-yunet-histogram": 0.95,
}


@dataclass(slots=True)
class FaceMatch:
    """Outcome of comparing one detected face against the registered gallery."""

    status: PersonStatus
    owner_id: int | None
    similarity: float
    bbox: tuple[int, int, int, int] | None = None

    @property
    def is_authorized(self) -> bool:
        return self.status is PersonStatus.AUTHORIZED


# =============================================================================
#  Backend initialisation
# =============================================================================
def _init_deepface() -> bool:
    global ACTIVE_BACKEND, _EMBEDDING_DIM
    try:
        from deepface import DeepFace

        # Force the weights to download now rather than mid-surveillance.
        DeepFace.build_model(settings.FACE_MODEL_NAME)

        # Output dimensionality is a property of the model, not of whether its
        # name happens to contain the digits "512". ArcFace is 512-d.
        dims = {
            "Facenet512": 512,
            "ArcFace": 512,
            "VGG-Face": 4096,
            "Facenet": 128,
            "OpenFace": 128,
            "DeepFace": 4096,
            "DeepID": 160,
            "Dlib": 128,
            "SFace": 128,
            "GhostFaceNet": 512,
        }
        _EMBEDDING_DIM = dims.get(settings.FACE_MODEL_NAME, 512)

        ACTIVE_BACKEND = "deepface"
        log.info(
            "face recognition ready: DeepFace/%s + %s (%d-d)",
            settings.FACE_MODEL_NAME,
            settings.FACE_DETECTOR_BACKEND,
            _EMBEDDING_DIM,
        )
        return True
    except ImportError:
        log.debug("deepface not installed")
    except Exception:
        log.exception("deepface initialisation failed")
    return False


def _init_facenet_pytorch() -> bool:
    global ACTIVE_BACKEND, _model, _detector, _EMBEDDING_DIM
    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore[reportMissingImports]

        device = torch.device(
            settings.DEVICE if settings.DEVICE.startswith("cuda") else "cpu"
        )
        _detector = MTCNN(keep_all=True, device=device, post_process=True)
        _model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        ACTIVE_BACKEND = "facenet-pytorch"
        _EMBEDDING_DIM = 512
        log.info("face recognition ready: facenet-pytorch (VGGFace2) on %s", device)
        return True
    except ImportError:
        log.debug("facenet-pytorch not installed")
    except Exception:
        log.exception("facenet-pytorch initialisation failed")
    return False


def _init_opencv_dnn() -> bool:
    """
    OpenCV's ONNX face stack: YuNet for detection, SFace for embeddings.

    This is the backend that actually runs on Python 3.14, where neither
    TensorFlow nor an installable facenet-pytorch exists.  Both models are
    small ONNX files under `models/` (see `models/README.md`); YuNet is 227 KB
    and SFace 37 MB, so an offline demonstration is realistic.
    """
    global ACTIVE_BACKEND, _model, _detector, _EMBEDDING_DIM
    try:
        if not hasattr(cv2, "FaceDetectorYN"):
            log.debug("this OpenCV build has no FaceDetectorYN")
            return False

        yunet_path = settings.models_dir / "face_detection_yunet.onnx"
        sface_path = settings.models_dir / "face_recognition_sface.onnx"

        if not yunet_path.exists():
            log.debug("YuNet weights not found at %s", yunet_path)
            return False

        # Input size is re-set per frame in detect_faces(); (320, 320) is just
        # the construction-time placeholder.
        _detector = cv2.FaceDetectorYN.create(
            model=str(yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=50,
        )

        if sface_path.exists():
            _model = cv2.FaceRecognizerSF.create(str(sface_path), "")
            ACTIVE_BACKEND = "opencv-yunet-sface"
            _EMBEDDING_DIM = 128           # SFace produces 128-d features
            log.info(
                "face recognition ready: OpenCV YuNet (detect) + SFace "
                "(128-d embeddings, ~99.6%% LFW), cosine cutoff %.3f",
                _BACKEND_THRESHOLDS["opencv-yunet-sface"],
            )
        else:
            _model = None
            ACTIVE_BACKEND = "opencv-yunet-histogram"
            _EMBEDDING_DIM = 256
            log.warning(
                "YuNet loaded but SFace weights are missing at %s - falling "
                "back to a coarse histogram descriptor. Detection is sound, "
                "but do NOT quote recognition accuracy from this path; "
                "download the SFace ONNX model (see models/README.md).",
                sface_path,
            )
        return True

    except Exception:
        log.exception("OpenCV DNN face backend initialisation failed")
    return False


def init_backend() -> str | None:
    """Load the best available backend. Idempotent; safe to call repeatedly."""
    global ACTIVE_BACKEND
    if ACTIVE_BACKEND is not None:
        return ACTIVE_BACKEND
    for initialise in (
        _init_deepface,
        _init_facenet_pytorch,
        _init_opencv_dnn,
    ):
        if initialise():
            return ACTIVE_BACKEND
    log.error("no face recognition backend available - all persons will be UNKNOWN")
    return None


def embedding_dim() -> int:
    init_backend()
    return _EMBEDDING_DIM


def match_threshold() -> float:
    """
    The cosine cutoff appropriate to the *active* backend.

    A single global number cannot serve all four backends: 0.60 is a reasonable
    Facenet512 operating point and simultaneously an impossible SFace one.  An
    explicit `settings.FACE_MATCH_THRESHOLD` still wins if you set it, but the
    default now follows the model.
    """
    init_backend()
    override = getattr(settings, "FACE_MATCH_THRESHOLD", None)
    if override is not None:
        return float(override)
    return _BACKEND_THRESHOLDS.get(str(ACTIVE_BACKEND), 0.5)


# =============================================================================
#  Detection
# =============================================================================
def _detect_rows(image: np.ndarray):
    """
    YuNet's full per-face rows, including the five facial landmarks.

    SFace needs those landmarks to align a crop before embedding it, and
    alignment is worth several points of accuracy - so this returns the raw
    array rather than just boxes.  Coordinates are relative to `image`.
    """
    if ACTIVE_BACKEND is None or not str(ACTIVE_BACKEND).startswith("opencv-yunet"):
        return None
    if image is None or image.size == 0:
        return None
    try:
        height, width = image.shape[:2]
        # The detector must be told the exact image size, or it silently
        # returns garbage coordinates.
        _detector.setInputSize((width, height))
        _, faces = _detector.detect(image)
        return faces
    except Exception:
        log.exception("raw face detection failed")
        return None


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Locate faces in a frame.

    Returns:
        Bounding boxes `(x1, y1, x2, y2)` in absolute pixels. Empty if no face
        is visible - a very common and entirely normal result for CCTV angles.
    """
    if frame is None or frame.size == 0:
        return []
    if init_backend() is None:
        return []

    try:
        if ACTIVE_BACKEND == "deepface":
            from deepface import DeepFace

            faces = DeepFace.extract_faces(
                frame,
                detector_backend=settings.FACE_DETECTOR_BACKEND,
                enforce_detection=False,
            )
            boxes = []
            for face in faces:
                area = face.get("facial_area", {})
                x, y = int(area.get("x", 0)), int(area.get("y", 0))
                w, h = int(area.get("w", 0)), int(area.get("h", 0))
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))
            return boxes

        if ACTIVE_BACKEND == "facenet-pytorch":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = _detector.detect(rgb)
            if boxes is None:
                return []
            return [tuple(int(v) for v in box) for box in boxes]

        rows = _detect_rows(frame)
        if rows is None:
            return []

        boxes = []
        for row in rows:
            # YuNet rows are [x, y, w, h, 5 landmark pairs..., score].
            x, y, w, h = (int(v) for v in row[:4])
            if w > 0 and h > 0:
                boxes.append((max(0, x), max(0, y), x + w, y + h))
        return boxes

    except Exception:
        log.exception("face detection failed")
        return []


def _shift_row(row: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Translate a YuNet row from sub-image coordinates into frame coordinates.

    The row layout is [x, y, w, h, lx0, ly0, ... lx4, ly4, score], so every
    even index up to 13 is an x and every odd index up to 13 is a y, except
    indices 2 and 3 which are width and height and must not be moved.

    This is the fix for the original bug: a row detected inside a cropped
    person box was handed to `alignCrop` together with the *full* frame, so
    SFace warped a patch of background and produced a vector that matched
    nobody.
    """
    shifted = np.array(row, dtype=np.float32).copy()
    shifted[0] += dx
    shifted[1] += dy
    for i in range(4, 14, 2):
        shifted[i] += dx
        shifted[i + 1] += dy
    return shifted


# =============================================================================
#  Embedding
# =============================================================================
def embed_from_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    The single embedding entry point. Use this for enrolment **and** inference.

    Detects a face, aligns it when the backend supports landmarks, and returns
    an L2-normalised vector.  Because registration and surveillance both come
    through here, a gallery vector and a live query are preprocessed the same
    way - which is what makes their cosine similarity meaningful.

    Args:
        image: full BGR image (a video frame, or an uploaded registration
            photo).
        bbox: optional region of interest, e.g. a YOLO person box. The face
            search is restricted to it, which is faster and less error-prone,
            but the returned box is always in `image` coordinates.

    Returns:
        `(embedding, face_bbox)`. `(None, None)` when no face was found;
        `(None, bbox)` when a face was found but embedding failed.
    """
    if image is None or image.size == 0:
        return None, None
    if init_backend() is None:
        return None, None

    region = image
    offset_x = offset_y = 0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        region = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    # ---- YuNet backends: detect in the region, then align in frame space ----
    rows = _detect_rows(region)
    if rows is not None and len(rows) > 0:
        best = max(rows, key=lambda r: float(r[2]) * float(r[3]))  # w * h
        fx, fy, fw, fh = (int(v) for v in best[:4])
        face_box = (
            fx + offset_x,
            fy + offset_y,
            fx + fw + offset_x,
            fy + fh + offset_y,
        )

        if ACTIVE_BACKEND == "opencv-yunet-sface":
            # Align against the full image, with the row translated to match.
            # Using the full image gives alignCrop room to warp near the edges
            # of the person box instead of clipping the jaw or forehead.
            aligned = _model.alignCrop(image, _shift_row(best, offset_x, offset_y))
            vector = _model.feature(aligned).flatten().astype(np.float32)
            return _l2_normalise(vector), face_box

        crop = region[max(0, fy) : fy + fh, max(0, fx) : fx + fw]
        if crop.size == 0:
            return None, face_box
        vector = _histogram_descriptor(crop)
        if vector is None:
            return None, face_box
        return _l2_normalise(vector), face_box

    # ---- DeepFace / facenet-pytorch: box-only detection ----
    faces = detect_faces(region)
    if not faces:
        return None, None

    fx1, fy1, fx2, fy2 = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    face_box = (fx1 + offset_x, fy1 + offset_y, fx2 + offset_x, fy2 + offset_y)
    crop = region[max(0, fy1) : fy2, max(0, fx1) : fx2]
    if crop.size == 0:
        return None, face_box

    return _embed_crop(crop), face_box


def _embed_crop(face_image: np.ndarray) -> np.ndarray | None:
    """Embed an already-cropped face. Internal; callers should use
    `embed_from_image()` so that detection and alignment stay consistent."""
    if face_image is None or face_image.size == 0:
        return None
    if init_backend() is None:
        return None

    try:
        if ACTIVE_BACKEND == "deepface":
            from deepface import DeepFace

            result = DeepFace.represent(
                face_image,
                model_name=settings.FACE_MODEL_NAME,
                detector_backend="skip",   # already cropped
                enforce_detection=False,
            )
            if not result:
                return None
            vector = np.asarray(result[0]["embedding"], dtype=np.float32)

        elif ACTIVE_BACKEND == "facenet-pytorch":
            import torch

            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (160, 160)).astype(np.float32)
            # facenet-pytorch expects inputs standardised to roughly [-1, 1].
            normalised = (resized - 127.5) / 128.0
            tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(next(_model.parameters()).device)
            with torch.no_grad():
                vector = _model(tensor).cpu().numpy()[0].astype(np.float32)

        elif ACTIVE_BACKEND == "opencv-yunet-sface":
            # No landmarks available. SFace expects an aligned 112x112; a bare
            # resize costs real accuracy, so this path is a fallback only.
            log.debug("SFace embedding without alignment - accuracy will suffer")
            aligned = cv2.resize(face_image, (112, 112))
            vector = _model.feature(aligned).flatten().astype(np.float32)

        else:
            vector = _histogram_descriptor(face_image)
            if vector is None:
                return None

        return _l2_normalise(vector)

    except Exception:
        log.exception("embedding extraction failed")
        return None


def extract_embedding(
    face_image: np.ndarray, face_row: np.ndarray | None = None
) -> np.ndarray | None:
    """
    Backwards-compatible wrapper kept so existing callers keep working.

    `face_row` must be in the coordinate space of `face_image`. New code should
    call `embed_from_image()` instead, which handles detection and alignment
    itself and cannot get that relationship wrong.
    """
    if face_row is not None and ACTIVE_BACKEND == "opencv-yunet-sface":
        try:
            aligned = _model.alignCrop(face_image, np.asarray(face_row, np.float32))
            return _l2_normalise(_model.feature(aligned).flatten().astype(np.float32))
        except Exception:
            log.exception("aligned embedding failed")
            return None
    return _embed_crop(face_image)


def _histogram_descriptor(face_image: np.ndarray) -> np.ndarray | None:
    """
    Weak hand-rolled descriptor for the no-recognition-model fallback.

    A 4x4 grid of local histograms, which captures coarse tonal layout and
    nothing more.  It will confuse similar-looking people under similar
    lighting.  Present only so the end-to-end flow is demonstrable; never
    quote accuracy numbers produced by this path.
    """
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        gray = cv2.equalizeHist(gray)

        cells = []
        for row in range(4):
            for col in range(4):
                cell = gray[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
                histogram = cv2.calcHist([cell], [0], None, [16], [0, 256]).flatten()
                cells.append(histogram)
        return np.concatenate(cells).astype(np.float32)
    except Exception:
        log.exception("histogram descriptor failed")
        return None


def _l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.asarray(vector, dtype=np.float32)
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32)


# =============================================================================
#  Enrolment
# =============================================================================
def enrol_embedding(image: np.ndarray) -> np.ndarray | None:
    """
    Produce a gallery vector from a registration photo.

    Deliberately identical to the inference path - same detector, same
    alignment, same normalisation.  Store the result together with
    `ACTIVE_BACKEND` and `embedding_dim()` so a later backend change can be
    detected instead of silently producing UNKNOWN for everyone.
    """
    embedding, box = embed_from_image(image)
    if embedding is None:
        log.warning("enrolment failed: no usable face in the supplied image")
        return None
    log.info("enrolled a %d-d embedding via %s (face at %s)",
             len(embedding), ACTIVE_BACKEND, box)
    return embedding


# =============================================================================
#  Matching against the registered gallery
# =============================================================================
class FaceGallery:
    """
    In-memory index of every registered owner embedding.

    Built once from `crud.load_face_gallery()` and refreshed after each
    enrolment.  Holding it in memory is the difference between a database
    round-trip per frame and a single matrix multiply per frame.
    """

    def __init__(self, entries: list[tuple] | None = None):
        self._owner_ids: list[int] = []
        self._matrix: np.ndarray | None = None
        self._dim: int = 0
        if entries:
            self.rebuild(entries)

    def rebuild(self, entries: list[tuple]) -> None:
        """
        Replace the gallery contents.

        `entries` may be `(owner_id, vector)` or `(owner_id, vector, backend)`.
        Vectors are validated against the *active* backend's dimensionality,
        not against whatever the first row happened to be - otherwise a gallery
        enrolled under a previous backend loads cleanly and then fails every
        single comparison.
        """
        expected = embedding_dim()
        self._dim = expected

        if not entries:
            self._owner_ids, self._matrix = [], None
            log.info("face gallery is empty - every person will be UNKNOWN")
            return

        owner_ids: list[int] = []
        vectors: list[np.ndarray] = []
        stale = 0

        for entry in entries:
            owner_id, vector = entry[0], np.asarray(entry[1], dtype=np.float32)
            source = entry[2] if len(entry) > 2 else None

            if source is not None and ACTIVE_BACKEND is not None and source != ACTIVE_BACKEND:
                stale += 1
                log.error(
                    "owner %s was enrolled with backend '%s' but '%s' is active "
                    "- skipped. Re-enrol this owner.",
                    owner_id, source, ACTIVE_BACKEND,
                )
                continue

            if vector.size != expected:
                stale += 1
                log.error(
                    "owner %s has a %d-d embedding but %s produces %d-d - "
                    "skipped. Re-enrol this owner after a backend change.",
                    owner_id, vector.size, ACTIVE_BACKEND, expected,
                )
                continue

            owner_ids.append(owner_id)
            vectors.append(_l2_normalise(vector))

        self._owner_ids = owner_ids
        self._matrix = np.vstack(vectors) if vectors else None

        log.info(
            "face gallery rebuilt: %d embeddings across %d owners (%d-d, %s), "
            "cutoff %.3f",
            len(owner_ids), len(set(owner_ids)), expected, ACTIVE_BACKEND,
            match_threshold(),
        )
        if stale:
            log.error(
                "%d of %d registered embeddings were rejected as incompatible - "
                "those owners CANNOT be recognised until they are re-enrolled.",
                stale, len(entries),
            )

    @property
    def size(self) -> int:
        return len(self._owner_ids)

    def match(self, embedding: np.ndarray, threshold: float | None = None) -> FaceMatch:
        """
        Find the closest registered owner.

        Args:
            embedding: L2-normalised query vector.
            threshold: minimum cosine similarity to accept. Defaults to the
                active backend's cutoff. Higher is stricter.

        Returns:
            AUTHORIZED with the owner id when the best similarity clears the
            threshold, otherwise UNAUTHORIZED with the best score observed - the
            score is retained either way so an operator reviewing the alert can
            see how close the call was.
        """
        cutoff = threshold if threshold is not None else match_threshold()

        if self._matrix is None or embedding is None:
            return FaceMatch(PersonStatus.UNKNOWN, None, 0.0)

        query = _l2_normalise(np.asarray(embedding, dtype=np.float32))
        if query.shape[0] != self._matrix.shape[1]:
            log.error(
                "query embedding is %d-d but the gallery is %d-d - cannot match. "
                "The gallery was built under a different backend; re-enrol.",
                query.shape[0], self._matrix.shape[1],
            )
            return FaceMatch(PersonStatus.UNKNOWN, None, 0.0)

        # Both sides are unit vectors, so the dot product IS cosine similarity.
        similarities = self._matrix @ query
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        log.debug(
            "match: best=%.3f cutoff=%.3f owner=%s gallery=%d",
            best_score, cutoff, self._owner_ids[best_index], self.size,
        )

        if best_score >= cutoff:
            return FaceMatch(
                PersonStatus.AUTHORIZED, self._owner_ids[best_index], best_score
            )
        return FaceMatch(PersonStatus.UNAUTHORIZED, None, best_score)


def identify_person(
    frame: np.ndarray,
    gallery: FaceGallery,
    person_bbox: tuple[int, int, int, int] | None = None,
) -> FaceMatch:
    """
    Full pipeline for one person: locate face → align → embed → match.

    Args:
        frame: the full BGR frame.
        gallery: the registered owner index.
        person_bbox: optional person box from YOLO. Restricting the face search
            to it is both faster and less error-prone than scanning the frame.

    Returns:
        A `FaceMatch`. `NO_FACE` means a person was present but their face was
        not visible - a routine outcome from overhead CCTV, and deliberately
        distinguished from UNAUTHORIZED so the theft engine can weight it
        differently.
    """
    if frame is None or frame.size == 0:
        return FaceMatch(PersonStatus.NO_FACE, None, 0.0)

    embedding, face_box = embed_from_image(frame, person_bbox)

    if embedding is None:
        if face_box is None:
            return FaceMatch(PersonStatus.NO_FACE, None, 0.0)
        # A face was located but could not be embedded - a real failure, not an
        # absence, so it is not reported as NO_FACE.
        return FaceMatch(PersonStatus.UNKNOWN, None, 0.0, face_box)

    result = gallery.match(embedding)
    result.bbox = face_box
    return result