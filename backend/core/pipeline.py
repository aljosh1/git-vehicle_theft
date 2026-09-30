"""
Real-time surveillance pipeline.

Handles:
    Camera/file capture
    Vehicle and person detection
    License plate detection and OCR
    Face recognition
    Vehicle-owner verification
    Threat scoring
    Database logging
    Alert dispatch
    MJPEG frame output

Important:
    Numeric camera sources such as "0", "1", "2" are automatically converted
    to integer camera indexes. This is important because values coming from a
    React input normally arrive as strings.

Example:
    "0"  -> camera 0
    "1"  -> camera 1
    "2"  -> camera 2

Non-numeric strings are treated as file/RTSP/HTTP sources.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.alerts.dispatcher import dispatch_alert, format_alert_for_console
from backend.config import settings
from backend.core.security import random_filename
from backend.core.theft_engine import (
    ThreatAssessment,
    TheftEngine,
    is_night_time,
)
from backend.database import crud
from backend.database.base import session_scope
from backend.database.models import (
    OwnerVerificationStatus,
    PersonStatus,
    ThreatLevel,
)
from backend.detection.base import Detection
from backend.detection.plate_detector import PlateDetector
from backend.detection.threat_detector import ThreatDetector
from backend.detection.vehicle_detector import VehicleDetector
from backend.recognition import face_recognizer as face
from backend.recognition.plate_ocr import fuzzy_match_plate, read_plate
from backend.utils.logger import get_logger


log = get_logger(__name__)

_ACTIVE_PIPELINES: dict[str, "Pipeline"] = {}


# ========================================================================
# DATA CLASSES
# ========================================================================

@dataclass
class VehicleAnalysis:
    """Recognition and theft verdict associated with exactly one vehicle."""

    vehicle: Detection

    nearby_persons: list[Detection] = field(default_factory=list)

    plate_text: str | None = None
    plate_confidence: float | None = None

    matched_vehicle_id: int | None = None
    plate_in_database: bool | None = None
    vehicle_flagged_stolen: bool = False

    expected_owner_id: int | None = None

    person_status: PersonStatus | None = None
    face_similarity: float | None = None
    matched_user_id: int | None = None

    owner_verification: OwnerVerificationStatus | None = None

    assessment: ThreatAssessment | None = None

    def to_dict(self) -> dict:
        assessment = self.assessment

        return {
            "vehicle": self.vehicle.to_dict(),
            "nearby_persons": [
                person.to_dict()
                for person in self.nearby_persons
            ],
            "plate": {
                "text": self.plate_text,
                "confidence": self.plate_confidence,
                "in_database": self.plate_in_database,
                "flagged_stolen": self.vehicle_flagged_stolen,
            },
            "matched_vehicle_id": self.matched_vehicle_id,
            "person_status": (
                self.person_status.value
                if self.person_status
                else None
            ),
            "face_similarity": self.face_similarity,
            "matched_user_id": self.matched_user_id,
            "owner_verification": (
                self.owner_verification.value
                if self.owner_verification
                else None
            ),
            "threat": {
                "score": assessment.score if assessment else 0,
                "level": (
                    assessment.level.value
                    if assessment
                    else "none"
                ),
                "reason": (
                    assessment.reason
                    if assessment
                    else ""
                ),
                "should_alert": (
                    assessment.should_alert
                    if assessment
                    else False
                ),
            },
        }


@dataclass
class FrameResult:
    """Everything that happened in one processed frame."""

    frame_number: int
    timestamp: float

    vehicles: list[Detection] = field(default_factory=list)
    persons: list[Detection] = field(default_factory=list)
    animals: list[Detection] = field(default_factory=list)
    plates: list[Detection] = field(default_factory=list)

    # Threat objects. Kept separate from `persons` because they are scored
    # separately, drawn differently, and an operator reviewing an alert needs to
    # see "this person is holding a knife" rather than a stray box in frame.
    weapons: list[Detection] = field(default_factory=list)
    masks: list[Detection] = field(default_factory=list)
    occluded_faces: list[Detection] = field(default_factory=list)

    plate_text: str | None = None
    plate_confidence: float | None = None

    matched_vehicle_id: int | None = None
    plate_in_database: bool | None = None
    vehicle_flagged_stolen: bool = False

    expected_owner_id: int | None = None

    person_status: PersonStatus | None = None
    face_similarity: float | None = None
    matched_user_id: int | None = None

    owner_verification: OwnerVerificationStatus | None = None

    vehicle_analyses: list[VehicleAnalysis] = field(
        default_factory=list
    )

    threat_score: int = 0
    threat_level_name: str = "none"
    threat_reason: str = ""
    threat_triggers: str = ""
    is_armed: bool = False
    is_concealed: bool = False

    processing_ms: float = 0.0
    fps: float = 0.0


@dataclass
class ClipJob:
    """Pending alert clip built from buffered pre-frames + streamed post-frames."""

    alert_id: int
    frame_bytes: list[bytes] = field(default_factory=list)
    remaining_post_frames: int = 0


# ========================================================================
# PIPELINE
# ========================================================================

class Pipeline:
    """
    One surveillance camera or video pipeline.

    Supports:

        0
        1
        2

    for camera indexes, as well as:

        "0"
        "1"
        "2"

    Numeric strings are automatically converted to integer camera indexes.

    File paths and network streams remain strings.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int = 0,
        enable_tracking: bool = True,
        enable_alerts: bool = True,
        enable_persistence: bool = True,
    ):
        self.camera_id = camera_id

        # ------------------------------------------------------------
        # IMPORTANT FIX:
        #
        # React normally sends input values as strings.
        #
        # Therefore:
        #     "1" -> 1
        #
        # instead of incorrectly treating "1" as a video filename.
        # ------------------------------------------------------------

        self.source = self._normalise_source(source)

        self.enable_tracking = enable_tracking
        self.enable_alerts = enable_alerts
        self.enable_persistence = enable_persistence

        self.running = False

        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

        self._frame_queue: queue.Queue[
            np.ndarray | None
        ] = queue.Queue(maxsize=2)

        self._latest_annotated: np.ndarray | None = None
        self._latest_result: FrameResult | None = None

        self._lock = threading.Lock()

        # ------------------------------------------------------------
        # DETECTORS
        # ------------------------------------------------------------

        self.vehicle_detector = VehicleDetector(
            enable_tracking=enable_tracking
        )

        # ---- threat (weapon / mask) detection -----------------------------
        # Shares the main detector so the COCO `knife` class can be recovered
        # from the pass that has already run, rather than paying for a second
        # inference over the same frame.
        self.threat_detector = ThreatDetector(
            fallback_detector=self.vehicle_detector
        )

        self.plate_detector = PlateDetector()

        self.face_gallery = face.FaceGallery()

        self.theft_engine = TheftEngine()

        # ------------------------------------------------------------
        # STAGE COUNTERS
        # ------------------------------------------------------------

        self._frame_count = 0
        self._ocr_counter = 0
        self._face_counter = 0

        # ------------------------------------------------------------
        # TRACK CACHES
        # ------------------------------------------------------------

        self._plate_cache: dict[
            int,
            tuple[
                str,
                float,
                int | None,
                bool,
                int | None,
                int,
            ],
        ] = {}

        self._face_cache: dict[
            int,
            tuple[
                face.FaceMatch,
                int,
            ],
        ] = {}

        # Track IDs can be recycled by trackers; keep cache lifetime short.
        self._cache_max_age_frames = 90

        # ------------------------------------------------------------
        # FPS
        # ------------------------------------------------------------

        self._fps_times: deque[float] = deque(maxlen=30)

        # ------------------------------------------------------------
        # ALERT CLIP RECORDING
        # ------------------------------------------------------------

        self._clip_fps = max(1, int(settings.ALERT_CLIP_FPS))
        self._clip_pre_frames = max(
            1,
            int(settings.ALERT_CLIP_PRE_SECONDS) * self._clip_fps,
        )
        self._clip_post_frames = max(
            1,
            int(settings.ALERT_CLIP_POST_SECONDS) * self._clip_fps,
        )
        self._recent_clip_frames: deque[bytes] = deque(
            maxlen=self._clip_pre_frames,
        )
        self._active_clip_jobs: list[ClipJob] = []

        log.info(
            "pipeline %s initialised: source=%r type=%s persistence=%s",
            self.camera_id,
            self.source,
            type(self.source).__name__,
            self.enable_persistence,
        )

    # ====================================================================
    # SOURCE HANDLING
    # ====================================================================

    @staticmethod
    def _normalise_source(
        source: str | int | None,
    ) -> str | int:
        """
        Convert numeric camera strings into integers.

        Examples:

            "0" -> 0
            "1" -> 1
            " 1 " -> 1

        But:

            "video.mp4" -> "video.mp4"
            "rtsp://..." -> unchanged
            "http://..." -> unchanged
        """

        if source is None:
            return 0

        if isinstance(source, int):
            return source

        source_text = str(source).strip()

        # Numeric camera index
        if source_text.isdigit():
            return int(source_text)

        return source_text

    @staticmethod
    def _is_camera_source(
        source: str | int,
    ) -> bool:
        """Return True when the source represents a local camera."""

        return isinstance(source, int)

    @staticmethod
    def _is_network_source(
        source: str | int,
    ) -> bool:
        """Return True for RTSP/HTTP/RTMP network streams."""

        if not isinstance(source, str):
            return False

        value = source.lower().strip()

        return value.startswith(
            (
                "rtsp://",
                "http://",
                "https://",
                "rtmp://",
            )
        )

    @classmethod
    def _is_file_source(
        cls,
        source: str | int,
    ) -> bool:
        """Return True for local video-file paths."""

        if not isinstance(source, str):
            return False

        if cls._is_network_source(source):
            return False

        return True

    # ====================================================================
    # FACE GALLERY
    # ====================================================================

    def _load_face_gallery(self) -> None:
        """Refresh the in-memory owner gallery from the database."""

        with session_scope() as db:
            gallery_data = crud.load_face_gallery(db)

        self.face_gallery.rebuild(gallery_data)

    # ====================================================================
    # LIFECYCLE
    # ====================================================================

    def start(self) -> None:
        """Launch capture and processing threads."""

        if self.running:
            log.warning(
                "pipeline %s is already running",
                self.camera_id,
            )
            return

        # Make sure any stale queue data is removed.
        self._clear_frame_queue()
        self._recent_clip_frames.clear()
        self._active_clip_jobs.clear()

        self._load_face_gallery()

        self.running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.camera_id}",
            daemon=True,
        )

        self._process_thread = threading.Thread(
            target=self._process_loop,
            name=f"process-{self.camera_id}",
            daemon=True,
        )

        self._capture_thread.start()
        self._process_thread.start()

        _ACTIVE_PIPELINES[self.camera_id] = self

        log.info(
            "pipeline %s started with source=%r",
            self.camera_id,
            self.source,
        )

    def stop(self) -> None:
        """Stop capture and processing threads safely."""

        if not self.running:
            _ACTIVE_PIPELINES.pop(
                self.camera_id,
                None,
            )
            return

        log.info(
            "stopping pipeline %s",
            self.camera_id,
        )

        self.running = False
        self._active_clip_jobs.clear()

        # Unblock process thread without risking a queue deadlock.
        self._put_sentinel_nonblocking()

        if self._capture_thread:
            self._capture_thread.join(timeout=5)

        if self._process_thread:
            self._process_thread.join(timeout=5)

        _ACTIVE_PIPELINES.pop(
            self.camera_id,
            None,
        )

        log.info(
            "pipeline %s stopped",
            self.camera_id,
        )

    def _clear_frame_queue(self) -> None:
        """Remove stale frames before starting."""

        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _put_sentinel_nonblocking(self) -> None:
        """Insert the shutdown sentinel without blocking."""

        try:
            self._frame_queue.put_nowait(None)
            return
        except queue.Full:
            pass

        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

    # ====================================================================
    # OUTPUT
    # ====================================================================

    def get_latest_frame(
        self,
    ) -> tuple[
        np.ndarray | None,
        FrameResult | None,
    ]:
        """Thread-safe accessor for the MJPEG streamer."""

        with self._lock:
            return (
                self._latest_annotated,
                self._latest_result,
            )

    # ====================================================================
    # IMAGE ANALYSIS
    # ====================================================================

    def analyse_image(
        self,
        frame: np.ndarray,
        *,
        dispatch_alerts: bool | None = None,
    ) -> tuple[np.ndarray, FrameResult]:
        """
        Run the complete pipeline synchronously on one image.
        """

        if frame is None or frame.size == 0:
            raise ValueError("image is empty")

        self._load_face_gallery()

        self._frame_count += 1

        started = time.perf_counter()

        result = self._process_frame(
            frame,
            force_recognition=True,
        )

        result.processing_ms = (
            time.perf_counter() - started
        ) * 1000

        result.fps = (
            1000.0 / result.processing_ms
            if result.processing_ms > 0
            else 0.0
        )

        annotated = self._annotate_frame(
            frame,
            result,
        )

        if (
            result.vehicles
            or result.persons
            or result.threat_score >= 20
        ):
            self._log_to_database(result)

        should_dispatch = (
            self.enable_alerts
            if dispatch_alerts is None
            else dispatch_alerts
        )

        if (
            should_dispatch
            and result.threat_score
            >= settings.THREAT_ALERT_THRESHOLD
        ):
            self._dispatch_alert(
                result,
                annotated,
            )

        with self._lock:
            self._latest_annotated = annotated
            self._latest_result = result

        return annotated, result

    # ====================================================================
    # CAPTURE
    # ====================================================================

    def _open_capture(self):
        """
        Open the configured video source.

        Windows local cameras use DirectShow first.
        """

        source = self.source

        # ------------------------------------------------------------
        # LOCAL CAMERA
        # ------------------------------------------------------------

        if self._is_camera_source(source):

            log.info(
                "opening local camera index %s using DirectShow",
                source,
            )

            cap = cv2.VideoCapture(
                source,
                cv2.CAP_DSHOW,
            )

            if cap.isOpened():
                log.info(
                    "camera %s opened successfully with DirectShow",
                    source,
                )
                return cap

            log.warning(
                "DirectShow failed for camera %s; "
                "trying default OpenCV backend",
                source,
            )

            cap.release()

            cap = cv2.VideoCapture(source)

            if cap.isOpened():
                log.info(
                    "camera %s opened with default OpenCV backend",
                    source,
                )
                return cap

            cap.release()

            return None

        # ------------------------------------------------------------
        # NETWORK OR FILE SOURCE
        # ------------------------------------------------------------

        log.info(
            "opening non-camera source: %s",
            source,
        )

        cap = cv2.VideoCapture(source)

        if cap.isOpened():
            return cap

        cap.release()

        return None

    def _configure_capture(self, cap) -> None:
        """Configure capture properties."""

        try:
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                settings.FRAME_WIDTH,
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                settings.FRAME_HEIGHT,
            )

            cap.set(
                cv2.CAP_PROP_FPS,
                settings.TARGET_FPS,
            )

            # Small buffer helps reduce latency for live cameras.
            if self._is_camera_source(self.source):
                cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1,
                )

        except Exception:
            log.exception(
                "failed to configure capture properties"
            )

    def _log_capture_properties(self, cap) -> None:
        """Log actual camera properties reported by OpenCV."""

        try:
            width = cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )

            height = cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            backend = cap.getBackendName()

            log.info(
                "capture properties: backend=%s "
                "width=%s height=%s fps=%s",
                backend,
                width,
                height,
                fps,
            )

        except Exception:
            log.exception(
                "could not read capture properties"
            )

    def _capture_loop(self) -> None:
        """
        Capture live frames or local video frames.

        For cameras:
            Keep the newest frame.

        For files:
            Process selected frames in order.
        """

        cap = self._open_capture()

        if cap is None or not cap.isOpened():

            log.error(
                "could not open video source %r",
                self.source,
            )

            self.running = False

            self._put_sentinel_nonblocking()

            return

        self._configure_capture(cap)

        self._log_capture_properties(cap)

        is_camera = self._is_camera_source(
            self.source
        )

        is_network = self._is_network_source(
            self.source
        )

        is_file = self._is_file_source(
            self.source
        )

        total_frames = 0
        file_stride = 1

        if is_file:

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            file_stride = max(
                1,
                settings.VIDEO_FILE_FRAME_STRIDE,
            )

            if total_frames > 0:

                selected_frames = (
                    total_frames
                    + file_stride
                    - 1
                ) // file_stride

                log.info(
                    "video file source: %d frames; "
                    "selecting about %d at stride %d",
                    total_frames,
                    selected_frames,
                    file_stride,
                )

        consecutive_failures = 0

        source_frame_count = 0
        selected_frame_count = 0

        max_failures = 30
        reconnect_attempts = 0
        # Keep monitoring resilient: 0 means unlimited reconnect attempts.
        max_reconnect_attempts = 0

        log.info(
            "capture started: source=%r "
            "camera=%s network=%s file=%s",
            self.source,
            is_camera,
            is_network,
            is_file,
        )

        # ============================================================
        # MAIN CAPTURE LOOP
        # ============================================================

        while self.running:

            ok, frame = cap.read()

            # --------------------------------------------------------
            # FAILED READ
            # --------------------------------------------------------

            if not ok or frame is None:

                # A local file ending is normal.
                if is_file:

                    log.info(
                        "end of video file %s - stopping capture",
                        self.source,
                    )

                    break

                consecutive_failures += 1

                if consecutive_failures >= max_failures:

                    if is_camera or is_network:

                        reconnect_attempts += 1

                        log.warning(
                                "source %r failed %d reads; reconnecting "
                                "(attempt %d%s)",
                            self.source,
                            consecutive_failures,
                            reconnect_attempts,
                                (
                                    f"/{max_reconnect_attempts}"
                                    if max_reconnect_attempts > 0
                                    else ""
                                ),
                        )

                        try:
                            cap.release()
                        except Exception:
                            pass

                        time.sleep(0.5)

                        cap = self._open_capture()

                        if cap is None or not cap.isOpened():

                            if (
                                max_reconnect_attempts > 0
                                and reconnect_attempts >= max_reconnect_attempts
                            ):
                                log.error(
                                    "source %r could not recover after %d "
                                    "reconnect attempts - stopping capture",
                                    self.source,
                                    reconnect_attempts,
                                )
                                break

                            # Continue retrying to keep long-running monitoring alive
                            # when a camera or network source glitches temporarily.
                            consecutive_failures = 0
                            continue

                        self._configure_capture(cap)
                        self._log_capture_properties(cap)
                        consecutive_failures = 0
                        continue

                    log.error(
                        "source %r failed %d consecutive reads - "
                        "stopping capture",
                        self.source,
                        consecutive_failures,
                    )

                    break

                if consecutive_failures % 5 == 1:

                    log.warning(
                        "frame grab failed on source %r "
                        "(attempt %d/%d)",
                        self.source,
                        consecutive_failures,
                        max_failures,
                    )

                # Give the camera/backend a moment to recover.
                time.sleep(0.05)

                continue

            # --------------------------------------------------------
            # SUCCESSFUL READ
            # --------------------------------------------------------

            consecutive_failures = 0
            reconnect_attempts = 0

            source_frame_count += 1

            # --------------------------------------------------------
            # FILE FRAME STRIDING
            # --------------------------------------------------------

            if is_file:

                if not self._should_process_file_frame(
                    source_frame_count,
                    file_stride,
                ):
                    continue

            selected_frame_count += 1

            # --------------------------------------------------------
            # LIVE CAMERA
            #
            # Keep newest frame.
            # If processing is slow, throw away stale frames.
            # --------------------------------------------------------

            if is_camera or is_network:

                try:

                    self._frame_queue.put_nowait(
                        frame
                    )

                except queue.Full:

                    try:
                        # Remove stale frame.
                        self._frame_queue.get_nowait()

                    except queue.Empty:
                        pass

                    try:
                        # Insert newest frame.
                        self._frame_queue.put_nowait(
                            frame
                        )

                    except queue.Full:
                        pass

            # --------------------------------------------------------
            # VIDEO FILE
            #
            # Preserve frame ordering.
            # --------------------------------------------------------

            else:

                while self.running:

                    try:

                        self._frame_queue.put(
                            frame,
                            timeout=0.5,
                        )

                        break

                    except queue.Full:
                        continue

        # ============================================================
        # CLEANUP
        # ============================================================

        cap.release()

        self._put_sentinel_nonblocking()

        if is_file:

            log.info(
                "capture thread exiting: "
                "decoded %d frames, selected %d "
                "at stride %d",
                source_frame_count,
                selected_frame_count,
                file_stride,
            )

        else:

            log.info(
                "capture thread exiting: "
                "decoded %d frames",
                source_frame_count,
            )

    # ====================================================================
    # FILE FRAME SELECTION
    # ====================================================================

    @staticmethod
    def _should_process_file_frame(
        source_frame_number: int,
        stride: int,
    ) -> bool:
        """Select first file frame and then every Nth frame."""

        return (
            stride <= 1
            or (source_frame_number - 1) % stride == 0
        )

    # ====================================================================
    # PROCESSING THREAD
    # ====================================================================

    def _process_loop(self) -> None:
        """Process frames from the queue."""

        while self.running:

            try:

                frame = self._frame_queue.get(
                    timeout=1
                )

            except queue.Empty:
                continue

            # --------------------------------------------------------
            # END-OF-STREAM
            # --------------------------------------------------------

            if frame is None:

                self.running = False

                break

            # --------------------------------------------------------
            # PROCESS FRAME
            # --------------------------------------------------------

            self._frame_count += 1

            started = time.perf_counter()

            try:

                result = self._process_frame(
                    frame
                )

            except Exception:

                log.exception(
                    "frame processing failed "
                    "on camera %s",
                    self.camera_id,
                )

                continue

            elapsed = (
                time.perf_counter()
                - started
            )

            result.processing_ms = (
                elapsed * 1000
            )

            self._fps_times.append(
                time.perf_counter()
            )

            result.fps = self._measure_fps()

            annotated = self._annotate_frame(
                frame,
                result,
            )

            self._update_clip_buffers(
                annotated,
            )

            # --------------------------------------------------------
            # STORE LATEST FRAME
            # --------------------------------------------------------

            with self._lock:

                self._latest_annotated = annotated

                self._latest_result = result

            # --------------------------------------------------------
            # DATABASE LOGGING
            # --------------------------------------------------------

            if self.enable_persistence:
                if (
                    result.vehicles
                    or result.persons
                    or result.threat_score >= 20
                ):

                    self._log_to_database(
                        result
                    )

            # --------------------------------------------------------
            # ALERT
            # --------------------------------------------------------

            if (
                self.enable_alerts
                and result.threat_score
                >= settings.THREAT_ALERT_THRESHOLD
            ):

                self._dispatch_alert(
                    result,
                    annotated,
                )

        log.info(
            "process thread exiting"
        )

    # ====================================================================
    # FRAME PROCESSING
    # ====================================================================

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        force_recognition: bool = False,
    ) -> FrameResult:
        """
        Full detection -> recognition -> threat assessment.
        """

        result = FrameResult(
            frame_number=self._frame_count,
            timestamp=time.time(),
        )

        # ============================================================
        # STAGE 1
        # VEHICLE + PERSON DETECTION
        # ============================================================

        detections = self.vehicle_detector.detect(
            frame
        )

        result.vehicles = [
            detection
            for detection in detections
            if detection.is_vehicle
        ]

        result.persons = [
            detection
            for detection in detections
            if detection.is_person
        ]

        result.animals = [
            detection
            for detection in detections
            if detection.is_animal
        ]

        # ============================================================
        # STAGE 1b
        # THREAT OBJECTS (weapons / masks)
        # ============================================================

        # Runs before the "nothing in frame" early return below, so an armed or
        # masked intruder is recorded even when no vehicle is in shot. A person
        # carrying a crowbar in an empty car park is still an incident, and
        # returning early on `no vehicles and no persons` would discard exactly
        # the case this stage exists to catch.
        threats = self.threat_detector.detect(frame, detections)
        result.weapons = [d for d in threats if d.is_threatening]
        result.masks = [d for d in threats if d.is_mask]
        result.occluded_faces = [d for d in threats if d.label == "face_occluded"]

        if threats:
            log.warning(
                "THREAT OBJECTS in frame %s: %s",
                self._frame_count,
                ", ".join(
                    f"{d.label} {d.confidence:.2f} ({d.meta.get('source', '?')})"
                    for d in threats
                ),
            )

        self._prune_track_caches(
            result.vehicles,
            result.persons,
        )

        # ------------------------------------------------------------
        # No vehicle/person found
        # ------------------------------------------------------------

        if (
            not result.vehicles
            and not result.persons
        ):

            # A weapon with nothing else in frame is still an alert. The
            # early return below would otherwise discard the one case that most
            # needs reporting: someone carrying a crowbar through an otherwise
            # empty car park.
            if threats:
                assessment = self.theft_engine.assess(
                    vehicles=[],
                    persons=[],
                    weapons=result.weapons,
                    masks=result.masks,
                    occluded_faces=result.occluded_faces,
                )
                result.threat_score = assessment.score
                result.threat_level_name = assessment.level.value
                result.threat_reason = assessment.reason
                result.threat_triggers = assessment.triggers_csv
                result.is_armed = assessment.is_armed
                result.is_concealed = assessment.is_concealed
                return result

            if force_recognition:

                (
                    text,
                    confidence,
                    matched_id,
                    in_db,
                    flagged,
                    owner_id,
                    plate_dets,
                ) = self._recognise_plate_in_frame(
                    frame
                )

                result.plates.extend(
                    plate_dets
                )

                result.plate_text = text
                result.plate_confidence = confidence
                result.matched_vehicle_id = matched_id
                result.plate_in_database = in_db
                result.vehicle_flagged_stolen = flagged
                result.expected_owner_id = owner_id

            return result

        result.vehicle_analyses = [
            VehicleAnalysis(
                vehicle=vehicle
            )
            for vehicle in result.vehicles
        ]

        # ============================================================
        # STAGE 2
        # PLATE LOCALISATION + OCR
        # ============================================================

        self._ocr_counter += 1

        run_ocr = (
            force_recognition
            or self._ocr_counter
            >= settings.OCR_EVERY_N_FRAMES
        )

        if result.vehicles and run_ocr:

            self._ocr_counter = 0

            for analysis in result.vehicle_analyses:

                (
                    text,
                    confidence,
                    matched_id,
                    in_db,
                    flagged,
                    owner_id,
                    plate_dets,
                ) = self._recognise_plate(
                    frame,
                    analysis.vehicle,
                )

                result.plates.extend(
                    plate_dets
                )

                analysis.plate_text = text
                analysis.plate_confidence = confidence
                analysis.matched_vehicle_id = matched_id
                analysis.plate_in_database = in_db
                analysis.vehicle_flagged_stolen = flagged
                analysis.expected_owner_id = owner_id

                if text:

                    analysis.vehicle.meta[
                        "plate"
                    ] = text

                    analysis.vehicle.meta[
                        "plate_conf"
                    ] = confidence

        elif result.vehicles:

            for analysis in result.vehicle_analyses:

                cached = self._get_cached_plate(
                    analysis.vehicle.track_id
                )

                if cached is None:
                    continue

                (
                    text,
                    confidence,
                    matched_id,
                    flagged_stolen,
                    owner_id,
                ) = cached

                analysis.plate_text = text
                analysis.plate_confidence = confidence
                analysis.matched_vehicle_id = matched_id
                analysis.plate_in_database = matched_id is not None
                analysis.vehicle_flagged_stolen = flagged_stolen
                analysis.expected_owner_id = owner_id

                analysis.vehicle.meta[
                    "plate"
                ] = text

                analysis.vehicle.meta[
                    "plate_conf"
                ] = confidence

        # ============================================================
        # STAGE 3
        # FACE RECOGNITION
        # ============================================================

        self._face_counter += 1

        run_faces = (
            force_recognition
            or self._face_counter
            >= settings.FACE_EVERY_N_FRAMES
        )

        if result.persons and run_faces:

            self._face_counter = 0

            for person in result.persons:

                match = self._recognise_face(
                    frame,
                    person,
                )

                if match:

                    person.meta[
                        "face_status"
                    ] = match.status.value

                    person.meta[
                        "face_sim"
                    ] = match.similarity

                    person.meta[
                        "matched_user_id"
                    ] = match.owner_id

        elif result.persons:

            for person in result.persons:

                match = self._get_cached_face(
                    person.track_id
                )

                if not match:
                    continue

                person.meta[
                    "face_status"
                ] = match.status.value

                person.meta[
                    "face_sim"
                ] = match.similarity

                person.meta[
                    "matched_user_id"
                ] = match.owner_id

        # ============================================================
        # STAGE 4
        # ASSOCIATE PEOPLE WITH VEHICLES
        # ============================================================

        night = is_night_time()

        person_assignments = (
            self._assign_persons_to_vehicles(
                result.vehicles,
                result.persons,
            )
        )

        for analysis in result.vehicle_analyses:

            analysis.nearby_persons = (
                person_assignments.get(
                    id(analysis.vehicle),
                    [],
                )
            )

            face_person = max(
                (
                    person
                    for person
                    in analysis.nearby_persons
                    if person.meta.get(
                        "face_status"
                    )
                    not in {
                        None,
                        PersonStatus.NO_FACE.value,
                    }
                ),
                key=lambda person: float(
                    person.meta.get(
                        "face_sim",
                        0.0,
                    )
                ),
                default=None,
            )

            if face_person is not None:

                analysis.person_status = (
                    PersonStatus(
                        face_person.meta[
                            "face_status"
                        ]
                    )
                )

                analysis.face_similarity = float(
                    face_person.meta.get(
                        "face_sim",
                        0.0,
                    )
                )

                analysis.matched_user_id = (
                    face_person.meta.get(
                        "matched_user_id"
                    )
                )

            # --------------------------------------------------------
            # VERIFY OWNER
            # --------------------------------------------------------

            self._verify_vehicle_owner(
                analysis
            )

            # --------------------------------------------------------
            # THREAT ASSESSMENT
            # --------------------------------------------------------

            nearby_weapons, nearby_masks, nearby_occlusions = (
                self._threats_for(result, analysis.nearby_persons)
            )

            analysis.assessment = (
                self.theft_engine.assess(
                    vehicles=[
                        analysis.vehicle
                    ],
                    persons=[
                        *analysis.nearby_persons
                    ],
                    plate_number=analysis.plate_text,
                    plate_in_database=(
                        analysis.plate_in_database
                    ),
                    matched_vehicle_id=(
                        analysis.matched_vehicle_id
                    ),
                    vehicle_flagged_stolen=(
                        analysis.vehicle_flagged_stolen
                    ),
                    person_status=(
                        analysis.person_status
                    ),
                    face_similarity=(
                        analysis.face_similarity
                    ),
                    matched_user_id=(
                        analysis.matched_user_id
                    ),
                    owner_verification=(
                        analysis.owner_verification
                    ),
                    weapons=nearby_weapons,
                    masks=nearby_masks,
                    occluded_faces=nearby_occlusions,
                    is_night=night,
                )
            )

            analysis.vehicle.meta[
                "threat_score"
            ] = analysis.assessment.score

            analysis.vehicle.meta[
                "threat_level"
            ] = analysis.assessment.level.value

        # ============================================================
        # PRIMARY RESULT
        # ============================================================

        if result.vehicle_analyses:

            highest = max(
                result.vehicle_analyses,
                key=lambda item:
                    item.assessment.score
                    if item.assessment
                    else 0,
            )

            self._apply_primary_analysis(
                result,
                highest,
            )

        elif result.persons:

            assessment = (
                self.theft_engine.assess(
                    vehicles=[],
                    persons=result.persons,
                    weapons=result.weapons,
                    masks=result.masks,
                    occluded_faces=result.occluded_faces,
                    is_night=night,
                )
            )

            result.threat_score = (
                assessment.score
            )

            result.threat_level_name = (
                assessment.level.value
            )

            result.threat_reason = (
                assessment.reason
            )

        # ------------------------------------------------------------
        # TRACKING
        # ------------------------------------------------------------

        self.theft_engine.update_tracking(
            result.persons
        )

        return result

    # ====================================================================
    # PERSON / VEHICLE ASSOCIATION
    # ====================================================================

    @staticmethod
    def _assign_persons_to_vehicles(
        vehicles: list[Detection],
        persons: list[Detection],
    ) -> dict[int, list[Detection]]:

        assignments: dict[
            int,
            list[Detection]
        ] = {
            id(vehicle): []
            for vehicle in vehicles
        }

        for person in persons:

            px, py = person.center

            candidates: list[
                tuple[float, Detection]
            ] = []

            for vehicle in vehicles:

                vx, vy = vehicle.center

                distance = (
                    (px - vx) ** 2
                    + (py - vy) ** 2
                ) ** 0.5

                scale = max(
                    vehicle.width,
                    vehicle.height,
                    1.0,
                )

                max_distance = max(
                    120.0,
                    0.75 * scale,
                )

                overlap = vehicle.iou(
                    person
                )

                if (
                    overlap > 0
                    or distance <= max_distance
                ):

                    score = (
                        distance / scale
                        - (
                            2.0
                            if overlap > 0
                            else 0.0
                        )
                    )

                    candidates.append(
                        (
                            score,
                            vehicle,
                        )
                    )

            if candidates:

                _, nearest = min(
                    candidates,
                    key=lambda item: item[0],
                )

                assignments[
                    id(nearest)
                ].append(person)

        return assignments

    # ====================================================================
    # PRIMARY ANALYSIS
    # ====================================================================

    @staticmethod
    def _apply_primary_analysis(
        result: FrameResult,
        analysis: VehicleAnalysis,
    ) -> None:

        assessment = analysis.assessment

        result.plate_text = (
            analysis.plate_text
        )

        result.plate_confidence = (
            analysis.plate_confidence
        )

        result.matched_vehicle_id = (
            analysis.matched_vehicle_id
        )

        result.plate_in_database = (
            analysis.plate_in_database
        )

        result.vehicle_flagged_stolen = (
            analysis.vehicle_flagged_stolen
        )

        result.expected_owner_id = (
            analysis.expected_owner_id
        )

        result.person_status = (
            analysis.person_status
        )

        result.face_similarity = (
            analysis.face_similarity
        )

        result.matched_user_id = (
            analysis.matched_user_id
        )

        result.owner_verification = (
            analysis.owner_verification
        )

        if assessment:

            result.threat_score = (
                assessment.score
            )

            result.threat_level_name = (
                assessment.level.value
            )

            result.threat_reason = (
                assessment.reason
            )

    # ====================================================================
    # OWNER VERIFICATION
    # ====================================================================

    @staticmethod
    def _verify_vehicle_owner(
        analysis: VehicleAnalysis,
    ) -> None:
        """
        Authorize a face only when it belongs to
        the plate-registered vehicle owner.
        """

        if (
            analysis.plate_in_database
            is not True
            or analysis.expected_owner_id
            is None
        ):

            analysis.owner_verification = (
                OwnerVerificationStatus.NOT_APPLICABLE
            )

            return

        if analysis.matched_user_id is None:

            analysis.owner_verification = (
                OwnerVerificationStatus.NOT_RECOGNIZED
            )

            return

        if (
            analysis.matched_user_id
            == analysis.expected_owner_id
        ):

            analysis.owner_verification = (
                OwnerVerificationStatus.VERIFIED
            )

            analysis.person_status = (
                PersonStatus.AUTHORIZED
            )

        else:

            analysis.owner_verification = (
                OwnerVerificationStatus.MISMATCH
            )

            analysis.person_status = (
                PersonStatus.UNAUTHORIZED
            )

    # ====================================================================
    # PLATE RECOGNITION
    # ====================================================================

    def _recognise_plate(
        self,
        frame: np.ndarray,
        vehicle: Detection,
    ) -> tuple[
        str | None,
        float | None,
        int | None,
        bool | None,
        bool,
        int | None,
        list[Detection],
    ]:

        # ------------------------------------------------------------
        # CACHE
        # ------------------------------------------------------------

        cached = self._get_cached_plate(
            vehicle.track_id
        )

        if cached is not None:

            (
                text,
                conf,
                matched_id,
                flagged_stolen,
                owner_id,
            ) = cached

            return (
                text,
                conf,
                matched_id,
                matched_id is not None,
                flagged_stolen,
                owner_id,
                [],
            )

        # ------------------------------------------------------------
        # VEHICLE CROP
        # ------------------------------------------------------------

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x1 = max(
            0,
            int(vehicle.x1),
        )

        y1 = max(
            0,
            int(vehicle.y1),
        )

        x2 = min(
            frame_width,
            int(vehicle.x2),
        )

        y2 = min(
            frame_height,
            int(vehicle.y2),
        )

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            return (
                None,
                None,
                None,
                None,
                False,
                None,
                [],
            )

        # ------------------------------------------------------------
        # PLATE DETECTION
        # ------------------------------------------------------------

        plate_dets = (
            self.plate_detector.detect(
                crop
            )
        )

        frame_plate_dets = [

            Detection(
                label=plate.label,
                confidence=plate.confidence,
                x1=plate.x1 + x1,
                y1=plate.y1 + y1,
                x2=plate.x2 + x1,
                y2=plate.y2 + y1,
                class_id=plate.class_id,
                track_id=vehicle.track_id,
                meta={
                    **plate.meta,
                    "vehicle_track_id":
                        vehicle.track_id,
                },
            )

            for plate in plate_dets
        ]

        if not plate_dets:

            return (
                None,
                None,
                None,
                None,
                False,
                None,
                frame_plate_dets,
            )

        # ------------------------------------------------------------
        # BEST PLATE
        # ------------------------------------------------------------

        best_plate = plate_dets[0]

        plate_crop = (
            self._extract_plate_crop(
                crop,
                best_plate,
            )
        )

        ocr_result = read_plate(
            plate_crop
        )

        if not ocr_result:

            return (
                None,
                None,
                None,
                None,
                False,
                None,
                frame_plate_dets,
            )

        text, conf = ocr_result

        # ------------------------------------------------------------
        # DATABASE MATCH
        # ------------------------------------------------------------

        (
            matched_id,
            flagged_stolen,
            owner_id,
        ) = self._lookup_plate_in_db(
            text
        )

        self._remember_plate(
            track_id=vehicle.track_id,
            text=text,
            confidence=conf,
            matched_id=matched_id,
            flagged_stolen=flagged_stolen,
            owner_id=owner_id,
        )

        return (
            text,
            conf,
            matched_id,
            matched_id is not None,
            flagged_stolen,
            owner_id,
            frame_plate_dets,
        )

    # ====================================================================
    # FULL-FRAME PLATE RECOGNITION
    # ====================================================================

    def _recognise_plate_in_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[
        str | None,
        float | None,
        int | None,
        bool | None,
        bool,
        int | None,
        list[Detection],
    ]:

        plate_dets = (
            self.plate_detector.detect(
                frame
            )
        )

        if not plate_dets:

            return (
                None,
                None,
                None,
                None,
                False,
                None,
                [],
            )

        best_plate = plate_dets[0]

        plate_crop = (
            self._extract_plate_crop(
                frame,
                best_plate,
            )
        )

        ocr_result = read_plate(
            plate_crop
        )

        if not ocr_result:

            return (
                None,
                None,
                None,
                None,
                False,
                None,
                plate_dets,
            )

        text, conf = ocr_result

        (
            matched_id,
            flagged_stolen,
            owner_id,
        ) = self._lookup_plate_in_db(
            text
        )

        return (
            text,
            conf,
            matched_id,
            matched_id is not None,
            flagged_stolen,
            owner_id,
            plate_dets,
        )

    # ====================================================================
    # DATABASE PLATE LOOKUP
    # ====================================================================

    @staticmethod
    def _lookup_plate_in_db(
        text: str,
    ) -> tuple[
        int | None,
        bool,
        int | None,
    ]:

        with session_scope() as db:

            matched = (
                crud.get_vehicle_by_plate(
                    db,
                    text,
                )
            )

            if not matched:

                candidates = (
                    crud.list_vehicles(
                        db,
                        limit=200,
                    )
                )

                for candidate in candidates:

                    if fuzzy_match_plate(
                        text,
                        candidate.plate_number,
                    ):

                        matched = candidate

                        break

            if not matched:

                return (
                    None,
                    False,
                    None,
                )

            return (
                matched.id,
                bool(
                    matched.is_flagged_stolen
                ),
                matched.owner_id,
            )

    # ====================================================================
    # FACE RECOGNITION
    # ====================================================================

    def _recognise_face(
        self,
        frame: np.ndarray,
        person: Detection,
    ) -> face.FaceMatch | None:

        cached = self._get_cached_face(
            person.track_id
        )

        if cached is not None:
            return cached

        match = face.identify_person(
            frame,
            self.face_gallery,
            person.bbox,
        )

        if match.status is not PersonStatus.NO_FACE:
            self._remember_face(
                person.track_id,
                match,
            )

        return match

    # ====================================================================
    # TRACK CACHES
    # ====================================================================

    def _prune_track_caches(
        self,
        vehicles: list[Detection],
        persons: list[Detection],
    ) -> None:
        vehicle_ids = {
            item.track_id
            for item in vehicles
            if item.track_id is not None
        }
        person_ids = {
            item.track_id
            for item in persons
            if item.track_id is not None
        }

        min_frame = (
            self._frame_count
            - self._cache_max_age_frames
        )

        self._plate_cache = {
            track_id: value
            for track_id, value in self._plate_cache.items()
            if track_id in vehicle_ids
            or value[-1] >= min_frame
        }

        self._face_cache = {
            track_id: value
            for track_id, value in self._face_cache.items()
            if track_id in person_ids
            or value[1] >= min_frame
        }

    def _remember_plate(
        self,
        *,
        track_id: int | None,
        text: str,
        confidence: float,
        matched_id: int | None,
        flagged_stolen: bool,
        owner_id: int | None,
    ) -> None:
        if track_id is None:
            return

        self._plate_cache[track_id] = (
            text,
            confidence,
            matched_id,
            flagged_stolen,
            owner_id,
            self._frame_count,
        )

    def _get_cached_plate(
        self,
        track_id: int | None,
    ) -> tuple[
        str,
        float,
        int | None,
        bool,
        int | None,
    ] | None:
        if track_id is None:
            return None

        payload = self._plate_cache.get(track_id)
        if payload is None:
            return None

        (
            text,
            confidence,
            matched_id,
            flagged_stolen,
            owner_id,
            seen_frame,
        ) = payload

        if (
            self._frame_count
            - seen_frame
            > self._cache_max_age_frames
        ):
            self._plate_cache.pop(track_id, None)
            return None

        self._plate_cache[track_id] = (
            text,
            confidence,
            matched_id,
            flagged_stolen,
            owner_id,
            self._frame_count,
        )

        return (
            text,
            confidence,
            matched_id,
            flagged_stolen,
            owner_id,
        )

    def _remember_face(
        self,
        track_id: int | None,
        match: face.FaceMatch,
    ) -> None:
        if track_id is None:
            return
        self._face_cache[track_id] = (
            match,
            self._frame_count,
        )

    def _get_cached_face(
        self,
        track_id: int | None,
    ) -> face.FaceMatch | None:
        if track_id is None:
            return None

        payload = self._face_cache.get(track_id)
        if payload is None:
            return None

        match, seen_frame = payload

        if (
            self._frame_count
            - seen_frame
            > self._cache_max_age_frames
        ):
            self._face_cache.pop(track_id, None)
            return None

        self._face_cache[track_id] = (
            match,
            self._frame_count,
        )
        return match

    # ====================================================================
    # FPS
    # ====================================================================

    def _measure_fps(self) -> float:

        if len(self._fps_times) < 2:
            return 0.0

        elapsed = (
            self._fps_times[-1]
            - self._fps_times[0]
        )

        if elapsed <= 0:
            return 0.0

        return (
            len(self._fps_times)
            / elapsed
        )

    # ====================================================================
    # ANNOTATION
    # ====================================================================

    def _threats_for(
        self, result: FrameResult, persons: list[Detection]
    ) -> tuple[list[Detection], list[Detection], list[Detection]]:
        """
        Attribute threat objects to a specific set of people.

        Necessary because the theft engine assesses *per vehicle*: a frame with
        two cars and a knife held by someone standing at car 1 must not produce
        a critical alert for car 2 as well. A weapon is attributed to the people
        whose box contains it; anything unattributable (a weapon on the ground, or
        one whose centre falls just outside a person box) is still counted,
        because dropping it would hide a weapon that is plainly in the frame.

        Masks are not attributed the same way. A covered face cannot be reliably
        located *relative to* a person box - the box is the face itself - so a
        mask seen anywhere applies to the people present.
        """
        attributed = [
            weapon
            for weapon in result.weapons
            if any(person.contains(weapon) for person in persons)
        ]
        return attributed or list(result.weapons), result.masks, result.occluded_faces

    def _annotate_frame(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:

        annotated = (
            self.vehicle_detector.annotate(
                frame,
                (
                    result.vehicles
                    + result.persons
                    + result.animals
                    + result.plates
                ),
            )
        )

        # Threat overlays go on last, so a weapon box is never painted over by
        # the person box that carries it.
        threats = (
            result.weapons + result.masks + result.occluded_faces
        )
        if threats:
            annotated = self.threat_detector.annotate(annotated, threats)

        colour = {
            "none": (100, 100, 100),
            "low": (0, 191, 255),
            "medium": (0, 165, 255),
            "high": (0, 100, 255),
            "critical": (0, 0, 255),
        }.get(
            result.threat_level_name,
            (255, 255, 255),
        )

        # ------------------------------------------------------------
        # HUD
        # ------------------------------------------------------------

        cv2.rectangle(
            annotated,
            (10, 10),
            (380, 90),
            (0, 0, 0),
            -1,
        )

        cv2.rectangle(
            annotated,
            (10, 10),
            (380, 90),
            colour,
            2,
        )

        cv2.putText(
            annotated,
            f"Threat: {result.threat_score}/100",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
        )

        cv2.putText(
            annotated,
            f"FPS: {result.fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            annotated,
            f"Frame: {result.frame_number}",
            (200, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Explicitly flag unregistered vehicles in red, independent of
        # whether they have crossed the alert threshold yet.
        for analysis in result.vehicle_analyses:

            if analysis.plate_in_database is not False:
                continue

            v = analysis.vehicle
            x1, y1, x2, y2 = int(v.x1), int(v.y1), int(v.x2), int(v.y2)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3,
            )

            label = "UNVERIFIED PLATE"
            if analysis.plate_text:
                label = f"{label}: {analysis.plate_text}"

            text_y = max(25, y1 - 10)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        return annotated

    # ====================================================================
    # DATABASE LOGGING
    # ====================================================================

    def _log_to_database(
        self,
        result: FrameResult,
    ) -> None:

        try:

            with session_scope() as db:

                analysed_ids = {
                    id(item.vehicle)
                    for item
                    in result.vehicle_analyses
                }

                records = [
                    (
                        item.vehicle,
                        item,
                    )
                    for item
                    in result.vehicle_analyses
                ]

                records += [
                    (
                        det,
                        None,
                    )
                    for det
                    in (
                        result.vehicles
                        + result.persons
                        + result.animals
                    )
                    if id(det)
                    not in analysed_ids
                ]

                for det, item in records:

                    assessment = (
                        item.assessment
                        if item
                        else None
                    )

                    crud.create_detection_log(
                        db,
                        camera_id=self.camera_id,
                        frame_number=result.frame_number,
                        object_class=det.label,
                        confidence=det.confidence,
                        bbox_x1=det.x1,
                        bbox_y1=det.y1,
                        bbox_x2=det.x2,
                        bbox_y2=det.y2,
                        track_id=det.track_id,
                        plate_number=(
                            item.plate_text
                            if item
                            else None
                        ),
                        plate_confidence=(
                            item.plate_confidence
                            if item
                            else None
                        ),
                        plate_in_database=(
                            item.plate_in_database
                            if item
                            else None
                        ),
                        person_status=(
                            item.person_status
                            if item
                            else None
                        ),
                        face_similarity=(
                            item.face_similarity
                            if item
                            else None
                        ),
                        matched_user_id=(
                            item.matched_user_id
                            if item
                            else None
                        ),
                        expected_owner_id=(
                            item.expected_owner_id
                            if item
                            else None
                        ),
                        owner_verification=(
                            item.owner_verification
                            if item
                            else None
                        ),
                        threat_score=(
                            assessment.score
                            if assessment
                            else 0
                        ),
                        threat_level=(
                            assessment.level.value
                            if assessment
                            else "none"
                        ),
                        reason=(
                            assessment.reason
                            if assessment
                            else ""
                        ),
                        processing_ms=(
                            result.processing_ms
                        ),
                        fps=result.fps,
                        vehicle_id=(
                            item.matched_vehicle_id
                            if item
                            else None
                        ),
                    )

        except Exception:

            log.exception(
                "failed to write detection log"
            )

    # ====================================================================
    # ALERTS
    # ====================================================================

    def _encode_clip_frame(
        self,
        frame: np.ndarray,
    ) -> bytes | None:
        if frame is None or frame.size == 0:
            return None

        target = frame

        try:
            max_width = max(160, int(settings.ALERT_CLIP_MAX_WIDTH))
            height, width = frame.shape[:2]

            if width > max_width:
                ratio = max_width / float(width)
                target = cv2.resize(
                    frame,
                    (max_width, max(1, int(height * ratio))),
                    interpolation=cv2.INTER_AREA,
                )

            ok, encoded = cv2.imencode(
                ".jpg",
                target,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    max(40, min(95, int(settings.ALERT_CLIP_JPEG_QUALITY))),
                ],
            )
            if not ok:
                return None
            return encoded.tobytes()
        except Exception:
            log.exception("failed to encode frame for alert clip")
            return None

    def _update_clip_buffers(
        self,
        annotated: np.ndarray,
    ) -> None:
        encoded = self._encode_clip_frame(annotated)
        if encoded is None:
            return

        self._recent_clip_frames.append(encoded)

        completed: list[ClipJob] = []

        for job in self._active_clip_jobs:
            job.frame_bytes.append(encoded)
            job.remaining_post_frames -= 1
            if job.remaining_post_frames <= 0:
                completed.append(job)

        if completed:
            done_ids = {job.alert_id for job in completed}
            self._active_clip_jobs = [
                job
                for job in self._active_clip_jobs
                if job.alert_id not in done_ids
            ]

            for job in completed:
                self._finalise_clip_async(job)

    def _start_alert_clip_job(
        self,
        alert_id: int,
    ) -> None:
        # Keep one job per alert ID.
        if any(job.alert_id == alert_id for job in self._active_clip_jobs):
            return

        job = ClipJob(
            alert_id=alert_id,
            frame_bytes=list(self._recent_clip_frames),
            remaining_post_frames=self._clip_post_frames,
        )
        self._active_clip_jobs.append(job)

    def _finalise_clip_async(
        self,
        job: ClipJob,
    ) -> None:
        frames = list(job.frame_bytes)

        def _worker() -> None:
            clip_rel_path = self._render_clip(frames)
            if clip_rel_path is None:
                return

            try:
                with session_scope() as db:
                    alert = crud.get_alert(db, job.alert_id)
                    if alert is None:
                        return
                    crud.set_alert_video_clip(db, alert, clip_rel_path)
            except Exception:
                log.exception(
                    "failed to attach clip %s to alert %s",
                    clip_rel_path,
                    job.alert_id,
                )

        threading.Thread(
            target=_worker,
            name=f"clip-finalise-{job.alert_id}",
            daemon=True,
        ).start()

    def _render_clip(
        self,
        frame_bytes: list[bytes],
    ) -> str | None:
        if not frame_bytes:
            return None

        settings.ensure_directories()
        filename = f"clip_{random_filename('.mp4')}"
        clip_path = settings.evidence_dir / filename

        writer = None

        try:
            first = cv2.imdecode(
                np.frombuffer(frame_bytes[0], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if first is None or first.size == 0:
                return None

            height, width = first.shape[:2]

            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self._clip_fps),
                (int(width), int(height)),
            )

            if not writer.isOpened():
                log.error("could not open clip writer for %s", clip_path)
                return None

            writer.write(first)

            for blob in frame_bytes[1:]:
                frame = cv2.imdecode(
                    np.frombuffer(blob, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None or frame.size == 0:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)

            return f"evidence/{filename}"

        except Exception:
            log.exception("failed to render alert clip")
            return None
        finally:
            if writer is not None:
                writer.release()

    def _dispatch_alert(
        self,
        result: FrameResult,
        annotated: np.ndarray,
    ) -> None:

        try:

            with session_scope() as db:

                for item in result.vehicle_analyses:

                    if (
                        not item.assessment
                        or not item.assessment.should_alert
                    ):
                        continue

                    person = next(
                        (
                            candidate
                            for candidate
                            in item.nearby_persons
                            if candidate.meta.get(
                                "face_status"
                            )
                            in {
                                "authorized",
                                "unauthorized",
                            }
                        ),
                        None,
                    )

                    face_crop = None

                    if person is not None:

                        face_crop = (
                            annotated[
                                person.y1:person.y2,
                                person.x1:person.x2,
                            ]
                        )

                    alert = dispatch_alert(
                        db,
                        item.assessment,
                        camera_id=self.camera_id,
                        frame=annotated,
                        face_crop=face_crop,
                    )

                    if alert:

                        if item.assessment.level in {
                            ThreatLevel.HIGH,
                            ThreatLevel.CRITICAL,
                        }:
                            self._start_alert_clip_job(alert.id)

                        print(
                            format_alert_for_console(
                                alert
                            ),
                            flush=True,
                        )

        except Exception:

            log.exception(
                "failed to dispatch alert"
            )

    # ====================================================================
    # PLATE CROP
    # ====================================================================

    @staticmethod
    def _extract_plate_crop(
        frame: np.ndarray,
        plate: Detection,
    ) -> np.ndarray:
        """Return cleaned, margin-expanded plate crop."""

        if (
            frame is None
            or frame.size == 0
        ):
            return frame

        height, width = frame.shape[:2]

        x1 = max(
            0,
            int(plate.x1),
        )

        y1 = max(
            0,
            int(plate.y1),
        )

        x2 = min(
            width,
            int(plate.x2),
        )

        y2 = min(
            height,
            int(plate.y2),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            return np.empty(
                (0, 0, 3),
                dtype=np.uint8,
            )

        pad_x = max(
            10,
            int(
                (x2 - x1)
                * 0.20
            ),
        )

        pad_y = max(
            10,
            int(
                (y2 - y1)
                * 0.20
            ),
        )

        x1 = max(
            0,
            x1 - pad_x,
        )

        y1 = max(
            0,
            y1 - pad_y,
        )

        x2 = min(
            width,
            x2 + pad_x,
        )

        y2 = min(
            height,
            y2 + pad_y,
        )

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return crop

        h, w = crop.shape[:2]

        target_h = max(
            80,
            h,
        )

        target_w = max(
            160,
            int(
                round(
                    w
                    * (
                        120.0
                        / max(h, 1)
                    )
                )
            ),
        )

        crop = cv2.resize(
            crop,
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        crop = cv2.copyMakeBorder(
            crop,
            8,
            8,
            8,
            8,
            cv2.BORDER_REPLICATE,
        )

        return crop


# ========================================================================
# GLOBAL PIPELINE ACCESS
# ========================================================================

def get_pipeline(
    camera_id: str,
) -> Pipeline | None:
    """
    Retrieve an active pipeline by camera ID.
    """

    return _ACTIVE_PIPELINES.get(
        camera_id
    )