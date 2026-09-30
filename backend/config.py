"""
Central configuration.

Every tunable value in the system lives here and is loaded from environment
variables / the `.env` file by pydantic-settings.  Nothing else in the codebase
reads `os.environ` directly, which means the whole system can be reconfigured
for a different deployment (new camera, new SMTP server, GPU instead of CPU)
without touching a single line of logic.

Usage:
    from backend.config import settings
    settings.VEHICLE_CONF_THRESHOLD
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../vehicle_theft_detection
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings with sensible demo-ready defaults."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------------------------------------------------------------- core --
    APP_NAME: str = "Vehicle Theft Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "insecure-dev-key-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720

    # ------------------------------------------------------------ database --
    DATABASE_URL: str = "sqlite:///./data/vtds.db"

    # --------------------------------------------------------- bootstrap ----
    ADMIN_EMAIL: str = "admin@vtds.example.com"
    ADMIN_PASSWORD: str = "Admin@12345"
    ADMIN_NAME: str = "System Administrator"

    # --------------------------------------------------------------- models -
    VEHICLE_MODEL_PATH: str = "models/yolov8n.pt"
    PLATE_MODEL_PATH: str = "models/license_plate.pt"
    PLATE_DETECTOR_BACKEND: str = "auto"  # auto | yolo | roboflow | opencv
    ROBOFLOW_API_URL: str = "https://serverless.roboflow.com"
    ROBOFLOW_MODEL_ID: str = "vehicle-license-plate-1hdcy/1"
    ROBOFLOW_API_KEY: str = ""
    FACE_MODEL_NAME: str = "Facenet512"      # DeepFace backbone
    FACE_DETECTOR_BACKEND: str = "retinaface"
    DEVICE: str = "cpu"

    # ---- Threat (weapon / mask) detection --------------------------------
    # The specification's third trigger - "unauthorised face attempts vehicle
    # access" - is only actionable if the system can tell that the person is
    # armed or is concealing their face. A mask defeats face recognition
    # entirely, so without this a thief in a balaclava is reported merely as
    # "unknown", which is the single most common real-world bypass.
    #
    # `auto` uses a purpose-trained model if THREAT_MODEL_PATH is present, and
    # otherwise falls back to the COCO `knife` class that ships with the stock
    # yolov8n.pt. Set THREAT_MODEL_PATH to get gun/knife/mask coverage.
    THREAT_DETECTION_ENABLED: bool = True
    THREAT_MODEL_PATH: str = ""            # e.g. "models/weapons_yolov8n.pt"
    THREAT_CONF_THRESHOLD: float = 0.30     # lower than COCO's: a small
                                             # distant knife is still a weapon
    # Occlusion-based mask inference. A dedicated mask model is far more
    # accurate, but this works with no extra weights at all, so the system has
    # a mask signal out of the box.
    FACE_OCCLUSION_ENABLED: bool = True
    FACE_OCCLUSION_RATIO: float = 0.55     # visible face area below this and
                                             # the face is considered covered

    # ----------------------------------------------------------- thresholds -
    VEHICLE_CONF_THRESHOLD: float = 0.45
    PERSON_CONF_THRESHOLD: float = 0.45
    PLATE_CONF_THRESHOLD: float = 0.35
    # Optional global override. When unset, backend-specific cutoffs are used
    # (for example SFace ~=0.363, Facenet512 ~=0.60).
    FACE_MATCH_THRESHOLD: float | None = None
    PLATE_FUZZY_THRESHOLD: float = 0.88

    # ---------------------------------------------------------- performance -
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    VIDEO_FILE_FRAME_STRIDE: int = 5         # analyse 1 in every N file frames
    DETECT_EVERY_N_FRAMES: int = 1
    FACE_EVERY_N_FRAMES: int = 2
    OCR_EVERY_N_FRAMES: int = 1
    JPEG_QUALITY: int = 75                   # MJPEG stream bandwidth/quality

    # --------------------------------------------------------- theft logic --
    THREAT_ALERT_THRESHOLD: int = 65
    ALERT_COOLDOWN_SECONDS: int = 120
    ALERT_CLIP_PRE_SECONDS: int = 10
    ALERT_CLIP_POST_SECONDS: int = 10
    ALERT_CLIP_MAX_WIDTH: int = 960
    ALERT_CLIP_JPEG_QUALITY: int = 70
    ALERT_CLIP_FPS: int = 12
    PROXIMITY_IOU_THRESHOLD: float = 0.02
    LOITER_SECONDS: int = 10                 # unknown person lingering -> +score

    # --------------------------------------------------------------- site ---
    SITE_NAME: str = "Main Gate"
    SITE_LATITUDE: float = 0.0
    SITE_LONGITUDE: float = 0.0

    # -------------------------------------------------------------- email ---
    ALERTS_EMAIL_ENABLED: bool = False
    EMAIL_PROVIDER: str = "smtp"          # smtp | resend

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_USE_TLS: bool = True
    ALERT_EMAIL_FROM: str = ""
    ALERT_EMAIL_TO: str = ""

    RESEND_API_URL: str = "https://api.resend.com/emails"
    RESEND_API_KEY: str = ""
    RESEND_FROM: str = ""

    # ---------------------------------------------------------------- sms ---
    ALERTS_SMS_ENABLED: bool = False
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""
    ALERT_SMS_TO: str = ""

    # --------------------------------------------------------------- cors ---
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ------------------------------------------------------------ logging ---
    LOG_LEVEL: str = "INFO"

    # -------------------------------------------------------- validators ----
    @field_validator("DEVICE")
    @classmethod
    def _normalise_device(cls, v: str) -> str:
        return v.strip().lower() or "cpu"

    @field_validator("PLATE_DETECTOR_BACKEND")
    @classmethod
    def _normalise_plate_backend(cls, v: str) -> str:
        mode = (v or "auto").strip().lower()
        allowed = {"auto", "yolo", "roboflow", "opencv"}
        return mode if mode in allowed else "auto"

    @field_validator("EMAIL_PROVIDER")
    @classmethod
    def _normalise_email_provider(cls, v: str) -> str:
        mode = (v or "smtp").strip().lower()
        allowed = {"smtp", "resend"}
        return mode if mode in allowed else "smtp"

    @field_validator("VIDEO_FILE_FRAME_STRIDE")
    @classmethod
    def _normalise_video_file_frame_stride(cls, v: int) -> int:
        return max(1, v)

    # ---------------------------------------------------- derived helpers ---
    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS as a list. `*` disables the allow-list entirely."""
        raw = self.CORS_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def email_recipients(self) -> list[str]:
        return [e.strip() for e in self.ALERT_EMAIL_TO.split(",") if e.strip()]

    @property
    def sms_recipients(self) -> list[str]:
        return [n.strip() for n in self.ALERT_SMS_TO.split(",") if n.strip()]

    # --- Runtime directories -------------------------------------------------
    # Kept as properties so they are always absolute regardless of the CWD the
    # server was launched from.
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def evidence_dir(self) -> Path:
        """Screenshots / video clips captured when a theft alert fires."""
        return self.data_dir / "evidence"

    @property
    def faces_dir(self) -> Path:
        """Registered owner face images."""
        return self.data_dir / "faces"

    @property
    def vehicles_dir(self) -> Path:
        """Registered vehicle photos."""
        return self.data_dir / "vehicles"

    @property
    def logs_dir(self) -> Path:
        return BASE_DIR / "logs"

    @property
    def models_dir(self) -> Path:
        return BASE_DIR / "models"

    def resolve_model(self, relative_or_absolute: str) -> Path:
        """Resolve a model path from `.env` against the project root."""
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else (BASE_DIR / p)

    def ensure_directories(self) -> None:
        """Create every runtime directory. Called once at application start."""
        for d in (
            self.data_dir,
            self.evidence_dir,
            self.faces_dir,
            self.vehicles_dir,
            self.logs_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the `.env` file is parsed only once per process."""
    return Settings()


settings = get_settings()
