"""
FastAPI application entry point.

    uvicorn backend.main:app --reload --port 8000

At this stage (Step 1 - foundation) the app wires up configuration, logging,
the database and CORS, and exposes health/info endpoints.  The feature routers
are registered in `_register_routers()`; each is added as its module is
implemented, and a missing module is reported as a warning instead of crashing
the server.  That is what lets the project be developed and demonstrated
incrementally.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from importlib import import_module

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.base import init_db
from backend.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# Routers to mount once implemented: (module path, url prefix, openapi tag).
ROUTER_SPECS: list[tuple[str, str, str]] = [
    ("backend.api.auth", "/api/auth", "Authentication"),
    ("backend.api.users", "/api/owners", "Owner management"),
    ("backend.api.vehicles", "/api/vehicles", "Vehicle management"),
    ("backend.api.faces", "/api/faces", "Face enrolment"),
    ("backend.api.barcodes", "/api/barcode", "Membership cards"),
    ("backend.api.detections", "/api/detections", "Detection logs"),
    ("backend.api.alerts", "/api/alerts", "Alerts"),
    ("backend.api.stream", "/api/stream", "Live monitoring"),
    ("backend.api.stats", "/api/stats", "Dashboard"),
    ("backend.api.training", "/api/training", "Model training"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown. Replaces the deprecated `@app.on_event` hooks."""
    setup_logging(settings.LOG_LEVEL, settings.logs_dir)
    settings.ensure_directories()
    init_db()
    log.info("%s v%s started (debug=%s)", settings.APP_NAME, settings.APP_VERSION, settings.DEBUG)
    if settings.SECRET_KEY.startswith("insecure") or "change-me" in settings.SECRET_KEY:
        log.warning("SECRET_KEY is still the default value - set it in .env")

    yield

    # Shutdown: the capture threads are stopped here once the pipeline exists.
    try:
        from backend.core.pipeline import shutdown_all_pipelines

        shutdown_all_pipelines()
    except ImportError:
        pass
    log.info("shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A Real-Time Deep Learning Framework for Vehicle Theft Detection "
        "Using Object Detection."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS: the React dashboard runs on a different origin in development ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Expose per-request latency - useful when profiling the detection API."""
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler: log the traceback server-side, return a generic body.

    Internal exception text can disclose file paths and query structure, so it
    is never sent to the client unless DEBUG is on.
    """
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    detail = str(exc) if settings.DEBUG else "Internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


def _register_routers() -> None:
    """Mount every router that has been implemented so far."""
    for module_path, prefix, tag in ROUTER_SPECS:
        try:
            module = import_module(module_path)
        except ImportError:
            log.debug("router %s not implemented yet - skipped", module_path)
            continue
        router = getattr(module, "router", None)
        if router is None:
            log.error("%s has no `router` attribute", module_path)
            continue
        app.include_router(router, prefix=prefix, tags=[tag])
        log.info("mounted %s -> %s", module_path, prefix)


_register_routers()

# Serve captured evidence and uploaded images so the dashboard can show them.
settings.ensure_directories()
app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")

# Training plots are generated outside the application data directory. Keep this
# read-only mount separate so the training monitor can display them safely.
_training_output_dir = settings.resolve_model("training/outputs")
_training_output_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/training-artifacts",
    StaticFiles(directory=_training_output_dir),
    name="training-artifacts",
)


@app.get("/health", tags=["System"])
def health() -> dict[str, object]:
    """Liveness probe. Used by the dashboard to show connection status."""
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/api/info", tags=["System"])
def info() -> dict[str, object]:
    """
    Runtime capability report.

    The dashboard uses this to grey out features whose optional dependency is
    not installed, instead of failing at the moment a user clicks them.
    """

    def installed(module: str) -> bool:
        try:
            import_module(module)
            return True
        except ImportError:
            return False

    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "site": settings.SITE_NAME,
        "device": settings.DEVICE,
        "target_fps": settings.TARGET_FPS,
        "capabilities": {
            "opencv": installed("cv2"),
            "torch": installed("torch"),
            "ultralytics": installed("ultralytics"),
            "inference_sdk": installed("inference_sdk"),
            "easyocr": installed("easyocr"),
            "deepface": installed("deepface"),
            "twilio": installed("twilio"),
        },
        "threat_detection": _threat_capability(),
        "alerts": {
            "email_enabled": settings.ALERTS_EMAIL_ENABLED,
            "sms_enabled": settings.ALERTS_SMS_ENABLED,
        },
        "mounted_routes": _api_route_paths(),
    }


def _threat_capability() -> dict[str, object]:
    """
    What weapon/mask detection is actually capable of right now.

    Reported rather than assumed, because the answer is materially different
    with and without `THREAT_MODEL_PATH`: a stock install can see a COCO knife
    and infer an occluded face, but it cannot see a gun. Claiming gun detection
    in a demonstration the system does not have would be the worst possible
    failure for a security tool.
    """
    model_path = settings.THREAT_MODEL_PATH
    has_model = bool(model_path) and settings.resolve_model(model_path).exists()
    return {
        "enabled": settings.THREAT_DETECTION_ENABLED,
        "weapons_model_installed": has_model,
        "gun_detection": has_model,
        "knife_detection": True,          # COCO class 79, always available
        "mask_detection": has_model,
        "mask_inference": settings.FACE_OCCLUSION_ENABLED,
        "confidence": settings.THREAT_CONF_THRESHOLD,
        "note": (
            "A weapons model is installed - full gun/knife/mask coverage."
            if has_model
            else "No THREAT_MODEL_PATH installed. Knives are detected via the "
                 "COCO class and masks are inferred from face occlusion. Set "
                 "THREAT_MODEL_PATH to detect guns and masks directly."
        ),
    }


def _api_route_paths() -> list[str]:
    """
    Every mounted `/api` path.

    Read from the generated OpenAPI schema rather than by walking `app.routes`:
    newer Starlette represents `include_router()` results as `_IncludedRouter`
    objects that expose neither `.path` nor a nested `.routes`, so a structural
    walk finds nothing behind a prefix. The schema is built from the real route
    table, so it always reflects what is actually mounted.
    """
    try:
        return sorted(p for p in app.openapi().get("paths", {}) if p.startswith("/api"))
    except Exception:
        log.exception("could not enumerate API routes from the OpenAPI schema")
        return []

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info("CORS origins: %s", settings.cors_origin_list)