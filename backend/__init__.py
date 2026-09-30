"""
Vehicle Theft Detection System - backend package.

Layer map (import direction always points downwards, never up):

    api/            HTTP surface: routers, request/response handling
      |
    core/           orchestration: theft engine, pipeline, security
      |
    detection/      YOLOv8 wrappers  (vehicles, persons, plates)
    recognition/    OCR + face embedding / matching
    alerts/         email + SMS dispatch
      |
    database/       SQLAlchemy models, CRUD, session management
      |
    config.py       settings   utils/  logging & helpers

Nothing in `database/` imports from `detection/`, and nothing in `detection/`
imports from `api/`.  Keeping that rule is what makes each module unit-testable
in isolation.
"""

__version__ = "1.0.0"
