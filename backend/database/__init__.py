"""
Database package.

* `base`     - engine, session factory, declarative `Base`, `get_db()` dependency
* `models`   - the five ORM tables (users, vehicles, face_embeddings,
               detection_logs, alerts)
* `schemas`  - pydantic request/response models used by the API
* `crud`     - every read/write helper, so no SQL lives in the routers
* `seed`     - creates the schema and the bootstrap administrator account
"""

from backend.database.base import Base, engine, get_db, init_db, session_scope

__all__ = ["Base", "engine", "get_db", "init_db", "session_scope"]
