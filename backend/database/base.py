"""
Database engine and session management.

Why these specific settings
---------------------------
SQLite is a file database, and the surveillance pipeline writes to it from a
background capture thread while FastAPI serves requests from its own threads.
Two consequences are handled here:

1. `check_same_thread=False` - SQLite's Python driver refuses cross-thread use
   of a connection by default.  We disable that guard and instead guarantee
   that every unit of work gets its own `Session` (via `get_db()` or
   `session_scope()`), which is the pattern SQLAlchemy is designed around.

2. `PRAGMA journal_mode=WAL` + `busy_timeout` - Write-Ahead Logging lets the
   dashboard keep reading while the detector is writing, instead of hitting
   "database is locked".  The 5 s busy timeout absorbs the remaining brief
   write contention.

For a multi-camera production deployment the only change needed is a
PostgreSQL `DATABASE_URL`; nothing above this module knows which engine it is
talking to.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model."""


def _build_engine() -> Engine:
    """Create the engine, making sure the SQLite file's folder exists first."""
    url = settings.DATABASE_URL
    connect_args: dict[str, object] = {}

    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 15}
        # "sqlite:///./data/vtds.db" -> ./data must exist before connecting.
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )

    return create_engine(
        url,
        connect_args=connect_args,
        echo=False,            # set True to dump every SQL statement
        future=True,
        pool_pre_ping=True,
    )


engine: Engine = _build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # keep ORM objects usable after commit()
    class_=Session,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
    """Enable WAL, foreign keys and a busy timeout on every new connection."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")   # off by default in SQLite
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a request-scoped session.

        @router.get("/vehicles")
        def list_vehicles(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Transactional session for background threads and CLI scripts.

    Commits on success, rolls back on any exception, and always closes:

        with session_scope() as db:
            crud.create_detection_log(db, ...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        log.exception("database transaction rolled back")
        raise
    finally:
        db.close()


def _upgrade_sqlite_schema() -> None:
    """
    Add columns introduced after the first release, without data loss.

    SQLite has no `ALTER COLUMN`, so a new column is added as nullable and
    backfilled in Python. `users.owner_code` is the only column that *cannot*
    stay nullable - it is unique and every card depends on it - so it is
    backfilled with a freshly generated code for each pre-existing account.
    Those owners must re-print their card; the log says so explicitly, because
    silently issuing a card that matches an old one would be worse.
    """
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    from sqlalchemy import select

    with engine.begin() as connection:
        inspector = inspect(connection)

        # --- detection_logs: two nullable columns --------------------------
        log_columns = {
            column["name"] for column in inspector.get_columns("detection_logs")
        }
        additions = {
            "expected_owner_id": "INTEGER REFERENCES users(id) ON DELETE SET NULL",
            "owner_verification": "VARCHAR(20)",
        }
        for name, definition in additions.items():
            if name not in log_columns:
                connection.execute(
                    text(f"ALTER TABLE detection_logs ADD COLUMN {name} {definition}")
                )
                log.info("added detection_logs.%s to the existing SQLite database", name)

        # --- users.owner_code ---------------------------------------------
        user_columns = {column["name"] for column in inspector.get_columns("users")}
        if "owner_code" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN owner_code VARCHAR(20)"))
            log.info("added users.owner_code to the existing SQLite database")
            _backfill_owner_codes(connection, select)
        else:
            # The column exists, but a row may still be missing a code if the
            # database was created between the two edits above.
            missing = connection.execute(
                text("SELECT COUNT(*) FROM users WHERE owner_code IS NULL")
            ).scalar()
            if missing:
                _backfill_owner_codes(connection, select)


def _backfill_owner_codes(connection, select) -> None:
    """
    Give every code-less account a unique owner code.

    Issued codes are re-drawn if one collides, rather than wrapping the whole
    backfill in a transaction and failing, because a single unlucky clash on an
    eight-character random code should not abort the startup path.
    """
    from backend.core.barcode import generate_owner_code

    rows = connection.execute(
        text("SELECT id FROM users WHERE owner_code IS NULL OR owner_code = ''")
    ).fetchall()

    taken = {
        value
        for (value,) in connection.execute(text("SELECT owner_code FROM users")).fetchall()
        if value
    }

    issued = 0
    for (user_id,) in rows:
        code = generate_owner_code()
        while code in taken:
            code = generate_owner_code()
        taken.add(code)
        connection.execute(
            text("UPDATE users SET owner_code = :code WHERE id = :id"),
            {"code": code, "id": user_id},
        )
        issued += 1

    if issued:
        log.warning(
            "assigned owner codes to %d existing account(s). These owners must "
            "re-print their membership card - the old card's barcode is now invalid.",
            issued,
        )



def init_db() -> None:
    """Create missing tables and apply additive SQLite prototype upgrades."""
    # Importing the models module registers all five tables on Base.metadata.
    from backend.database import models  # noqa: F401

    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)
    _upgrade_sqlite_schema()
    log.info("database ready at %s", settings.DATABASE_URL)
