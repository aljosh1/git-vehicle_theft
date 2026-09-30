"""
Password hashing and JWT access tokens.

Password storage
----------------
bcrypt with a per-password random salt and a work factor of 12.  bcrypt is
deliberately slow, which is what makes offline brute-forcing of a stolen
database expensive.  If the `bcrypt` wheel is unavailable the module degrades
to PBKDF2-HMAC-SHA256 (600k iterations, stdlib only) so the project still runs
on a machine without a compiler.  Both formats are recognised on verify, so an
existing database keeps working after bcrypt is installed.

Tokens
------
Stateless JWTs signed with `settings.SECRET_KEY` (HS256).  The payload carries
the user id (`sub`), `email`, `role` and an expiry (`exp`); the API can
therefore authorise a request without a database round-trip for the common
case, and role checks are a dictionary lookup.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)

# bcrypt is optional - fall back to PBKDF2 when the wheel is missing.
try:
    import bcrypt as _bcrypt

    _BCRYPT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    _bcrypt = None  # type: ignore[assignment]
    _BCRYPT_AVAILABLE = False
    log.warning(
        "bcrypt not installed - falling back to PBKDF2-HMAC-SHA256 for "
        "password hashing. Run `pip install bcrypt` for the stronger default."
    )

_BCRYPT_ROUNDS = 12
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_PREFIX = "pbkdf2_sha256$"


def _build_dummy_password_hash() -> str:
    """
    Valid hash used for timing-safe verification when a user does not exist.

    It must be syntactically valid for the active hashing backend, or auth
    fallback checks will raise and pollute logs with avoidable tracebacks.
    """
    if _BCRYPT_AVAILABLE:
        return _bcrypt.hashpw(b"dummy-password", _bcrypt.gensalt(_BCRYPT_ROUNDS)).decode()

    salt = b"\x00" * 16
    dk = hashlib.pbkdf2_hmac("sha256", b"dummy-password", salt, _PBKDF2_ITERATIONS)
    return (
        f"{_PBKDF2_PREFIX}{_PBKDF2_ITERATIONS}$"
        f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"
    )


DUMMY_PASSWORD_HASH = _build_dummy_password_hash()


# =============================================================================
#  Passwords
# =============================================================================
def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage in `users.hashed_password`."""
    if not plain_password:
        raise ValueError("password must not be empty")

    if _BCRYPT_AVAILABLE:
        # bcrypt silently truncates at 72 bytes; pre-hashing keeps long
        # passphrases fully significant.
        pw = _prehash(plain_password)
        return _bcrypt.hashpw(pw, _bcrypt.gensalt(_BCRYPT_ROUNDS)).decode()

    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", plain_password.encode(), salt, _PBKDF2_ITERATIONS
    )
    return (
        f"{_PBKDF2_PREFIX}{_PBKDF2_ITERATIONS}$"
        f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"
    )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Constant-time check of a login attempt.  Never raises - a malformed or
    truncated hash in the database is reported as "does not match".
    """
    if not plain_password or not hashed_password:
        return False

    try:
        if hashed_password.startswith(_PBKDF2_PREFIX):
            _, iterations, salt_b64, digest_b64 = hashed_password.split("$")
            dk = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode(),
                base64.b64decode(salt_b64),
                int(iterations),
            )
            return hmac.compare_digest(dk, base64.b64decode(digest_b64))

        if _BCRYPT_AVAILABLE:
            return _bcrypt.checkpw(
                _prehash(plain_password), hashed_password.encode()
            )

        log.error(
            "stored hash is bcrypt but the bcrypt package is not installed; "
            "cannot verify password"
        )
        return False
    except ValueError:
        # Most commonly: invalid bcrypt salt/hash string in the database.
        log.warning("password verification failed due to invalid hash format")
        return False
    except Exception:
        log.exception("password verification failed on a malformed hash")
        return False


def _prehash(password: str) -> bytes:
    """SHA-256 -> base64 so any-length password fits bcrypt's 72-byte input."""
    return base64.b64encode(hashlib.sha256(password.encode()).digest())


# =============================================================================
#  JWT access tokens
# =============================================================================
def create_access_token(
    *,
    user_id: int,
    email: str,
    role: str,
    expires_minutes: int | None = None,
) -> str:
    """Sign a JWT for a freshly authenticated user."""
    expire_minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
        "jti": secrets.token_urlsafe(8),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Validate signature + expiry and return the payload, or `None` if the token
    is expired, tampered with, or not a JWT at all.
    """
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        log.info("rejected an expired access token")
    except jwt.InvalidTokenError as exc:
        log.warning("rejected an invalid access token: %s", exc)
    return None


def generate_api_key(nbytes: int = 24) -> str:
    """Opaque key for unattended camera clients that cannot perform a login."""
    return secrets.token_urlsafe(nbytes)


def random_filename(suffix: str) -> str:
    """Collision-free name for an uploaded or captured image."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}{suffix}"
