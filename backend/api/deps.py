"""
Shared FastAPI dependencies.

Authentication is enforced here rather than in each route, so a new endpoint is
protected by default: adding `user: User = Depends(get_current_user)` is the
only step required.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.core.security import decode_access_token
from backend.database import crud
from backend.database.base import get_db
from backend.database.models import User, UserRole
from backend.utils.logger import get_logger

log = get_logger(__name__)

# auto_error=False so we can return our own 401 body rather than FastAPI's.
_bearer = HTTPBearer(auto_error=False)

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the bearer token to a live, active user."""
    if credentials is None or not credentials.credentials:
        raise _CREDENTIALS_ERROR

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise _CREDENTIALS_ERROR

    try:
        user_id = int(payload.get("sub", ""))
    except (TypeError, ValueError):
        raise _CREDENTIALS_ERROR from None

    # The token is signed, but the account may have been deleted or disabled
    # since it was issued - so the database is still the authority.
    user = crud.get_user(db, user_id)
    if user is None or not user.is_active:
        raise _CREDENTIALS_ERROR
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Restrict an endpoint to administrators."""
    if user.role is not UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires administrator privileges",
        )
    return user


def get_user_from_query_token(
    token: str = Query(..., description="JWT access token"),
    db: Session = Depends(get_db),
) -> User:
    """
    Authenticate from a query parameter instead of a header.

    Needed only for the MJPEG stream: it is consumed by an `<img src="...">`
    tag, and a browser will not attach an Authorization header to an image
    request. The token is short-lived and the endpoint is read-only, but it does
    mean the token appears in server logs - which is why this dependency is not
    used anywhere else.
    """
    payload = decode_access_token(token)
    if payload is None:
        raise _CREDENTIALS_ERROR
    try:
        user = crud.get_user(db, int(payload.get("sub", "")))
    except (TypeError, ValueError):
        raise _CREDENTIALS_ERROR from None
    if user is None or not user.is_active:
        raise _CREDENTIALS_ERROR
    return user
