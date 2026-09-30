"""
Authentication endpoints.

    POST /api/auth/login      email + password -> JWT
    POST /api/auth/register   self-service vehicle-owner registration
    GET  /api/auth/me         current profile
    PATCH /api/auth/me        update own profile / password
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.config import settings
from backend.core.security import create_access_token
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User, UserRole
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.post("/login", response_model=schemas.Token)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    """
    Exchange credentials for an access token.

    Returns an identical 401 for unknown email, wrong password and disabled
    account - distinguishing them would turn this endpoint into an
    account-enumeration oracle.
    """
    user = crud.authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(
        user_id=user.id, email=user.email, role=user.role.value
    )
    return schemas.Token(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=crud.to_user_read(db, user),
    )


@router.post(
    "/register", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED
)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Self-service vehicle-owner registration.

    The `role` field is ignored here and forced to OWNER - accepting it from the
    request body would let anyone create themselves an administrator account.
    Admins are created by another admin via `/api/owners`, or by the seed script.
    """
    payload.role = UserRole.OWNER
    try:
        user = crud.create_user(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return crud.to_user_read(db, user)


@router.get("/me", response_model=schemas.UserRead)
def read_own_profile(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return crud.to_user_read(db, user)


@router.patch("/me", response_model=schemas.UserRead)
def update_own_profile(
    payload: schemas.UserUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update your own profile.

    `is_active` is stripped: a user must not be able to disable (or re-enable)
    their own account through the self-service endpoint.
    """
    payload.is_active = None
    updated = crud.update_user(db, user, payload)
    return crud.to_user_read(db, updated)
