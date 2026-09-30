"""
Owner management (administrator only).

    GET    /api/owners        list owners
    POST   /api/owners        create an owner or admin
    GET    /api/owners/{id}   one owner
    PATCH  /api/owners/{id}   edit
    DELETE /api/owners/{id}   remove (cascades to vehicles + face embeddings)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.api.deps import require_admin
from backend.database import crud, schemas
from backend.database.base import get_db
from backend.database.models import User, UserRole
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("", response_model=list[schemas.UserRead])
def list_owners(
    role: UserRole | None = None,
    search: str | None = Query(None, description="match name or email"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    users = crud.list_users(db, role=role, search=search, skip=skip, limit=limit)
    return [crud.to_user_read(db, u) for u in users]


@router.post("", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
def create_owner(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create an owner or another administrator. Role IS honoured here."""
    try:
        user = crud.create_user(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return crud.to_user_read(db, user)


@router.get("/{user_id}", response_model=schemas.UserRead)
def get_owner(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Owner not found")
    return crud.to_user_read(db, user)


@router.patch("/{user_id}", response_model=schemas.UserRead)
def update_owner(
    user_id: int, payload: schemas.UserUpdate, db: Session = Depends(get_db)
):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Owner not found")
    return crud.to_user_read(db, crud.update_user(db, user, payload))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_owner(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Delete an owner and their vehicles + face embeddings.

    Refuses self-deletion: an administrator removing their own account could
    leave the system with no administrator at all.
    """
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Owner not found")
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account",
        )
    crud.delete_user(db, user)
