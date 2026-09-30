"""
Database bootstrap.

    python -m backend.database.seed            # schema + admin account
    python -m backend.database.seed --demo     # also add demo owners/vehicles

Idempotent: existing rows are left alone, so it is safe to re-run.  The `--demo`
data exists to make the dashboard presentable during a project demonstration
before any real registration has happened.
"""

from __future__ import annotations

import argparse

from backend.config import settings
from backend.database import crud, schemas
from backend.database.base import init_db, session_scope
from backend.database.models import UserRole, VehicleType
from backend.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

DEMO_OWNERS = [
    {
        "email": "grace.okafor@example.com",
        "full_name": "Grace Okafor",
        "phone_number": "+2348012345678",
        "password": "Owner@12345",
        "vehicle": {
            "plate_display": "ABC-123XY",
            "make": "Toyota",
            "model": "Camry",
            "year": 2019,
            "color": "Silver",
            "vehicle_type": VehicleType.CAR,
        },
    },
    {
        "email": "daniel.mensah@example.com",
        "full_name": "Daniel Mensah",
        "phone_number": "+2348023456789",
        "password": "Owner@12345",
        "vehicle": {
            "plate_display": "LAG-456KJ",
            "make": "Honda",
            "model": "CR-V",
            "year": 2021,
            "color": "Black",
            "vehicle_type": VehicleType.CAR,
        },
    },
    {
        "email": "amina.bello@example.com",
        "full_name": "Amina Bello",
        "phone_number": "+2348034567890",
        "password": "Owner@12345",
        "vehicle": {
            "plate_display": "KJA-789ZB",
            "make": "Bajaj",
            "model": "Boxer",
            "year": 2022,
            "color": "Red",
            "vehicle_type": VehicleType.MOTORCYCLE,
        },
    },
]


def create_admin() -> None:
    """Create the bootstrap administrator from the `.env` credentials."""
    with session_scope() as db:
        if crud.get_user_by_email(db, settings.ADMIN_EMAIL):
            log.info("admin %s already exists - skipped", settings.ADMIN_EMAIL)
            return
        crud.create_user(
            db,
            schemas.UserCreate(
                email=settings.ADMIN_EMAIL,
                password=settings.ADMIN_PASSWORD,
                full_name=settings.ADMIN_NAME,
                role=UserRole.ADMIN,
            ),
        )
        log.info("created administrator %s", settings.ADMIN_EMAIL)
        if settings.ADMIN_PASSWORD == "Admin@12345":
            log.warning(
                "the admin account is using the default password - change "
                "ADMIN_PASSWORD in .env before any real deployment"
            )


def create_demo_data() -> None:
    """Three owners, each with one registered vehicle."""
    with session_scope() as db:
        for entry in DEMO_OWNERS:
            vehicle_spec = entry.pop("vehicle")
            owner = crud.get_user_by_email(db, entry["email"])
            if owner is None:
                owner = crud.create_user(
                    db, schemas.UserCreate(**entry, role=UserRole.OWNER)
                )
            entry["vehicle"] = vehicle_spec       # keep DEMO_OWNERS reusable

            if crud.get_vehicle_by_plate(db, vehicle_spec["plate_display"]) is None:
                crud.create_vehicle(
                    db, schemas.VehicleCreate(owner_id=owner.id, **vehicle_spec)
                )
        log.info("demo data ready (%d owners)", len(DEMO_OWNERS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the VTDS database")
    parser.add_argument(
        "--demo", action="store_true", help="insert demo owners and vehicles"
    )
    args = parser.parse_args()

    setup_logging(settings.LOG_LEVEL, settings.logs_dir)
    init_db()
    create_admin()
    if args.demo:
        create_demo_data()

    with session_scope() as db:
        stats = crud.dashboard_stats(db)
    print("\n  Database initialised")
    print(f"     owners   : {stats.total_owners}")
    print(f"     vehicles : {stats.total_vehicles}")
    print(f"\n  Login with  {settings.ADMIN_EMAIL}  /  {settings.ADMIN_PASSWORD}\n")


if __name__ == "__main__":
    main()
