"""
API package - FastAPI routers.  *Interfaces only at this stage.*

Planned routers (Step 6):

    deps.py       shared dependencies: current user, require_admin
    auth.py       POST /login, POST /register, GET/PATCH /me
    users.py      owner management (admin only)
    vehicles.py   vehicle CRUD + image upload
    faces.py      owner face enrolment
    detections.py detection log queries
    alerts.py     alert history, acknowledge / resolve
    stream.py     MJPEG live feed, pipeline start/stop
    stats.py      dashboard aggregates
"""
