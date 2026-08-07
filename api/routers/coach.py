"""The coach's athlete roster.

A coach is just an athlete id listed in ``COACH_ATHLETE_IDS``. Everything about
actually *viewing* another athlete's account happens for free: ``current_athlete_id``
(api/deps.py) resolves to the athlete named by the ``X-View-As-Athlete-Id`` header
instead of the signed-in coach's own id, so every endpoint already scoped to
``athlete.id`` picks it up with no change. This router only answers "who can I
switch to."
"""

from fastapi import APIRouter, Depends

from api.deps import get_athlete_repository, require_coach
from api.serialization import athlete_summary_payload

router = APIRouter(prefix="/coach", tags=["coach"])


@router.get("/athletes")
def list_athletes(_: int = Depends(require_coach)) -> dict:
    return {
        "athletes": [
            athlete_summary_payload(athlete)
            for athlete in get_athlete_repository().list_all()
        ]
    }
