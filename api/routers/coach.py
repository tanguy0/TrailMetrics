"""The coach's athlete roster — the rail's "Athlete" switcher.

A coach is an account whose ``role`` is coach or master, and its roster is the
athletes it coaches (``coaching``, design/specs/coaching.md) — not every
athlete. Viewing one happens for free: ``current_athlete_id`` (api/deps.py)
resolves to the athlete named by the ``X-View-As-Athlete-Id`` header, after
checking the same table, so every endpoint scoped to ``athlete.id`` picks it up.
"""

from fastapi import APIRouter, Depends

from api.deps import get_coaching_repository, require_coach
from src.domain.ports.accounts import Account

router = APIRouter(prefix="/coach", tags=["coach"])


@router.get("/athletes")
def list_athletes(coach: Account = Depends(require_coach)) -> dict:
    """The coached athletes that can be viewed — those with Strava attached."""
    return {
        "athletes": [
            {
                "id": row["athlete_id"],
                "display_name": row["display_name"],
                "profile_url": row["profile_url"],
            }
            for row in get_coaching_repository().coached_by(coach.id)
            if row["athlete_id"] is not None
        ]
    }
