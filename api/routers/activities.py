"""Activity listing and the Strava import.

The sync runs as a background task and reports progress through the ``sync_state``
table rather than the response, because a first import of a long history takes far
longer than any sensible HTTP timeout. The client starts it and polls.

That state lives in the database rather than in memory so progress survives a
container restart — on Railway that happens on every deploy.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.deps import (
    current_athlete,
    get_activity_repository,
    get_athlete_repository,
    get_stream_store,
    get_token_service,
    invalidate_caches,
)
from api.serialization import summary_payload
from src.domain.ports.storage import Athlete, SyncState
from src.infrastructure.strava.strava_client import StravaClient
from src.usecases.sync_athlete_activities import (
    DEFAULT_SPORT_TYPES,
    SyncAthleteActivities,
    SyncAthleteActivitiesInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activities", tags=["activities"])


class SyncRequest(BaseModel):
    sport_types: List[str] = Field(default_factory=lambda: list(DEFAULT_SPORT_TYPES))
    # Re-fetch activities already stored — needed after a feature-computation change.
    force: bool = False
    # Cap the import, mostly to try things out without spending the rate limit.
    max_activities: Optional[int] = Field(default=None, ge=1, le=5000)


@router.get("")
def list_activities(
    athlete: Athlete = Depends(current_athlete),
    sport_type: Optional[str] = Query(None),
    limit: int = Query(0, ge=0, le=10000),
) -> dict:
    """Selection-level fields for every stored activity, newest first.

    This is what the data-source editor's activity picker reads; it never touches
    per-second data.
    """
    from src.infrastructure.postgres.stored_activity_data import StoredActivityData

    data = StoredActivityData(
        athlete.id, get_activity_repository(), get_stream_store(),
        mass_kg=athlete.weight_kg,
    )
    summaries = list(reversed(data.summaries()))
    if sport_type:
        summaries = [s for s in summaries if s.sport_type == sport_type]
    if limit:
        summaries = summaries[:limit]
    return {
        "activities": [summary_payload(s) for s in summaries],
        "total": len(summaries),
    }


@router.get("/{activity_id}/route")
def activity_route(
    activity_id: int,
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """One activity's route as coordinates, for the map.

    The generalized form of ``GET /home/last-activity/route`` — that endpoint keeps
    its own name and shape (it also carries the "no activities at all" case), but
    both call the same :func:`resolve_activity_route`.
    """
    from api.routers.home import resolve_activity_route

    return resolve_activity_route(athlete, activity_id)


@router.get("/sync")
def sync_status(athlete: Athlete = Depends(current_athlete)) -> dict:
    state = get_athlete_repository().get_sync_state(athlete.id)
    return {
        "status": state.status,
        "done": state.done,
        "total": state.total,
        "message": state.message,
        "last_synced_at": state.last_synced_at.isoformat()
        if state.last_synced_at else None,
    }


@router.post("/sync", status_code=status.HTTP_202_ACCEPTED)
def start_sync(
    background: BackgroundTasks,
    payload: SyncRequest = SyncRequest(),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Kick off an import. Returns immediately; poll ``GET /activities/sync``."""
    athletes = get_athlete_repository()
    state = athletes.get_sync_state(athlete.id)
    if state.status == "running":
        # Two concurrent syncs would double-spend the Strava rate limit and race
        # on the same rows.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="A sync is already running for this athlete.",
        )

    client = get_token_service().client_for(athlete.id)
    if client is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Strava authorization has expired — please reconnect.",
        )

    # Mark it running before returning, so an immediate poll doesn't report idle.
    athletes.set_sync_state(athlete.id, SyncState(status="running", message="starting"))
    background.add_task(
        _run_sync, athlete.id, client, payload.sport_types,
        payload.force, payload.max_activities,
    )
    return {"status": "running"}


def _run_sync(
    athlete_id: int,
    stravalib_client,
    sport_types: List[str],
    force: bool,
    max_activities: Optional[int],
) -> None:
    athletes = get_athlete_repository()
    try:
        usecase = SyncAthleteActivities(
            strava=StravaClient(stravalib_client),
            activities=get_activity_repository(),
            streams=get_stream_store(),
            athletes=athletes,
        )
        result = usecase.execute(SyncAthleteActivitiesInput(
            athlete_id=athlete_id,
            sport_types=sport_types,
            force=force,
            max_activities=max_activities,
        ))
        logger.info("sync for %s: %s", athlete_id, result)
    except Exception as error:
        logger.exception("sync failed for athlete %s", athlete_id)
        athletes.set_sync_state(
            athlete_id, SyncState(status="error", message=str(error)[:400])
        )
    finally:
        # New activities invalidate every cached plot output for this athlete.
        invalidate_caches(athlete_id)
