"""Strava OAuth and the athlete's own profile.

The browser never sees a Strava token. The flow is:

1. web app asks for an authorize URL and redirects the user to Strava;
2. Strava redirects back to the **web app**, which posts the code here
   server-to-server with the shared service token;
3. this API exchanges it, stores the (encrypted) Strava tokens and returns a
   session token that the web app puts in a first-party ``httpOnly`` cookie.

Landing the callback on the web app rather than here is what keeps the session
cookie first-party, so it survives browser third-party-cookie restrictions without
any CORS credential juggling.
"""

import threading
import time
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.config import get_settings
from api.deps import (
    current_athlete,
    get_activity_repository,
    get_athlete_repository,
    get_token_service,
    invalidate_caches,
    language,
    require_service_token,
)
from api.security import create_session_token
from api.serialization import athlete_payload
from src.domain.ports.storage import Athlete

router = APIRouter(prefix="/auth", tags=["auth"])

# A Strava authorization code can be exchanged exactly once, but the callback can
# easily arrive twice — a double-clicked button, a browser retrying a redirect, a
# platform replaying the request. The second call would then fail and *that* is the
# response the user sees, even though the first one signed them in. Replaying the
# session token for a code we just exchanged makes the callback idempotent.
#
# Deliberately in-process and short-lived: the retry we care about lands within
# seconds, on the same worker. Across replicas the duplicate still fails and the
# user retries the login, which is the status quo, not a regression.
_RECENT_EXCHANGE_TTL_S = 120.0
_recent_exchanges: Dict[str, Tuple[float, dict]] = {}
_recent_lock = threading.Lock()


def _remember_exchange(code: str, response: dict) -> None:
    now = time.monotonic()
    with _recent_lock:
        _recent_exchanges[code] = (now, response)
        stale = [k for k, (at, _) in _recent_exchanges.items()
                 if now - at > _RECENT_EXCHANGE_TTL_S]
        for key in stale:
            del _recent_exchanges[key]


def _replay_exchange(code: str) -> Optional[dict]:
    with _recent_lock:
        entry = _recent_exchanges.get(code)
        if entry is None:
            return None
        at, response = entry
        if time.monotonic() - at > _RECENT_EXCHANGE_TTL_S:
            del _recent_exchanges[code]
            return None
    return response


class AuthorizeUrlRequest(BaseModel):
    redirect_uri: str = Field(min_length=1, max_length=500)
    state: str = ""


class ExchangeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=500)


class WeightUpdate(BaseModel):
    # Wide but sane bounds; power is unmodellable from a nonsense weight.
    weight_kg: Optional[float] = Field(default=None, ge=25, le=250)


@router.post("/strava/url")
def authorize_url(payload: AuthorizeUrlRequest) -> dict:
    """The Strava consent URL to send the user to."""
    service = get_token_service()
    return {"url": service.authorization_url(payload.redirect_uri, payload.state)}


@router.post("/strava/exchange", dependencies=[Depends(require_service_token)])
def exchange(payload: ExchangeRequest) -> dict:
    """Trade an authorization code for a session. Service-to-service only."""
    settings = get_settings()
    missing = settings.missing_for_auth()
    if missing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server not configured for login; missing: {', '.join(missing)}",
        )

    replayed = _replay_exchange(payload.code)
    if replayed is not None:
        return replayed

    service = get_token_service()
    try:
        athlete, _ = service.exchange_code(payload.code)
    except Exception as error:
        # An expired code, or one whose exchange we no longer remember; nothing
        # actionable for the client beyond starting again.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Could not complete Strava authorization: {error}",
        )

    token = create_session_token(
        athlete.id, settings.session_secret, settings.session_ttl_days
    )
    response = {
        "session_token": token,
        "expires_in_days": settings.session_ttl_days,
        "athlete": {"id": athlete.id, "display_name": athlete.display_name},
    }
    _remember_exchange(payload.code, response)
    return response


@router.get("/me")
def me(athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)) -> dict:
    """The signed-in athlete, their data range and any sync in progress."""
    return _me_payload(athlete)


@router.patch("/me")
def update_me(
    payload: WeightUpdate = Body(...),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Set the body weight that unlocks the power metrics.

    Stored power is per-kilogram, so this takes effect immediately across the whole
    history — no recomputation. The cached plot outputs do have to go, since their
    numbers were scaled with the old weight.
    """
    get_athlete_repository().set_weight(athlete.id, payload.weight_kg)
    invalidate_caches(athlete.id)
    athlete.weight_kg = payload.weight_kg
    return _me_payload(athlete)


@router.post("/logout")
def logout() -> dict:
    """Nothing to do server-side — the web app drops the cookie.

    Sessions are stateless by design; revocation would need a token store, which
    is not worth it until there is a reason to revoke.
    """
    return {"ok": True}


def _me_payload(athlete: Athlete) -> dict:
    activities = get_activity_repository()
    summaries = activities.summaries(athlete.id)
    return athlete_payload(
        athlete,
        sync=get_athlete_repository().get_sync_state(athlete.id),
        date_range=activities.date_range(athlete.id),
        activity_count=len(summaries),
        sport_types=sorted({row["sport_type"] for row in summaries}),
    )
