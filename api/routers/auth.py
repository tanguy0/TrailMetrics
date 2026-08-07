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
from datetime import date
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
    session_context,
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


# Deliberately permissive: "something@something.something", no dots-in-local-part
# rules, no TLD list. A stricter pattern rejects real addresses, and the only thing
# this validation can honestly promise is that the value is shaped like an email —
# whether it *works* is a question only a sent message answers.
_EMAIL_PATTERN = r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$"


class ProfileUpdate(BaseModel):
    """A partial update of the athlete's own self-reported fields.

    Every field is optional *and* nullable, which are different things here: an
    absent key leaves the stored value alone, an explicit ``null`` clears it. That
    distinction is what lets one endpoint back several independently-edited widgets
    without them overwriting each other.
    """

    # Wide but sane bounds; power is unmodellable from a nonsense weight.
    weight_kg: Optional[float] = Field(default=None, ge=25, le=250)
    birthdate: Optional[date] = None
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    email: Optional[str] = Field(default=None, max_length=254, pattern=_EMAIL_PATTERN)

    # Self-reported zones and VMA pace — display-only, see the module docstring
    # on `Athlete` for why there's no cross-field validation between them.
    hr_zone1_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone2_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone3_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone4_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_max: Optional[int] = Field(default=None, ge=30, le=250)
    vma_pace_s_per_km: Optional[float] = Field(default=None, ge=90, le=900)

    model_config = {"extra": "forbid"}


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
        # Strava returns no email address, so the app has to ask. Reported here
        # rather than discovered later so the web app can route a brand-new athlete
        # straight to the question instead of letting them wander past it.
        "needs_email": athlete.needs_email,
    }
    _remember_exchange(payload.code, response)
    return response


@router.get("/me")
def me(
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
    session: dict = Depends(session_context),
) -> dict:
    """The signed-in athlete — or, for a coach viewing another one, that athlete."""
    return _me_payload(athlete, session)


@router.patch("/me")
def update_me(
    payload: ProfileUpdate = Body(...),
    athlete: Athlete = Depends(current_athlete),
    session: dict = Depends(session_context),
) -> dict:
    """Update the athlete's self-reported body fields.

    Weight is the one with computational consequences: stored power is
    per-kilogram, so a new weight takes effect across the whole history with no
    recomputation — but the cached plot outputs have to go, since their numbers were
    scaled with the old value. Birthdate and height feed no metric, so they leave
    the cache alone.
    """
    athletes = get_athlete_repository()
    touched = payload.model_dump(exclude_unset=True)

    if "weight_kg" in touched:
        athletes.set_weight(athlete.id, payload.weight_kg)
        athlete.weight_kg = payload.weight_kg
        invalidate_caches(athlete.id)

    if "birthdate" in touched or "height_cm" in touched:
        # A partial update must not blank the field the client didn't mention.
        birthdate = payload.birthdate if "birthdate" in touched else athlete.birthdate
        height_cm = payload.height_cm if "height_cm" in touched else athlete.height_cm
        athletes.set_body(athlete.id, birthdate, height_cm)
        athlete.birthdate = birthdate
        athlete.height_cm = height_cm

    if "email" in touched:
        athletes.set_email(athlete.id, payload.email)
        athlete.email = payload.email

    zone_fields = (
        "hr_zone1_end", "hr_zone2_end", "hr_zone3_end", "hr_zone4_end",
        "hr_max", "vma_pace_s_per_km",
    )
    if touched.keys() & set(zone_fields):
        # A partial update must not blank a zone the client didn't mention.
        values = {
            field: getattr(payload, field) if field in touched else getattr(athlete, field)
            for field in zone_fields
        }
        athletes.set_zones(athlete.id, **values)
        for field, value in values.items():
            setattr(athlete, field, value)

    return _me_payload(athlete, session)


@router.post("/logout")
def logout() -> dict:
    """Nothing to do server-side — the web app drops the cookie.

    Sessions are stateless by design; revocation would need a token store, which
    is not worth it until there is a reason to revoke.
    """
    return {"ok": True}


def _me_payload(athlete: Athlete, session: dict) -> dict:
    activities = get_activity_repository()
    summaries = activities.summaries(athlete.id)
    payload = athlete_payload(
        athlete,
        sync=get_athlete_repository().get_sync_state(athlete.id),
        date_range=activities.date_range(athlete.id),
        activity_count=len(summaries),
        sport_types=sorted({row["sport_type"] for row in summaries}),
    )
    payload.update(session)
    return payload
