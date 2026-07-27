"""Strava OAuth: exchange a code, refresh an access token, build a client.

All Strava credential handling funnels through here so that expiry is dealt with
in exactly one place. Strava access tokens last six hours, and a sync over a long
history easily outlives that, so every client is built from a *freshly checked*
token rather than whatever was stored at login.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from stravalib import Client

from src.domain.ports.storage import (
    Athlete,
    AthleteRepository,
    StravaCredentials,
)

logger = logging.getLogger(__name__)

# Reading private / followers-only activities needs activity:read_all; the plain
# activity:read scope silently returns only public ones.
SCOPES = ["read", "activity:read_all"]


class StravaTokenService:
    """Owns the OAuth dance and keeps stored credentials usable."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        athletes: AthleteRepository,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.athletes = athletes

    def authorization_url(self, redirect_uri: str, state: str = "") -> str:
        return Client().authorization_url(
            client_id=int(self.client_id),
            redirect_uri=redirect_uri,
            scope=SCOPES,
            state=state or None,
            approval_prompt="auto",
        )

    def exchange_code(self, code: str) -> Tuple[Athlete, StravaCredentials]:
        """Trade an authorization code for tokens, and identify the athlete.

        Persists both, so a caller only has to remember the athlete id.
        """
        client = Client()
        response = client.exchange_code_for_token(
            client_id=int(self.client_id),
            client_secret=self.client_secret,
            code=code,
        )
        credentials = _credentials(response)

        client = Client(access_token=credentials.access_token)
        profile = client.get_athlete()
        athlete = Athlete(
            id=int(profile.id),
            firstname=profile.firstname or "",
            lastname=profile.lastname or "",
            profile_url=getattr(profile, "profile_medium", None),
            # Strava exposes a weight for some athletes; a good starting default,
            # and the app lets it be overridden.
            weight_kg=_weight(profile),
        )
        stored = self.athletes.upsert(athlete)
        self.athletes.save_credentials(stored.id, credentials)
        if stored.weight_kg is None and athlete.weight_kg:
            self.athletes.set_weight(stored.id, athlete.weight_kg)
            stored.weight_kg = athlete.weight_kg
        return stored, credentials

    def valid_credentials(self, athlete_id: int) -> Optional[StravaCredentials]:
        """Stored credentials, refreshed first if the access token is due to expire."""
        credentials = self.athletes.get_credentials(athlete_id)
        if credentials is None:
            return None
        if not credentials.is_expired(datetime.now(timezone.utc)):
            return credentials

        try:
            response = Client().refresh_access_token(
                client_id=int(self.client_id),
                client_secret=self.client_secret,
                refresh_token=credentials.refresh_token,
            )
        except Exception as error:
            # A revoked authorization lands here; the athlete must reconnect.
            logger.warning("token refresh failed for athlete %s: %s", athlete_id, error)
            return None

        refreshed = _credentials(response, fallback_scope=credentials.scope)
        self.athletes.save_credentials(athlete_id, refreshed)
        return refreshed

    def client_for(self, athlete_id: int) -> Optional[Client]:
        """An authenticated ``stravalib`` client, or ``None`` if not connected."""
        credentials = self.valid_credentials(athlete_id)
        if credentials is None:
            return None
        return Client(access_token=credentials.access_token)


def _credentials(response, fallback_scope: str = "") -> StravaCredentials:
    """Normalize stravalib's token response (dict-like) into our value object."""
    expires_at = response.get("expires_at")
    if isinstance(expires_at, (int, float)):
        expiry = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    elif isinstance(expires_at, datetime):
        expiry = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
    else:
        # Strava's documented lifetime, used only if the field is missing.
        expiry = datetime.now(timezone.utc) + timedelta(hours=6)
    return StravaCredentials(
        access_token=response["access_token"],
        refresh_token=response["refresh_token"],
        expires_at=expiry,
        scope=" ".join(SCOPES) if not fallback_scope else fallback_scope,
    )


def _weight(profile) -> Optional[float]:
    value = getattr(profile, "weight", None)
    value = getattr(value, "magnitude", value)
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return None
    return weight if 25.0 < weight < 250.0 else None
