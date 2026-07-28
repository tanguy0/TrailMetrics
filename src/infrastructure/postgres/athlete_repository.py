"""Postgres :class:`AthleteRepository`, with Strava tokens encrypted at rest.

Tokens are sealed with Fernet before they ever reach SQL, so a database dump — or
a Supabase dashboard session — cannot be replayed against Strava. The key comes
from configuration and never leaves this process.
"""

from datetime import date, datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from src.domain.ports.storage import (
    Athlete,
    AthleteRepository,
    StravaCredentials,
    SyncState,
)
from src.infrastructure.postgres.pool import Database


class PostgresAthleteRepository(AthleteRepository):
    def __init__(self, db: Database, encryption_key: str):
        self.db = db
        self._fernet = Fernet(encryption_key)

    # --- Athlete -----------------------------------------------------------

    def upsert(self, athlete: Athlete) -> Athlete:
        # Weight is intentionally not overwritten: the athlete sets it in this app,
        # and re-authenticating with Strava must not wipe it. Birthdate and height
        # are not in the insert list at all — they only ever come from this app, so
        # a re-auth has nothing to say about them.
        row = self.db.fetch_one(
            """
            insert into athletes (id, firstname, lastname, profile_url, weight_kg)
            values (%s, %s, %s, %s, %s)
            on conflict (id) do update set
                firstname = excluded.firstname,
                lastname = excluded.lastname,
                profile_url = excluded.profile_url,
                updated_at = now()
            returning id, firstname, lastname, profile_url, weight_kg,
                      birthdate, height_cm
            """,
            (athlete.id, athlete.firstname, athlete.lastname,
             athlete.profile_url, athlete.weight_kg),
        )
        return _athlete(row)

    def get(self, athlete_id: int) -> Optional[Athlete]:
        row = self.db.fetch_one(
            "select id, firstname, lastname, profile_url, weight_kg, "
            "birthdate, height_cm from athletes where id = %s",
            (athlete_id,),
        )
        return _athlete(row) if row else None

    def set_weight(self, athlete_id: int, weight_kg: Optional[float]) -> None:
        self.db.execute(
            "update athletes set weight_kg = %s, updated_at = now() where id = %s",
            (weight_kg, athlete_id),
        )

    def set_body(
        self,
        athlete_id: int,
        birthdate: Optional[date],
        height_cm: Optional[float],
    ) -> None:
        self.db.execute(
            "update athletes set birthdate = %s, height_cm = %s, "
            "updated_at = now() where id = %s",
            (birthdate, height_cm, athlete_id),
        )

    # --- Credentials -------------------------------------------------------

    def save_credentials(self, athlete_id: int, credentials: StravaCredentials) -> None:
        self.db.execute(
            """
            insert into strava_credentials
                (athlete_id, access_token_enc, refresh_token_enc, expires_at, scope)
            values (%s, %s, %s, %s, %s)
            on conflict (athlete_id) do update set
                access_token_enc = excluded.access_token_enc,
                refresh_token_enc = excluded.refresh_token_enc,
                expires_at = excluded.expires_at,
                scope = excluded.scope,
                updated_at = now()
            """,
            (
                athlete_id,
                self._fernet.encrypt(credentials.access_token.encode()),
                self._fernet.encrypt(credentials.refresh_token.encode()),
                credentials.expires_at,
                credentials.scope,
            ),
        )

    def get_credentials(self, athlete_id: int) -> Optional[StravaCredentials]:
        row = self.db.fetch_one(
            "select access_token_enc, refresh_token_enc, expires_at, scope "
            "from strava_credentials where athlete_id = %s",
            (athlete_id,),
        )
        if row is None:
            return None
        try:
            access = self._fernet.decrypt(bytes(row["access_token_enc"])).decode()
            refresh = self._fernet.decrypt(bytes(row["refresh_token_enc"])).decode()
        except InvalidToken:
            # The encryption key changed; the athlete has to reconnect Strava.
            return None
        return StravaCredentials(
            access_token=access,
            refresh_token=refresh,
            expires_at=_aware(row["expires_at"]),
            scope=row["scope"] or "",
        )

    # --- Sync state --------------------------------------------------------

    def get_sync_state(self, athlete_id: int) -> SyncState:
        row = self.db.fetch_one(
            "select status, done, total, message, last_synced_at, updated_at "
            "from sync_state where athlete_id = %s",
            (athlete_id,),
        )
        if row is None:
            return SyncState()
        return SyncState(
            status=row["status"],
            done=row["done"],
            total=row["total"],
            message=row["message"] or "",
            updated_at=row["updated_at"],
            last_synced_at=row["last_synced_at"],
        )

    def set_sync_state(self, athlete_id: int, state: SyncState) -> None:
        self.db.execute(
            """
            insert into sync_state
                (athlete_id, status, done, total, message, last_synced_at, updated_at)
            values (%s, %s, %s, %s, %s, %s, now())
            on conflict (athlete_id) do update set
                status = excluded.status,
                done = excluded.done,
                total = excluded.total,
                message = excluded.message,
                last_synced_at = coalesce(excluded.last_synced_at,
                                          sync_state.last_synced_at),
                updated_at = now()
            """,
            (athlete_id, state.status, state.done, state.total,
             state.message, state.last_synced_at),
        )


def _athlete(row) -> Athlete:
    return Athlete(
        id=int(row["id"]),
        firstname=row["firstname"] or "",
        lastname=row["lastname"] or "",
        profile_url=row["profile_url"],
        weight_kg=row["weight_kg"],
        birthdate=row["birthdate"],
        height_cm=row["height_cm"],
    )


def _aware(value: datetime) -> datetime:
    """Postgres timestamptz comes back aware; be defensive for other drivers."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
