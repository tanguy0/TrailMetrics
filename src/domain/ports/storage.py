"""Ports for persisting an athlete, their activities and their raw streams.

Three interfaces, split by how the data behaves rather than by convenience:

* :class:`AthleteRepository` — the account and its Strava credentials. Tokens are
  written and read through here so that encryption lives in exactly one adapter.
* :class:`ActivityRepository` — the small, queryable feature rows. Every
  activity-level plot ultimately reads these.
* :class:`StreamStore` — the large per-second arrays. Separate because they are
  blobs: written once, read rarely, and far too big to sit next to the rows.

Splitting the last two is the whole reason the app stays cheap. A year of trends
touches only :class:`ActivityRepository`; the streams are only pulled when
something genuinely needs per-second data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Athlete:
    """An account, keyed by Strava's athlete id."""

    id: int
    firstname: str = ""
    lastname: str = ""
    weight_kg: Optional[float] = None
    profile_url: Optional[str] = None

    @property
    def display_name(self) -> str:
        name = f"{self.firstname} {self.lastname}".strip()
        return name or f"Athlete {self.id}"


@dataclass
class StravaCredentials:
    """A Strava token pair and when the access token stops working."""

    access_token: str
    refresh_token: str
    expires_at: datetime
    scope: str = ""

    def is_expired(self, now: datetime, leeway_seconds: int = 120) -> bool:
        """Treat a token as expired slightly early, so a refresh never races a call."""
        return (self.expires_at - now).total_seconds() <= leeway_seconds


@dataclass
class SyncState:
    """Progress of an athlete's activity import, for the UI to poll."""

    status: str = "idle"  # idle | running | error | done
    done: int = 0
    total: int = 0
    message: str = ""
    updated_at: Optional[datetime] = None
    last_synced_at: Optional[datetime] = None


class AthleteRepository(ABC):
    @abstractmethod
    def upsert(self, athlete: Athlete) -> Athlete:
        ...

    @abstractmethod
    def get(self, athlete_id: int) -> Optional[Athlete]:
        ...

    @abstractmethod
    def set_weight(self, athlete_id: int, weight_kg: Optional[float]) -> None:
        ...

    @abstractmethod
    def save_credentials(self, athlete_id: int, credentials: StravaCredentials) -> None:
        ...

    @abstractmethod
    def get_credentials(self, athlete_id: int) -> Optional[StravaCredentials]:
        ...

    @abstractmethod
    def get_sync_state(self, athlete_id: int) -> SyncState:
        ...

    @abstractmethod
    def set_sync_state(self, athlete_id: int, state: SyncState) -> None:
        ...


class ActivityRepository(ABC):
    """Stores one feature row per activity, per athlete."""

    @abstractmethod
    def upsert_rows(self, athlete_id: int, rows: Sequence[Dict[str, Any]]) -> int:
        """Insert or replace feature rows. Returns how many were written."""

    @abstractmethod
    def summaries(self, athlete_id: int) -> List[Dict[str, Any]]:
        """Selection-level fields for every activity, oldest first."""

    @abstractmethod
    def rows(
        self, athlete_id: int, activity_ids: Optional[Sequence[int]] = None
    ) -> List[Dict[str, Any]]:
        """Full feature rows; every activity when ``activity_ids`` is ``None``."""

    @abstractmethod
    def known_ids(self, athlete_id: int) -> set:
        """Activity ids already stored — so a sync only fetches what is new."""

    @abstractmethod
    def date_range(self, athlete_id: int) -> Optional[Tuple[datetime, datetime]]:
        ...

    @abstractmethod
    def set_stream_object(
        self, athlete_id: int, activity_id: int, object_path: Optional[str]
    ) -> None:
        """Record where an activity's raw streams were stored."""

    @abstractmethod
    def stream_object(self, athlete_id: int, activity_id: int) -> Optional[str]:
        ...


class StreamStore(ABC):
    """Blob storage for per-second arrays, one object per activity."""

    @abstractmethod
    def put(self, athlete_id: int, activity_id: int, payload: bytes) -> str:
        """Store the encoded streams; returns the object path."""

    @abstractmethod
    def get(self, path: str) -> Optional[bytes]:
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        ...
