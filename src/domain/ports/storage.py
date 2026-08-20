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
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Athlete:
    """An account, keyed by Strava's athlete id.

    ``birthdate``, ``height_cm`` and ``email`` are self-reported: Strava's API
    carries none of them, under any scope. Storing a birthdate rather than an age
    keeps the derived value correct without anyone re-typing it every year.
    """

    id: int
    firstname: str = ""
    lastname: str = ""
    weight_kg: Optional[float] = None
    profile_url: Optional[str] = None
    birthdate: Optional[date] = None
    height_cm: Optional[float] = None
    # Asked for once, right after the first sign-in. ``None`` for an account that has
    # not answered yet, which is what the app's first-run prompt keys on.
    email: Optional[str] = None
    # Self-reported training zones and VMA pace (seconds per km). Purely a
    # reference the athlete writes down for themselves — nothing here feeds any
    # computation, so there is no cross-field validation between them.
    hr_zone1_end: Optional[int] = None
    hr_zone2_end: Optional[int] = None
    hr_zone3_end: Optional[int] = None
    hr_zone4_end: Optional[int] = None
    hr_max: Optional[int] = None
    vma_pace_s_per_km: Optional[float] = None
    # The UI language the athlete has chosen. Always set — see the schema
    # column's comment for why this, unlike the fields above, is never `None`.
    lang: str = "en"

    @property
    def display_name(self) -> str:
        name = f"{self.firstname} {self.lastname}".strip()
        return name or f"Athlete {self.id}"

    @property
    def needs_email(self) -> bool:
        return not (self.email or "").strip()

    def age_on(self, today: date) -> Optional[int]:
        """Completed years as of ``today``, or ``None`` when no birthdate is set."""
        if self.birthdate is None:
            return None
        born = self.birthdate
        had_birthday = (today.month, today.day) >= (born.month, born.day)
        return today.year - born.year - (0 if had_birthday else 1)


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
    def list_all(self) -> List[Athlete]:
        """Every athlete — the coach roster's source, nothing else reads this wide."""

    @abstractmethod
    def set_weight(self, athlete_id: int, weight_kg: Optional[float]) -> None:
        ...

    @abstractmethod
    def set_body(
        self,
        athlete_id: int,
        birthdate: Optional[date],
        height_cm: Optional[float],
    ) -> None:
        """Set the self-reported fields Strava does not provide."""

    @abstractmethod
    def set_email(self, athlete_id: int, email: Optional[str]) -> None:
        """Set the athlete's email address, which Strava never provides."""

    @abstractmethod
    def set_lang(self, athlete_id: int, lang: str) -> None:
        """Set the athlete's preferred UI language."""

    @abstractmethod
    def set_zones(
        self,
        athlete_id: int,
        hr_zone1_end: Optional[int],
        hr_zone2_end: Optional[int],
        hr_zone3_end: Optional[int],
        hr_zone4_end: Optional[int],
        hr_max: Optional[int],
        vma_pace_s_per_km: Optional[float],
    ) -> None:
        """Set the athlete's self-reported training zones and VMA pace."""

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
    def set_relative_efforts(
        self, athlete_id: int, values: Sequence[Tuple[int, Optional[float]]]
    ) -> int:
        """Update Strava's Relative Effort on rows that already exist.

        Separate from :meth:`upsert_rows` because it is a *reported* value that
        arrives with the activity list, not a computed one: refreshing it for a whole
        history needs no per-activity request, which is what makes backfilling it
        affordable.
        """

    @abstractmethod
    def stream_object(self, athlete_id: int, activity_id: int) -> Optional[str]:
        ...

    @abstractmethod
    def route_polyline(self, athlete_id: int, activity_id: int) -> Optional[str]:
        """The activity's encoded route, or ``None`` if it has none stored."""

    @abstractmethod
    def set_rpe_feeling(
        self, athlete_id: int, activity_id: int, *,
        rpe: Optional[int] = None, feeling: Optional[str] = None,
    ) -> None:
        """Athlete-entered RPE (1-10) and/or feeling ('faible'/'ok'/'fort').

        A field left as ``None`` is left untouched — each is set independently by
        its own tag in the UI, and neither is ever synced from Strava.
        """

    @abstractmethod
    def set_route_polyline(
        self, athlete_id: int, activity_id: int, polyline: Optional[str]
    ) -> None:
        """Record a route fetched after the activity was first imported."""


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
