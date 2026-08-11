"""Import an athlete's Strava activities: fetch, featurize, store.

The one write path into the database. For each activity it does the expensive work
exactly once — the per-second pass that produces a feature row — then keeps the
row in Postgres and the raw arrays in object storage.

Two properties make it usable on a real history:

* **Incremental.** Activities already stored are skipped, so the first sync is
  long and every later one is short. Strava's rate limit (100 requests / 15 min)
  makes this non-optional.
* **Resumable.** Rows are written in batches as it goes, so a sync that dies
  halfway leaves everything it had already fetched. Re-running continues.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence

from src.domain.dataset.features import build_activity_features
from src.domain.dataset.sport import CYCLING_SPORT_TYPES, RUNNING_SPORT_TYPES
from src.domain.models.activity import ActivityStream
from src.domain.ports.storage import (
    ActivityRepository,
    AthleteRepository,
    StreamStore,
    SyncState,
)
from src.infrastructure.storage.codec import encode_stream
from src.infrastructure.strava.strava_client import StravaClient
from src.usecases.base import UseCase

logger = logging.getLogger(__name__)

# Sport types imported: every running and cycling variant known to
# src.domain.dataset.sport. Everything else on Strava is neither and stays out.
# Derived rather than listed here again, so this can't drift from the family
# definitions the rest of the app classifies activities by.
DEFAULT_SPORT_TYPES = sorted(RUNNING_SPORT_TYPES | CYCLING_SPORT_TYPES)

# Rows are flushed this often, so a crash costs at most this much work.
BATCH_SIZE = 25


@dataclass
class SyncAthleteActivitiesInput:
    athlete_id: int
    sport_types: List[str] = field(default_factory=lambda: list(DEFAULT_SPORT_TYPES))
    # None = the athlete's whole history.
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    # Re-fetch and recompute activities already stored (after a feature change).
    force: bool = False
    max_activities: Optional[int] = None


@dataclass
class SyncAthleteActivitiesOutput:
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    total_seen: int = 0
    # Already-stored activities whose Relative Effort was refreshed from the listing.
    refreshed: int = 0


class SyncAthleteActivities(UseCase):
    """Fetch new activities from Strava and persist their features and streams."""

    def __init__(
        self,
        strava: StravaClient,
        activities: ActivityRepository,
        streams: StreamStore,
        athletes: Optional[AthleteRepository] = None,
    ):
        self.strava = strava
        self.activities = activities
        self.streams = streams
        self.athletes = athletes
        # Stream paths queued for the next flush: rows must exist before they can
        # be updated with a blob location.
        self._pending_objects: List[tuple] = []

    def execute(
        self,
        params: SyncAthleteActivitiesInput,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SyncAthleteActivitiesOutput:
        athlete_id = params.athlete_id
        result = SyncAthleteActivitiesOutput()
        # Cycling's modelled power (src/domain/cycling/power.py) isn't linear in
        # mass the way running's is, so it can't be cached per-kg and rescaled on
        # read — it's computed once here, against whatever weight is on file right
        # now. A weight change later needs a resync to refresh past rides, same as
        # any other cached feature.
        mass_kg = None
        if self.athletes is not None:
            athlete = self.athletes.get(athlete_id)
            mass_kg = athlete.weight_kg if athlete else None

        self._report(athlete_id, SyncState(
            status="running", message="listing activities",
        ))

        try:
            listed = self.strava.list_activities(
                sport_types=params.sport_types,
                from_date=params.from_date,
                to_date=params.to_date,
                # Volume and record trends don't need heart rate, so keep
                # everything and let each plot decide what it can use.
                require_heartrate=False,
            )
        except Exception as error:
            self._report(athlete_id, SyncState(status="error", message=str(error)[:400]))
            raise

        known = set() if params.force else self.activities.known_ids(athlete_id)
        pending = [act for act in listed if int(act["id"]) not in known]
        if params.max_activities is not None:
            pending = pending[: params.max_activities]

        result.total_seen = len(listed)
        result.skipped = len(listed) - len(pending)
        # Reported values ride on the listing we just walked, so refreshing them for
        # activities we already have is free — and it is what backfills a column
        # added after the history was first imported.
        result.refreshed = self._refresh_reported(athlete_id, listed, known)
        total = len(pending)
        self._report(athlete_id, SyncState(
            status="running", done=0, total=total, message="importing",
        ))

        batch: List[dict] = []
        for index, activity in enumerate(pending, start=1):
            try:
                if self._import_one(athlete_id, activity, batch, mass_kg):
                    result.imported += 1
                else:
                    result.failed += 1
            except Exception as error:
                # One bad activity must never abort a multi-year import.
                logger.warning("activity %s failed: %s", activity.get("id"), error)
                result.failed += 1

            if len(batch) >= BATCH_SIZE:
                self._flush(athlete_id, batch)
            if progress_callback:
                progress_callback(index, total)
            if index % 5 == 0 or index == total:
                self._report(athlete_id, SyncState(
                    status="running", done=index, total=total, message="importing",
                ))
            # Stay well inside Strava's short-term rate limit.
            time.sleep(self.strava.throttle_seconds)

        self._flush(athlete_id, batch)
        self._report(athlete_id, SyncState(
            status="done", done=total, total=total,
            message=f"imported {result.imported}, failed {result.failed}",
            last_synced_at=datetime.now(timezone.utc),
        ))
        return result

    # --- One activity ------------------------------------------------------

    def _import_one(
        self, athlete_id: int, activity: dict, batch: List[dict],
        mass_kg: Optional[float] = None,
    ) -> bool:
        """Fetch, featurize and queue one activity. Returns whether it produced a row."""
        activity_id = int(activity["id"])
        stream = self.strava.fetch_activity(activity)
        if stream is None:
            return False

        row = build_activity_features(stream, mass_kg=mass_kg)
        if row is None:
            return False
        # Route metadata rides along on the row rather than through the feature
        # builder: it is a string for drawing a map, not a quantity to aggregate,
        # so it has no place in the numeric feature frame.
        row["summary_polyline"] = activity.get("summary_polyline")
        batch.append(row)

        # Only activities that really have per-second data get a blob.
        if getattr(stream, "has_streams", True) and len(stream.time):
            try:
                path = self.streams.put(athlete_id, activity_id, encode_stream(stream))
                self._pending_objects.append((activity_id, path))
            except Exception as error:
                # A failed upload costs stream-level plots for this activity, not
                # the activity itself — the feature row is already good.
                logger.warning("stream upload failed for %s: %s", activity_id, error)
        return True

    # --- Reported values on already-stored activities ----------------------

    def _refresh_reported(
        self, athlete_id: int, listed: Sequence[dict], known: set
    ) -> int:
        """Update Relative Effort on activities we already have.

        Strava computes Relative Effort from the athlete's own heart-rate zones, so
        it can only be *read*, never derived here — and it can change after the fact
        (Strava recomputes it when zones are edited). Since it arrives on the listing
        this sync already walked, keeping it current costs one UPDATE and no API
        call, which is what makes it worth doing on every sync instead of only at
        import time.
        """
        values = [
            (int(act["id"]), act.get("relative_effort"))
            for act in listed
            if int(act["id"]) in known and act.get("relative_effort") is not None
        ]
        if not values:
            return 0
        try:
            return self.activities.set_relative_efforts(athlete_id, values)
        except Exception as error:
            # A cosmetic refresh must never fail an import.
            logger.warning("could not refresh relative effort: %s", error)
            return 0

    # --- Persistence -------------------------------------------------------

    def _flush(self, athlete_id: int, batch: List[dict]) -> None:
        if batch:
            self.activities.upsert_rows(athlete_id, batch)
            batch.clear()
        # Stream paths are set after the rows exist, since they update them.
        for activity_id, path in self._pending_objects:
            try:
                self.activities.set_stream_object(athlete_id, activity_id, path)
            except Exception as error:
                logger.warning("could not record stream path for %s: %s",
                               activity_id, error)
        self._pending_objects = []

    def _report(self, athlete_id: int, state: SyncState) -> None:
        if self.athletes is None:
            return
        try:
            self.athletes.set_sync_state(athlete_id, state)
        except Exception as error:
            logger.warning("could not update sync state: %s", error)
