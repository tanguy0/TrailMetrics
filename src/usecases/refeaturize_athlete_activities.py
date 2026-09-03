"""Rebuild stored feature rows from the stream blobs, without calling Strava.

A feature row is a pure function of an activity's per-second arrays and the
athlete's body mass, and those arrays are already in object storage. So when a
``FEATURE_VERSION`` bump changes *the maths* rather than *which streams are
read*, the whole history can be recomputed locally: read the blob, run
:func:`~src.domain.dataset.features.build_activity_features`, write the row back.

Why that is worth its own pass rather than letting the sync re-import:

* **No rate limit.** Re-importing spends one Strava request per activity against
  a 100-per-15-minutes budget, so a decade of running takes a long, visible
  sync. Reading blobs is bounded by object storage and runs in one go.
* **No Strava token needed.** A re-import cannot fix an athlete whose token has
  expired or who disconnected Strava — they would keep the old numbers forever.
  This pass does not care.

What it deliberately does *not* do is decide when a rebuild is valid. Two rules
bound that, both owned by the storage layer:
:data:`~src.infrastructure.postgres.activity_repository.MIN_LOCAL_REBUILD_VERSION`
says which blobs carry every stream the current featurizer reads, and anything
older, or with no blob at all, is left untouched at its old version so the
Strava path still picks it up as pending. This pass never makes a row *more*
stale, and re-running it is free once everything is current.

Rows are re-derived rather than patched, so a rebuild is the same computation an
import would do — with one difference worth knowing: blobs store float32 (see
:mod:`src.infrastructure.storage.codec`), so a rebuilt number is not bit-identical
to a freshly imported one. Measured on a 5,000-point run, the disagreement is
around 2e-9 relative — nine significant digits, i.e. tens of microns on a 16 km
distance — against sensors whose own noise is metres. It is not a reason to
prefer the Strava path.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.domain.dataset.features import build_activity_features
from src.domain.models.activity import ActivityStream
from src.domain.ports.storage import (
    ActivityRepository,
    AthleteRepository,
    StreamStore,
)
from src.infrastructure.postgres.activity_repository import MIN_LOCAL_REBUILD_VERSION
from src.infrastructure.storage.codec import decode_stream
from src.usecases.base import UseCase

logger = logging.getLogger(__name__)

# Rows are flushed this often, so an interrupted pass keeps what it finished —
# the same reasoning as the sync's batching, and the same size.
BATCH_SIZE = 25


@dataclass
class RefeaturizeAthleteActivitiesInput:
    athlete_id: int
    # Mostly for trying the pass out without rewriting a whole history.
    max_activities: Optional[int] = None


@dataclass
class RefeaturizeAthleteActivitiesOutput:
    # Activities found at an older feature_version.
    stale: int = 0
    # Rebuilt from their stream blob.
    rebuilt: int = 0
    # Streamless activities (manual entries): re-derived from the summary totals
    # already in their row, since that is all they ever held.
    summary_only: int = 0
    # Left at the old version, for the Strava path to re-import: no blob stored,
    # a blob too old to trust, or one that would not decode.
    needs_strava: int = 0
    failed: int = 0

    @property
    def done(self) -> int:
        return self.rebuilt + self.summary_only


class RefeaturizeAthleteActivities(UseCase):
    """Recompute out-of-date feature rows from stored streams."""

    def __init__(
        self,
        activities: ActivityRepository,
        streams: StreamStore,
        athletes: Optional[AthleteRepository] = None,
    ):
        self.activities = activities
        self.streams = streams
        self.athletes = athletes

    def execute(
        self,
        params: RefeaturizeAthleteActivitiesInput,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RefeaturizeAthleteActivitiesOutput:
        athlete_id = params.athlete_id
        result = RefeaturizeAthleteActivitiesOutput()

        stale = self.activities.stale_activities(athlete_id)
        if params.max_activities is not None:
            stale = stale[: params.max_activities]
        result.stale = len(stale)
        if not stale:
            return result

        # Cycling's modelled power is not linear in mass, so it is baked in at
        # featurize time against the weight on file — read once here, exactly as
        # the sync does. Rebuilding therefore also refreshes past rides against
        # the athlete's current weight, which an untouched row would not get.
        mass_kg = None
        if self.athletes is not None:
            athlete = self.athletes.get(athlete_id)
            mass_kg = athlete.weight_kg if athlete else None

        total = len(stale)
        batch: List[Dict[str, Any]] = []
        for index, activity in enumerate(stale, start=1):
            try:
                row = self._rebuild(athlete_id, activity, mass_kg, result)
                if row is not None:
                    batch.append(row)
            except Exception as error:
                # One unreadable activity must not cost the rest of the history;
                # it keeps its old feature_version and stays pending for Strava.
                logger.warning(
                    "could not rebuild activity %s: %s",
                    activity.get("activity_id"), error,
                )
                result.failed += 1

            if len(batch) >= BATCH_SIZE:
                self._flush(athlete_id, batch)
            if progress_callback:
                progress_callback(index, total)

        self._flush(athlete_id, batch)
        return result

    # --- One activity ------------------------------------------------------

    def _rebuild(
        self,
        athlete_id: int,
        activity: Dict[str, Any],
        mass_kg: Optional[float],
        result: RefeaturizeAthleteActivitiesOutput,
    ) -> Optional[Dict[str, Any]]:
        """The new row for one stale activity, or ``None`` if it needs Strava."""
        # A streamless activity's row is built entirely from Strava's summary
        # totals, which are in the row already — so it can be re-derived through
        # the same featurizer as everything else rather than special-cased, or
        # worse, re-stamped on the assumption that nothing about it changed.
        if not activity["has_streams"]:
            row = build_activity_features(_summary_stream(activity))
            if row is None:
                result.needs_strava += 1
                return None
            result.summary_only += 1
            return row

        if activity["feature_version"] < MIN_LOCAL_REBUILD_VERSION:
            result.needs_strava += 1
            return None
        path = activity["stream_object"]
        if not path:
            result.needs_strava += 1
            return None

        payload = self.streams.get(path)
        stream = decode_stream(payload) if payload is not None else None
        if stream is None:
            result.needs_strava += 1
            return None

        row = build_activity_features(stream, mass_kg=mass_kg)
        if row is None:
            result.needs_strava += 1
            return None
        result.rebuilt += 1
        return row

    def _flush(self, athlete_id: int, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            # No polyline and no stream path in these rows: `upsert_rows`
            # coalesces the first and leaves the second alone, which is what
            # makes a re-featurize non-destructive.
            self.activities.upsert_rows(athlete_id, batch)
        except Exception as error:
            # Same contract as the sync's flush: the batch keeps its old
            # feature_version and is simply picked up again next time.
            logger.warning(
                "could not write %d rebuilt rows for athlete %s: %s",
                len(batch), athlete_id, error,
            )
        batch.clear()


def _summary_stream(activity: Dict[str, Any]) -> ActivityStream:
    """A streamless activity as an :class:`ActivityStream`, from its stored row.

    The columns being read back are the ones the summary wrote in the first
    place, so this round-trips: distance, elevation and time came off Strava's
    activity summary and Relative Effort off the listing.
    """
    empty = np.full(0, np.nan)
    return ActivityStream(
        activity_id=activity["activity_id"],
        sport_type=activity["sport_type"],
        time=empty,
        distance=empty,
        altitude=empty,
        heartrate=empty,
        start_date=activity["start_date"],
        has_streams=False,
        summary_distance_m=activity["distance_m"],
        summary_moving_time_s=activity["moving_s"],
        summary_elevation_gain_m=activity["elevation_gain_m"],
        summary_relative_effort=activity["relative_effort"],
    )
