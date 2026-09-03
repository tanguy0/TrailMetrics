""":class:`ActivityDataSource` backed by Postgres rows + object storage.

The server-side counterpart of
:class:`~src.domain.dataset.in_memory.InMemoryActivityData`. Plots cannot tell them
apart, which is the point: the same page renders identically from a notebook's
in-memory streams or from a database.

Each level is read at the cost it deserves — summaries and features are SQL,
streams are blobs fetched one at a time and only when something asks.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.domain.dataset.features import apply_mass, frame_from_rows
from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_data import ActivityDataSource, ActivitySummary
from src.domain.ports.storage import ActivityRepository, StreamStore
from src.infrastructure.storage.codec import decode_stream


class StoredActivityData(ActivityDataSource):
    def __init__(
        self,
        athlete_id: int,
        activities: ActivityRepository,
        streams: StreamStore,
        *,
        mass_kg: Optional[float] = None,
    ):
        self.athlete_id = athlete_id
        self.activities = activities
        self.streams = streams
        self.mass_kg = mass_kg
        self._summaries: Optional[List[ActivitySummary]] = None
        self._features: Dict[int, dict] = {}
        self._stream_cache: Dict[int, Optional[ActivityStream]] = {}

    def summaries(self) -> List[ActivitySummary]:
        if self._summaries is None:
            self._summaries = [
                ActivitySummary(
                    activity_id=int(row["activity_id"]),
                    # Windows and binning are naive; normalize once, here.
                    start_date=_naive(row["start_date"]),
                    sport_type=row["sport_type"],
                    has_streams=bool(row["has_streams"]),
                    distance_m=float(row["distance_m"] or 0.0),
                    moving_s=float(row["moving_s"] or 0.0),
                    relative_effort=(
                        float(row["relative_effort"])
                        if row.get("relative_effort") is not None else None
                    ),
                    rpe=int(row["rpe"]) if row.get("rpe") is not None else None,
                    feeling=row.get("feeling") or None,
                )
                for row in self.activities.summaries(self.athlete_id)
            ]
        return self._summaries

    def features(self, activity_ids: Sequence[int]) -> pd.DataFrame:
        wanted = [int(i) for i in activity_ids]
        missing = [i for i in wanted if i not in self._features]
        if missing:
            for row in self.activities.rows(self.athlete_id, missing):
                self._features[int(row["activity_id"])] = row
        rows = [self._features[i] for i in wanted if i in self._features]
        return apply_mass(frame_from_rows(rows), self.mass_kg)

    def stream(self, activity_id: int) -> Optional[ActivityStream]:
        key = int(activity_id)
        if key in self._stream_cache:
            return self._stream_cache[key]
        path = self.activities.stream_object(self.athlete_id, key)
        stream = None
        if path:
            payload = self.streams.get(path)
            if payload is not None:
                stream = decode_stream(payload)
        self._stream_cache[key] = stream
        return stream


def _naive(value):
    return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value
