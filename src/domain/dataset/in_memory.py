"""In-memory activity data: a list of streams, features computed on demand.

The backend for notebooks, tests and any "load it all, then analyse" flow. It
computes feature rows lazily through a :class:`FeatureStore`, so a panel that only
needs summaries never pays for the per-second pass.
"""

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.domain.dataset.features import FeatureStore, apply_mass, sport_name
from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_data import ActivityDataSource, ActivitySummary


class InMemoryActivityData(ActivityDataSource):
    """:class:`ActivityDataSource` over a list of already-fetched streams."""

    def __init__(
        self,
        streams: Sequence[ActivityStream],
        *,
        mass_kg: Optional[float] = None,
        feature_cache: Optional[Dict] = None,
    ):
        self._streams: Dict[int, ActivityStream] = {}
        for stream in streams:
            if isinstance(stream.start_date, datetime):
                self._streams[int(stream.activity_id)] = stream
        self.mass_kg = mass_kg
        self.store = FeatureStore(cache=feature_cache)
        self._summaries: Optional[List[ActivitySummary]] = None

    def summaries(self) -> List[ActivitySummary]:
        if self._summaries is None:
            self._summaries = sorted(
                (self._summarize(s) for s in self._streams.values()),
                key=lambda s: s.start_date,
            )
        return self._summaries

    @staticmethod
    def _summarize(stream: ActivityStream) -> ActivitySummary:
        """Totals straight off the raw arrays — no smoothing, no feature pass.

        Deliberately cheap: this only has to be good enough to filter and label,
        and the precise figures come from the feature table.
        """
        distance = 0.0
        moving = 0.0
        if stream.distance is not None and len(stream.distance):
            distance = float(stream.distance[-1] - stream.distance[0])
        elif stream.summary_distance_m is not None:
            distance = float(stream.summary_distance_m)
        if stream.time is not None and len(stream.time):
            moving = float(stream.time[-1] - stream.time[0])
        elif stream.summary_moving_time_s is not None:
            moving = float(stream.summary_moving_time_s)
        return ActivitySummary(
            activity_id=int(stream.activity_id),
            start_date=stream.start_date,
            sport_type=sport_name(stream.sport_type),
            has_streams=bool(getattr(stream, "has_streams", True)),
            distance_m=distance,
            moving_s=moving,
        )

    def features(self, activity_ids: Sequence[int]) -> pd.DataFrame:
        wanted = [self._streams[int(i)] for i in activity_ids if int(i) in self._streams]
        return apply_mass(self.store.table(wanted), self.mass_kg)

    def stream(self, activity_id: int) -> Optional[ActivityStream]:
        return self._streams.get(int(activity_id))
