"""Port for where a render gets its activity data.

The three data levels have very different access costs, and this port makes that
explicit so each backend can be efficient in its own way:

* :meth:`ActivityDataSource.summaries` — cheap, always available. Enough to resolve
  a data source (date, sport, distance, whether streams exist). One SQL query
  server-side; no per-second data touched.
* :meth:`ActivityDataSource.features` — the tidy activity table. Read from a
  database, or computed on demand from streams.
* :meth:`ActivityDataSource.stream` — one activity's per-second arrays. The
  expensive one, fetched only by stream-level plots and model fits.

Two implementations exist: one over an in-memory list of streams (notebooks and
tests) and one over Postgres + object storage. Neither the resolver nor any plot
knows which it is talking to.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

import pandas as pd

from src.domain.models.activity import ActivityStream


@dataclass(frozen=True)
class ActivitySummary:
    """The little we need to know about an activity to *select* it."""

    activity_id: int
    start_date: datetime
    sport_type: str
    has_streams: bool
    distance_m: float = 0.0
    moving_s: float = 0.0
    # Strava's Relative Effort — carried on the cheap summary (not just the full
    # feature row) so a cross-sport, whole-history read (the Fitness & Fatigue
    # plot) never has to pull every activity's feature row just for this one
    # number. ``None`` when the activity has no heart rate.
    relative_effort: Optional[float] = None
    # The athlete's own two ratings — effort (1-10) and how it felt
    # ("faible"/"ok"/"fort"). Carried here for the same reason as
    # ``relative_effort``: the weekly RPE/feel plot needs them across the whole
    # cross-sport history (``ResolvedPanelData.all_summaries``), and pulling
    # every feature row just for two small columns would cost the read it saves.
    # ``None`` when the athlete has not rated that activity — which is the normal
    # case, so every reader has to handle it.
    rpe: Optional[int] = None
    feeling: Optional[str] = None

    @property
    def label(self) -> str:
        """``2025-10-12 · TrailRun · 21.10 km · 1:45:03`` — pickers and legends."""
        total = int(round(self.moving_s or 0))
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        duration = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
        return (
            f"{self.start_date:%Y-%m-%d} · {self.sport_type} · "
            f"{(self.distance_m or 0) / 1000:.2f} km · {duration}"
        )


class ActivityDataSource(ABC):
    """Read access to one athlete's activities, at each of the three levels."""

    @abstractmethod
    def summaries(self) -> List[ActivitySummary]:
        """Every known activity, oldest first. Cheap enough to call per panel."""

    @abstractmethod
    def features(self, activity_ids: Sequence[int]) -> pd.DataFrame:
        """Feature rows for ``activity_ids``, one row each, in ``FEATURE_COLUMNS``."""

    @abstractmethod
    def stream(self, activity_id: int) -> Optional[ActivityStream]:
        """Per-second arrays for one activity, or ``None`` when unavailable."""

    # --- Derived helpers, the same for every backend -----------------------

    def summary(self, activity_id: int) -> Optional[ActivitySummary]:
        for item in self.summaries():
            if item.activity_id == activity_id:
                return item
        return None

    def date_range(self) -> Optional[tuple]:
        """``(oldest, newest)`` start dates, or ``None`` with no activities."""
        dates = [s.start_date for s in self.summaries()]
        return (min(dates), max(dates)) if dates else None

    def sport_types(self) -> List[str]:
        return sorted({s.sport_type for s in self.summaries()})
