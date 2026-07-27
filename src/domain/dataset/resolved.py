"""What a panel's data source resolves to — the input every plot receives.

A data source becomes an ordered list of named :class:`ResolvedGroup`\\ s (one per
time window, or one for a hand-picked selection). On top of that sit the three
**data levels** a plot can ask for:

===========  ==========================================  ==============================
level        shape                                       used by
===========  ==========================================  ==============================
``ACTIVITY`` tidy frame, one row per (group, activity)    trends, records, tables, scatter
``STREAM``   per-second series for one activity           within-activity evolution
``SPLIT``    samples pooled across a group's activities   GAP model fitting
===========  ==========================================  ==============================

Every level is **lazy and memoized**: a panel with three activity-level plots
builds the feature table once, and a panel with none never touches a stream. The
memo dict is injected by the caller, so it survives across the panels of a page
and across requests — which is what keeps an interactive builder responsive over a
multi-year history.

Where the data physically comes from is behind
:class:`~src.domain.ports.activity_data.ActivityDataSource`, so the same resolved
object works over in-memory streams or over a database.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_data import ActivityDataSource, ActivitySummary
from src.domain.spec.datasource import TimeWindow

# Columns the resolver adds to the feature table.
GROUP_COLUMN = "group"
GROUP_INDEX_COLUMN = "group_index"


class DataLevel(str, Enum):
    ACTIVITY = "activity"
    STREAM = "stream"
    SPLIT = "split"


@dataclass
class ResolvedGroup:
    """One named series' worth of activities.

    ``window`` is kept when the group came from a time window: the elapsed-months
    axis needs the window's start to align groups to a common 0.
    """

    label: str
    index: int
    activity_ids: List[int] = field(default_factory=list)
    window: Optional[TimeWindow] = None

    @property
    def size(self) -> int:
        return len(self.activity_ids)


class ResolvedPanelData:
    """A panel's resolved data source, with each level computed on demand."""

    def __init__(
        self,
        groups: Sequence[ResolvedGroup],
        data: ActivityDataSource,
        *,
        lang: str,
        mass_kg: Optional[float] = None,
        memo: Optional[Dict[Any, Any]] = None,
        dropped_streamless: int = 0,
        summaries: Optional[Sequence[ActivitySummary]] = None,
    ):
        self.groups = list(groups)
        self.data = data
        self.lang = lang
        self.mass_kg = mass_kg
        self._memo: Dict[Any, Any] = memo if memo is not None else {}
        # How many activities were skipped for lacking per-second data.
        self.dropped_streamless = dropped_streamless
        self._features: Optional[pd.DataFrame] = None
        self._summaries: Optional[Dict[int, ActivitySummary]] = (
            {s.activity_id: s for s in summaries} if summaries is not None else None
        )

    # --- Shape ------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not any(g.activity_ids for g in self.groups)

    @property
    def has_multiple_groups(self) -> bool:
        return len([g for g in self.groups if g.activity_ids]) > 1

    @property
    def activity_ids(self) -> List[int]:
        """Every selected activity, in group order, without duplicates."""
        seen, out = set(), []
        for group in self.groups:
            for activity_id in group.activity_ids:
                if activity_id not in seen:
                    seen.add(activity_id)
                    out.append(activity_id)
        return out

    # --- ACTIVITY level ----------------------------------------------------

    @property
    def features(self) -> pd.DataFrame:
        """One row per (group, activity), tagged with the group it belongs to.

        Fetched once for every selected activity, then sliced per group — so
        overlapping windows cost one read, not two. Overlap legitimately produces
        two rows for one activity: it really is in both groups.
        """
        if self._features is None:
            base = self.data.features(self.activity_ids)
            frames = []
            for group in self.groups:
                if not group.activity_ids:
                    continue
                slice_ = base[base["activity_id"].isin(group.activity_ids)]
                if slice_.empty:
                    continue
                frames.append(slice_.assign(**{
                    GROUP_COLUMN: group.label,
                    GROUP_INDEX_COLUMN: group.index,
                }))
            self._features = (
                pd.concat(frames, ignore_index=True) if frames else base
            )
        return self._features

    def group_features(self, group: ResolvedGroup) -> pd.DataFrame:
        frame = self.features
        if frame.empty or GROUP_INDEX_COLUMN not in frame.columns:
            return frame
        return frame[frame[GROUP_INDEX_COLUMN] == group.index]

    # --- STREAM level -----------------------------------------------------

    def stream(self, activity_id: int) -> Optional[ActivityStream]:
        """One activity's per-second arrays, fetched at most once per render."""
        return self.memo(
            ("stream", int(activity_id)),
            lambda: self.data.stream(int(activity_id)),
        )

    def group_streams(self, group: ResolvedGroup) -> List[ActivityStream]:
        streams = (self.stream(activity_id) for activity_id in group.activity_ids)
        return [s for s in streams if s is not None]

    # --- Labels -----------------------------------------------------------

    def activity_label(self, activity_id: int) -> str:
        summary = self._summary(int(activity_id))
        return summary.label if summary else f"Activity {activity_id}"

    def _summary(self, activity_id: int) -> Optional[ActivitySummary]:
        if self._summaries is None:
            self._summaries = {s.activity_id: s for s in self.data.summaries()}
        return self._summaries.get(activity_id)

    # --- Memoization for the expensive levels ------------------------------

    def memo(self, key: Any, factory: Callable[[], Any]) -> Any:
        """Compute ``factory()`` once per ``key`` for the life of the memo dict.

        Used for streams, per-second series and fitted models, so re-rendering a
        page after one widget changed never refits or re-smooths anything whose
        inputs are unchanged.
        """
        if key not in self._memo:
            self._memo[key] = factory()
        return self._memo[key]
