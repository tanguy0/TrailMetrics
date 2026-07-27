"""Turn a panel's data source into resolved, grouped data.

The single place that answers "which activities is this panel about, and how do
they group" — mode, filters, window assignment. Every plot receives the result, so
none of them re-implements selection logic.

Selection runs entirely on activity *summaries*, never on streams or features:
server-side that is one query, and a panel of purely activity-level plots never
touches per-second data at all.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.domain.dataset.binning import to_date
from src.domain.dataset.resolved import ResolvedGroup, ResolvedPanelData
from src.domain.ports.activity_data import ActivityDataSource, ActivitySummary
from src.domain.spec.datasource import (
    DEFAULT_SELECTION_LABEL,
    ActivityFilter,
    DataSourceSpec,
    SourceMode,
)
from src.usecases.base import UseCase


@dataclass
class ResolvePanelDataInput:
    source: DataSourceSpec
    data: ActivityDataSource
    lang: str = "en"
    mass_kg: Optional[float] = None
    # Memo shared across the panels of one render.
    memo: Dict[Any, Any] = field(default_factory=dict)
    # Set when at least one plot in the panel needs per-second data, so
    # summary-only activities are excluded rather than drawn as flat lines.
    require_streams: bool = False
    # Label for the single group of a hand-picked selection.
    selection_fallback_label: str = DEFAULT_SELECTION_LABEL


class ResolvePanelData(UseCase):
    """Resolve a :class:`DataSourceSpec` against an athlete's activities."""

    def execute(self, params: ResolvePanelDataInput) -> ResolvedPanelData:
        summaries = params.data.summaries()
        candidates, dropped = self._filter(summaries, params)
        groups = self._groups(params.source, candidates, params)

        return ResolvedPanelData(
            groups=groups,
            data=params.data,
            lang=params.lang,
            mass_kg=params.mass_kg,
            memo=params.memo,
            dropped_streamless=dropped,
            summaries=summaries,
        )

    # --- Filtering ---------------------------------------------------------

    def _filter(
        self, summaries: List[ActivitySummary], params: ResolvePanelDataInput
    ) -> Tuple[List[ActivitySummary], int]:
        """Apply the source's filters.

        Returns the survivors plus how many summary-only activities were dropped
        because the panel needs per-second data — reported to the user rather than
        silently swallowed.
        """
        filters = params.source.filters
        wanted_sports = set(filters.sport_types or [])
        dropped_streamless = 0
        out: List[ActivitySummary] = []

        for summary in summaries:
            if params.require_streams and not summary.has_streams:
                dropped_streamless += 1
                continue
            if wanted_sports and summary.sport_type not in wanted_sports:
                continue
            if not self._passes_distance(summary, filters):
                continue
            out.append(summary)
        return out, dropped_streamless

    @staticmethod
    def _passes_distance(summary: ActivitySummary, filters: ActivityFilter) -> bool:
        kilometres = (summary.distance_m or 0.0) / 1000.0
        if filters.min_distance_km is not None and kilometres < filters.min_distance_km:
            return False
        if filters.max_distance_km is not None and kilometres > filters.max_distance_km:
            return False
        return True

    # --- Grouping ----------------------------------------------------------

    def _groups(
        self,
        source: DataSourceSpec,
        candidates: List[ActivitySummary],
        params: ResolvePanelDataInput,
    ) -> List[ResolvedGroup]:
        if source.mode is SourceMode.ACTIVITIES:
            available = {s.activity_id for s in candidates}
            # Preserve the order the user picked them in.
            ids = [int(i) for i in source.activity_ids if int(i) in available]
            label = source.selection_label.strip() or params.selection_fallback_label
            return [ResolvedGroup(label=label, index=0, activity_ids=ids)]

        groups: List[ResolvedGroup] = []
        for index, window in enumerate(source.active_windows):
            ids = [
                s.activity_id
                for s in candidates
                if window.start <= to_date(s.start_date) <= window.end
            ]
            groups.append(ResolvedGroup(
                label=window.name, index=index, activity_ids=ids, window=window,
            ))
        return groups
