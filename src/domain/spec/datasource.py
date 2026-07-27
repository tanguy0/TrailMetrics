"""A panel's data source: *which activities* the panel is about.

One panel has exactly one data source, and it can be defined three ways:

* :attr:`SourceMode.ACTIVITIES` — a hand-picked set of activities;
* :attr:`SourceMode.WINDOW` — one time window, every activity inside it;
* :attr:`SourceMode.WINDOWS` — several named time windows, compared side by side.

A source always resolves to an ordered list of **named groups** (one per window,
or one for a hand-picked set). Groups are what plots colour and label by, which
is why one abstraction covers the "time scales" of the GAP simulator and the
"seasons" of the long-term page — those were the same feature written twice.

Note what is deliberately *not* here: whether a plot draws one series per group
or one per activity. That is a property of the plot, not the source, so a single
panel can hold a pooled model fit and per-activity traces over the same data.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

# Label used for the single group of a hand-picked selection when the user gave
# the panel no better name.
DEFAULT_SELECTION_LABEL = "Selection"


class SourceMode(str, Enum):
    ACTIVITIES = "activities"
    WINDOW = "window"
    WINDOWS = "windows"


@dataclass
class TimeWindow:
    """A named, inclusive ``[start, end]`` period. One window → one group."""

    name: str
    start: date
    end: date

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "start": self.start.isoformat(),
                "end": self.end.isoformat()}

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "TimeWindow":
        return TimeWindow(
            name=str(raw.get("name") or ""),
            start=date.fromisoformat(raw["start"]),
            end=date.fromisoformat(raw["end"]),
        )

    @property
    def is_valid(self) -> bool:
        return bool(self.name.strip()) and self.start <= self.end

    @property
    def length_days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class ActivityFilter:
    """Filters applied *within* whatever the mode selected.

    Empty ``sport_types`` means every sport. ``require_streams`` is set by the
    resolver, not the user: plots that need per-second data ask for it so
    summary-only activities (manual entries) are dropped rather than plotted as
    flat lines.
    """

    sport_types: List[str] = field(default_factory=list)
    min_distance_km: Optional[float] = None
    max_distance_km: Optional[float] = None
    require_streams: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport_types": list(self.sport_types),
            "min_distance_km": self.min_distance_km,
            "max_distance_km": self.max_distance_km,
        }

    @staticmethod
    def from_dict(raw: Optional[Dict[str, Any]]) -> "ActivityFilter":
        raw = raw or {}
        return ActivityFilter(
            sport_types=list(raw.get("sport_types") or []),
            min_distance_km=raw.get("min_distance_km"),
            max_distance_km=raw.get("max_distance_km"),
        )


@dataclass
class DataSourceSpec:
    """Which activities a panel works on, and how they group."""

    mode: SourceMode = SourceMode.WINDOW
    # ACTIVITIES mode.
    activity_ids: List[int] = field(default_factory=list)
    selection_label: str = ""
    # WINDOW / WINDOWS mode. WINDOW uses the first entry.
    windows: List[TimeWindow] = field(default_factory=list)
    filters: ActivityFilter = field(default_factory=ActivityFilter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "activity_ids": list(self.activity_ids),
            "selection_label": self.selection_label,
            "windows": [w.to_dict() for w in self.windows],
            "filters": self.filters.to_dict(),
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "DataSourceSpec":
        return DataSourceSpec(
            mode=SourceMode(raw.get("mode") or SourceMode.WINDOW.value),
            activity_ids=[int(i) for i in (raw.get("activity_ids") or [])],
            selection_label=str(raw.get("selection_label") or ""),
            windows=[TimeWindow.from_dict(w) for w in (raw.get("windows") or [])],
            filters=ActivityFilter.from_dict(raw.get("filters")),
        )

    @property
    def active_windows(self) -> List[TimeWindow]:
        """The windows this mode actually uses, invalid ones dropped."""
        if self.mode is SourceMode.WINDOW:
            return [w for w in self.windows[:1] if w.is_valid]
        if self.mode is SourceMode.WINDOWS:
            return [w for w in self.windows if w.is_valid]
        return []

    def describe(self) -> str:
        """Short, language-neutral summary for the panel header."""
        if self.mode is SourceMode.ACTIVITIES:
            return f"{len(self.activity_ids)} activities"
        windows = self.active_windows
        if not windows:
            return "—"
        if len(windows) == 1:
            w = windows[0]
            return f"{w.start.isoformat()} → {w.end.isoformat()}"
        return " · ".join(w.name for w in windows)
