"""Plot registry: what a plot type *is*, and the helpers they all share.

A plot type is a :class:`PlotDefinition` — a key, a declarative parameter schema,
the data level it consumes, and a pure ``compute`` that turns resolved panel data
into chart IR. Nothing else. It never renders, never reads request state, and never
imports a UI framework.

That contract is what the whole app is built on: the panel editor generates a
plot's form from ``params``, the resolver builds only the ``level`` it needs, the
renderer draws whatever IR comes back, and adding a plot type is one module plus
one :func:`register` call — no page, no wiring.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.domain.charts.ir import Axis, AxisKind, CellFormat, PlotOutput
from src.domain.dataset.metrics import COUNT, DURATION, PACE, ActivityMetric
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.gap import theme
from src.domain.plotting_common import CURVE_PALETTE, fmt_hms, fmt_pace
from src.domain.spec.params import ParamSpec
from src.translations import translate

# How a plot's series relate to the panel's data source.
SERIES_BY_GROUP = "group"        # one series per window / per selection
SERIES_BY_ACTIVITY = "activity"  # one series per activity

# Cost hints. "expensive" plots default to manual refresh in the editor, so
# tweaking a neighbouring widget never silently refits a model.
CHEAP = "cheap"
EXPENSIVE = "expensive"

ComputeFn = Callable[[ResolvedPanelData, dict], PlotOutput]


@dataclass
class PlotDefinition:
    """One plot type in the registry."""

    key: str
    label_key: str
    description_key: str
    level: DataLevel
    compute: ComputeFn
    params: List[ParamSpec] = field(default_factory=list)
    series_level: str = SERIES_BY_GROUP
    requires_streams: bool = False
    requires_weight: bool = False
    # False for plot types whose output comes from their parameters alone (prose, an
    # image). Without this a text block in a panel that selects nothing would render
    # as "no activities in this selection", which is true and useless.
    requires_data: bool = True
    # True for a plot whose output depends on what the athlete *typed* (RPE,
    # feeling) rather than on what was imported. Those values change under
    # activity ids that stay the same, which the render signature cannot see on
    # its own — so it folds in a digest of them for these plots, and only for
    # these: rating one run must not invalidate a fitted GAP curve.
    reads_ratings: bool = False
    cost: str = CHEAP
    # Shown in the "add plot" picker to group related types.
    category_key: str = "plotcat.general"

    def label(self, lang: str) -> str:
        return translate(self.label_key, lang)

    def description(self, lang: str) -> str:
        return translate(self.description_key, lang)


_REGISTRY: Dict[str, PlotDefinition] = {}


def register(definition: PlotDefinition) -> PlotDefinition:
    """Add a plot type to the registry (raises on a duplicate key)."""
    if definition.key in _REGISTRY:
        raise ValueError(f"plot type already registered: {definition.key!r}")
    _REGISTRY[definition.key] = definition
    return definition


def get(key: str) -> Optional[PlotDefinition]:
    return _REGISTRY.get(key)


def all_plots() -> List[PlotDefinition]:
    """Every registered plot type, in registration order."""
    return list(_REGISTRY.values())


def by_level(level: DataLevel) -> List[PlotDefinition]:
    return [d for d in _REGISTRY.values() if d.level is level]


# --- Colors ----------------------------------------------------------------

def group_color(index: int) -> str:
    """Stable colour for a data-source group.

    Groups are the top-level thing a reader compares (this window vs that one), so
    they get the widely-spaced hues; activities within a group use the general
    palette.
    """
    cycle = theme.TIME_SCALE_CYCLE
    return cycle[index % len(cycle)]


def series_color(index: int) -> str:
    return CURVE_PALETTE[index % len(CURVE_PALETTE)]


# --- Display window --------------------------------------------------------

def display_window(
    resolved: ResolvedPanelData, *, fallback_end: date, fallback_days: int,
) -> Tuple[date, date]:
    """The panel's own selected date range, or a trailing fallback without one.

    For the plots that draw a **continuous daily or weekly timeline** rather than
    one point per activity (training load, and the weekly RPE/feel review). A time
    window is the only data-source mode that defines such a range; a hand-picked
    activity list does not, hence the fallback.

    These plots borrow the window's ``[start, end]`` purely to decide what to
    *show* — what they read is ``all_summaries()``, the whole cross-sport history,
    so the window's own activity match is irrelevant to them.
    """
    starts = [g.window.start for g in resolved.groups if g.window is not None]
    ends = [g.window.end for g in resolved.groups if g.window is not None]
    if starts and ends:
        return min(starts), max(ends)
    return fallback_end - timedelta(days=fallback_days), fallback_end


# --- Metric-driven axis and value formatting -------------------------------

def metric_axis(metric: ActivityMetric, lang: str, *, title: Optional[str] = None) -> Axis:
    """The y-axis a metric should be drawn on, including its quirks.

    Paces get a reversed duration axis so a faster pace sits *higher* — the only
    orientation that reads as "improving" — and durations tick as clock times
    rather than raw seconds.
    """
    label = title if title is not None else metric_label(metric, lang)
    if metric.value_kind == PACE:
        return Axis(title=label, kind=AxisKind.DURATION, reversed=True,
                    tick_format="%M:%S")
    if metric.value_kind == DURATION:
        return Axis(title=label, kind=AxisKind.DURATION, tick_format="%H:%M:%S")
    suffix = f" {metric.unit}" if metric.unit else None
    return Axis(title=label, kind=AxisKind.LINEAR,
                tick_format=f",.{metric.decimals}f", suffix=suffix)


def metric_label(metric: ActivityMetric, lang: str) -> str:
    label = translate(metric.label_key, lang)
    if metric.unit and metric.value_kind not in (DURATION, PACE):
        return f"{label} ({metric.unit})"
    return label


def format_metric_value(metric: ActivityMetric, value: Optional[float]) -> str:
    """Human string for one metric value — used in hovers and table cells."""
    if value is None or not np.isfinite(value):
        return "—"
    if metric.value_kind == PACE:
        return fmt_pace(value)
    if metric.value_kind == DURATION:
        return fmt_hms(value)
    if metric.value_kind == COUNT:
        return f"{int(round(value))}"
    text = f"{value:,.{metric.decimals}f}"
    return f"{text} {metric.unit}".strip()


def metric_hover_texts(metric: ActivityMetric, values) -> List[str]:
    return [format_metric_value(metric, v) for v in values]


def metric_cell_format(metric: ActivityMetric) -> CellFormat:
    """Table-cell formatting mirroring :func:`format_metric_value`."""
    if metric.value_kind == PACE:
        return CellFormat(kind="pace")
    if metric.value_kind == DURATION:
        return CellFormat(kind="duration")
    if metric.value_kind == COUNT:
        return CellFormat(kind="integer")
    return CellFormat(kind="number", decimals=metric.decimals, suffix=metric.unit)


def hover_template(x_part: str) -> str:
    """Hover showing a pre-formatted value (``customdata``) under an x read-out."""
    return f"{x_part}<br>%{{customdata}}<extra>%{{fullData.name}}</extra>"


# --- Shared note helpers ---------------------------------------------------

def note(key: str, lang: str, **fmt) -> str:
    text = translate(key, lang)
    return text.format(**fmt) if fmt else text


def weight_note(lang: str) -> str:
    return translate("races.weight_needed", lang)


def streamless_note(lang: str, count: int) -> str:
    return translate("panel.dropped_streamless", lang).format(count=count)
