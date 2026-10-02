"""Which family a figure belongs to, and what that means for each of its traces.

design/tagg/charts.md § v1.2: the gradient area is a *sign* — "a quantity you
follow on its own, over time" — so it belongs to one family only. Every figure
is classed once, here, and both renderers apply the resulting :class:`Plan`
(``src/domain/charts/plotly.py`` and ``web/components/ChartView.tsx``, whose
twin of this module is ``web/lib/chartFamily.ts``; a parity test keeps the two
deciding the same thing).

A plot that knows its figure declares ``ChartData.family`` and the per-trace
options (``Trace.area``, ``end_label``, ``background``); classification is the
fallback for whatever does not — in practice a generic ``metric_trend`` built
in a cached output from before v1.2.

Pure: no Plotly, no theme lookups beyond the two colours that say what a trace
*is* (the reference grey, series 1).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.domain.charts.ir import AxisKind, ChartData, Trace, TraceKind
from src.domain.gap import theme

FAMILIES = ("tracking", "comparison", "function", "oscillation", "composition", "scatter")

# Lines start at zero only when zero is already near the data: within this share
# of the data's span below its minimum. Only a fallback now — the aggregation a
# plot declares decides first (charts.md § v1.2 — l'agrégation décide).
ZERO_REACH = 0.5

# Stroke weights of charts.md § v1.2, by family.
TRACKING_WIDTH = 2.2
FUNCTION_WIDTH = 2.2
OSCILLATION_WIDTH = 2.0
CURRENT_WIDTH = 2.4
OTHER_WIDTH = 1.5
OTHER_OPACITY = 0.7
# Past this many other series, they fade further.
CROWD = 3
CROWD_OPACITY = 0.55
SCATTER_OPACITY = 0.6
SCATTER_MARKER_SIZE = 8  # r 4
SCATTER_TREND_WIDTH = 2.0


@dataclass
class Plan:
    """What a renderer does with each trace, by index into ``chart.traces``."""

    family: str
    area: Optional[int] = None
    end_labels: List[int] = field(default_factory=list)
    hidden_legend: List[int] = field(default_factory=list)
    widths: Dict[int, float] = field(default_factory=dict)
    opacities: Dict[int, float] = field(default_factory=dict)
    marker_sizes: Dict[int, float] = field(default_factory=dict)
    # The series whose value leads the unified hover.
    main: Optional[int] = None
    baseline: Optional[float] = None


def is_reference(color: Optional[str]) -> bool:
    return (color or "").upper() == theme.CHART_REF.upper()


def has_points(trace: Trace) -> bool:
    return any(v is not None for v in trace.y)


def finite(values) -> List[float]:
    out = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _color(chart: ChartData, index: int) -> str:
    trace = chart.traces[index]
    return trace.color or theme.CURVE_CYCLE[index % len(theme.CURVE_CYCLE)]


def _athlete(chart: ChartData) -> List[int]:
    """The athlete's own series: not a reference, not a backdrop, with data."""
    return [
        i for i, t in enumerate(chart.traces)
        if has_points(t) and not t.background and not is_reference(_color(chart, i))
    ]


def _is_line(trace: Trace) -> bool:
    return trace.kind in (TraceKind.LINE, TraceKind.STEP)


def _on_left(chart: ChartData, trace: Trace) -> bool:
    return trace.axis != "y2" or chart.y2_axis is None


def classify(chart: ChartData) -> str:
    """The figure's family: declared, else read off its shape."""
    if chart.family in FAMILIES:
        return chart.family
    athlete = [chart.traces[i] for i in _athlete(chart)]
    if any(t.stack_group or t.kind in (TraceKind.AREA, TraceKind.BAR) for t in athlete):
        return "composition"
    if any(t.kind is TraceKind.SCATTER for t in athlete):
        return "scatter"
    if chart.x_axis.kind is not AxisKind.DATE:
        return "function"
    left = [t for t in athlete if _is_line(t) and _on_left(chart, t)]
    if len(left) >= 2:
        return "comparison"
    if not left:
        return "function"
    values = finite(left[0].y)
    lo, hi = min(values), max(values)
    span = hi - lo or abs(hi) or 1.0
    if chart.y_axis.kind is not AxisKind.LINEAR or lo < 0 < hi or lo > ZERO_REACH * span:
        return "oscillation"
    return "tracking"


def _last_x(trace: Trace):
    for x, y in zip(reversed(trace.x), reversed(trace.y)):
        if y is not None:
            return str(x)
    return ""


def _current(chart: ChartData, lines: List[int]) -> Optional[int]:
    """A comparison's current series: declared, else the latest period, else series 1."""
    for i in lines:
        if chart.traces[i].end_label is True:
            return i
    if chart.x_axis.kind is AxisKind.DATE and lines:
        # ISO strings order chronologically; the series reaching furthest is current.
        return max(lines, key=lambda i: (_last_x(chart.traces[i]), -i))
    for i in lines:
        if _color(chart, i).upper() == theme.CHART_YOU_1.upper():
            return i
    return lines[0] if lines else None


def _can_fill(chart: ChartData, trace: Trace) -> bool:
    """Whether a gradient area may sit under this line at all."""
    return (
        chart.y2_axis is None
        and chart.y_axis.kind is AxisKind.LINEAR
        and not chart.y_axis.reversed
        and _is_line(trace)
        and trace.band_upper is None
        and not trace.stack_group
    )


def plan(chart: ChartData) -> Plan:
    family = classify(chart)
    declared = chart.family in FAMILIES
    result = Plan(family=family)
    athlete = _athlete(chart)
    lines = [i for i in athlete if _is_line(chart.traces[i]) and _on_left(chart, chart.traces[i])]
    dual = chart.y2_axis is not None

    labelled: List[int] = []
    if family == "tracking":
        candidates = [i for i in lines if chart.traces[i].area is not False]
        if candidates and _can_fill(chart, chart.traces[candidates[0]]):
            result.area = candidates[0]
        labelled = lines
        if not declared:
            result.widths.update({i: TRACKING_WIDTH for i in lines})
    elif family == "comparison":
        current = _current(chart, lines)
        others = [i for i in lines if i != current]
        if current is not None:
            result.widths[current] = CURRENT_WIDTH
            labelled = [current]
        fade = CROWD_OPACITY if len(others) > CROWD else OTHER_OPACITY
        for i in others:
            result.widths[i] = OTHER_WIDTH
            result.opacities[i] = fade
        result.main = current
    elif family == "function":
        if not declared:
            result.widths.update({i: FUNCTION_WIDTH for i in lines})
    elif family == "oscillation":
        labelled = lines
        if not declared:
            result.widths.update({i: OSCILLATION_WIDTH for i in lines})
        result.baseline = _baseline(chart, lines)
    elif family == "scatter":
        for i in athlete:
            if chart.traces[i].kind is TraceKind.SCATTER:
                result.opacities[i] = SCATTER_OPACITY
                result.marker_sizes[i] = SCATTER_MARKER_SIZE
            elif _is_line(chart.traces[i]):
                result.widths[i] = SCATTER_TREND_WIDTH

    # A declared area (the fatigue) outranks the family's own pick.
    for i in lines:
        if chart.traces[i].area is True and _can_fill(chart, chart.traces[i]):
            result.area = i
            break

    # Per-trace end_label wins over the family, except that a second axis never
    # carries labels; on a comparison, `end_label=True` already named the current.
    if family != "comparison":
        labelled = [i for i in labelled if chart.traces[i].end_label is not False]
        labelled += [i for i in lines if chart.traces[i].end_label is True and i not in labelled]
    result.end_labels = [] if dual else sorted(labelled)

    # One athlete line is named well enough by its end label; with several the
    # legend says which is which.
    if family in ("tracking", "oscillation") and len(lines) == 1 and result.end_labels:
        result.hidden_legend = list(result.end_labels)

    if result.main is None:
        result.main = result.area if result.area is not None else (lines[0] if lines else None)
    return result


def _baseline(chart: ChartData, lines: List[int]) -> Optional[float]:
    if chart.baseline is not None:
        return chart.baseline
    values = [v for i in lines for v in finite(chart.traces[i].y)]
    # A pace axis takes its mean too; a date or category axis has no level.
    if not values or chart.y_axis.kind not in (AxisKind.LINEAR, AxisKind.DURATION):
        return None
    if min(values) < 0 < max(values):
        return 0.0
    return float(np.mean(values))
