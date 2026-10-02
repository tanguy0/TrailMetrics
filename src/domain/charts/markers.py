"""Today and the athlete's races, pinned on every date-axis chart that covers them.

charts.md § v1.1 — repères: "today" is a dotted sun line as soon as a chart's
window includes the date; a race (a goal on the training calendar) is a terra
dot on the x-axis, named, on every time chart that spans its day.

Applied when a result is *served*, not when it is computed: plot outputs are
cached, and a goal added on the calendar tomorrow must show on a curve fitted
today without invalidating that cache. Plots never place these themselves — the
renderer draws them (see :class:`src.domain.charts.ir.Marker`).
"""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from src.domain.charts.ir import AxisKind, ChartData, Marker, PlotOutput

# Bins at least this far apart (weekly, monthly) cover the days up to the next
# one, so a weekly chart whose last bar is Monday still "includes" Thursday.
_BINNED_STEP = timedelta(days=7)


@dataclass(frozen=True)
class Race:
    day: date
    name: str


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:19]).date()
        except ValueError:
            return None
    return None


def chart_span(chart: ChartData) -> Optional[Tuple[date, date]]:
    """The days a date-axis chart covers, or ``None`` for any other chart."""
    if chart.x_axis.kind is not AxisKind.DATE:
        return None
    days = sorted({
        d for trace in chart.traces if any(v is not None for v in trace.y)
        for d in (_as_date(x) for x in trace.x) if d is not None
    })
    if not days:
        return None
    end = days[-1]
    if len(days) > 1:
        step = timedelta(days=median((b - a).days for a, b in zip(days, days[1:])))
        if step >= _BINNED_STEP:
            end += step - timedelta(days=1)
    return days[0], end


def outputs_span(outputs: Iterable[PlotOutput]) -> Optional[Tuple[date, date]]:
    """The union of every date-axis chart's span — one calendar query for a page."""
    spans = [s for output in outputs for s in map(chart_span, output.charts) if s]
    if not spans:
        return None
    return min(s[0] for s in spans), max(s[1] for s in spans)


def pin_markers(
    output: PlotOutput, races: Sequence[Race], today: date, today_label: str,
) -> PlotOutput:
    """A copy of ``output`` with today and the races marked on each chart they fall in.

    A copy, never in place: the output may be the cached object every later
    request is served from.
    """
    charts: List[ChartData] = []
    changed = False
    for chart in output.charts:
        span = chart_span(chart)
        if span is None:
            charts.append(chart)
            continue
        lo, hi = span
        markers = [
            Marker(kind="race", x=race.day.isoformat(), label=race.name)
            for race in races if lo <= race.day <= hi
        ]
        if lo <= today <= hi:
            markers.append(Marker(kind="today", x=today.isoformat(), label=today_label))
        if markers:
            chart = replace(chart, markers=[*chart.markers, *markers])
            changed = True
        charts.append(chart)
    return replace(output, charts=charts) if changed else output
