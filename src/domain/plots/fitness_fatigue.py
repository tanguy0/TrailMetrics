"""Fitness & Fatigue — the Banister two-time-constant impulse-response model.

The one plot type in this app that is deliberately cross-sport: every other
plot stays within a single sport family (see ``_filter_family`` in
:mod:`src.usecases.resolve_panel_data`), because GAP, modelled power and
PR/gradient-band figures aren't comparable between a foot split and a bike
split. Training load has no such problem — Strava's Relative Effort is derived
from heart rate against the athlete's own zones, sport-agnostic by
construction — so this is the one plot that reads every activity regardless of
the panel's sport filter, via :meth:`ResolvedPanelData.all_summaries`.

The model itself (see :mod:`src.domain.dataset.training_load` for the maths)
runs over the athlete's *whole* history — so the 42-day fitness time constant
is warmed up before the visible range even starts — and is only then sliced
down to whatever date range the panel's own data source selects here.

This module always draws both curves together. The same model is also
reusable as two ordinary, individually-selectable metrics inside
:mod:`src.domain.plots.metric_trend` — see that module's ``_is_ff``.
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

from src.domain.charts.ir import Axis, AxisKind, ChartData, PlotOutput, Trace, TraceKind, empty_output
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.dataset.training_load import daily_training_load, fitness_fatigue_series
from src.domain.plots.base import PlotDefinition, register
from src.translations import translate

# Fallback display window when the data source doesn't define one (a
# hand-picked activity list, not a time window) — this plot is a continuous
# daily timeline, which a discrete pick list doesn't naturally define.
_FALLBACK_DISPLAY_DAYS = 182  # ~6 months

_FITNESS_COLOR = "#3E7C59"  # forest — slow, steady
_FATIGUE_COLOR = "#C9622B"  # terracotta — fast, reactive


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    summaries = resolved.all_summaries()
    daily, missing_count = daily_training_load(summaries)
    if not daily:
        return empty_output(translate("plot.no_data", lang))

    start = min(daily)
    today = date.today()
    dates, fitness, fatigue = fitness_fatigue_series(daily, start, today)

    lo, hi = _display_window(resolved, fallback_end=today)
    indices = [i for i, d in enumerate(dates) if lo <= d <= hi]
    if not indices:
        return empty_output(translate("plot.no_data", lang))

    x = [dates[i] for i in indices]
    fitness_y = [fitness[i] for i in indices]
    fatigue_y = [fatigue[i] for i in indices]

    notes: List[str] = []
    if missing_count:
        notes.append(
            translate("plot.fitness_fatigue.missing_relative_effort", lang)
            .format(count=missing_count)
        )

    chart = ChartData(
        title=translate("plot.fitness_fatigue.label", lang),
        x_axis=Axis(title="", kind=AxisKind.DATE),
        y_axis=Axis(
            title=translate("plot.fitness_fatigue.y", lang),
            kind=AxisKind.LINEAR, tick_format=",.0f",
        ),
        traces=[
            Trace(
                name=translate("plot.fitness_fatigue.fitness", lang),
                x=x, y=fitness_y, kind=TraceKind.LINE,
                color=_FITNESS_COLOR, width=10.4,
            ),
            Trace(
                name=translate("plot.fitness_fatigue.fatigue", lang),
                x=x, y=fatigue_y, kind=TraceKind.LINE,
                color=_FATIGUE_COLOR, width=8.0,
            ),
        ],
        height=420,
        hover_mode="x unified",
    )
    return PlotOutput(charts=[chart], notes=notes)


def _display_window(
    resolved: ResolvedPanelData, *, fallback_end: date,
) -> Tuple[date, date]:
    """The panel's own selected date range, or a trailing fallback without one.

    A time window is the only data-source mode with a natural continuous date
    range. This plot borrows a window's ``[start, end]`` purely to decide what
    to *show* — the activities that window matched were already discarded in
    favour of ``all_summaries()`` above, since this plot is cross-sport by
    design.
    """
    starts = [g.window.start for g in resolved.groups if g.window is not None]
    ends = [g.window.end for g in resolved.groups if g.window is not None]
    if starts and ends:
        return min(starts), max(ends)
    return fallback_end - timedelta(days=_FALLBACK_DISPLAY_DAYS), fallback_end


register(PlotDefinition(
    key="fitness_fatigue",
    label_key="plot.fitness_fatigue.label",
    description_key="plot.fitness_fatigue.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=[],
    requires_streams=False,
    # This plot reads `resolved.all_summaries()` — the athlete's whole
    # cross-sport history — not the panel's own window/filter-matched
    # activities, so `resolved.is_empty` (which only reflects that match) is
    # the wrong gate: a narrow window with nothing in it would otherwise
    # short-circuit this to "no activity" before `compute()` ever runs, even
    # with years of history to draw the curve from. `compute()` already has
    # its own empty-history guard (`if not daily: return empty_output(...)`).
    requires_data=False,
    category_key="plotcat.trends",
))
