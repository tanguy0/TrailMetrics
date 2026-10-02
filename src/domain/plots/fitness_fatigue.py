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

from datetime import date
from typing import Any, Dict, List

from src.domain.charts.ir import Axis, AxisKind, ChartData, PlotOutput, Trace, TraceKind, empty_output
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.dataset.training_load import daily_training_load, fitness_fatigue_series
from src.domain.gap import theme
from src.domain.plots.base import PlotDefinition, display_window, register
from src.translations import translate

# Fallback display window when the data source doesn't define one (a
# hand-picked activity list, not a time window) — this plot is a continuous
# daily timeline, which a discrete pick list doesn't naturally define.
_FALLBACK_DISPLAY_DAYS = 182  # ~6 months

# charts.md § v1.2 — cas particulier: the question is "am I tired?", so the
# fatigue carries the figure's area (terra, 2.2 px); the fitness is a thin
# forest line without one; form stays sun bars around a zero baseline. Both
# lines end on their value. The one figure where the area is not on series 1.
_FITNESS_COLOR = theme.FITNESS  # forest — slow, steady
_FATIGUE_COLOR = theme.FATIGUE  # terra — fast, reactive
_FORM_COLOR = theme.FORM        # sun — the signal of the moment
_FITNESS_WIDTH = 1.5
_FATIGUE_WIDTH = 2.2
# A fresh day (form above zero) reads stronger than a tired one.
_FORM_OPACITY_POSITIVE = 0.75
_FORM_OPACITY_NEGATIVE = 0.4
# Bars take 60 % of each day's slot.
_FORM_BARGAP = 0.4


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    summaries = resolved.all_summaries()
    daily, missing_count = daily_training_load(summaries)
    if not daily:
        return empty_output(translate("plot.no_data", lang))

    start = min(daily)
    today = date.today()
    dates, fitness, fatigue = fitness_fatigue_series(daily, start, today)

    lo, hi = display_window(
        resolved, fallback_end=today, fallback_days=_FALLBACK_DISPLAY_DAYS,
    )
    indices = [i for i, d in enumerate(dates) if lo <= d <= hi]
    if not indices:
        return empty_output(translate("plot.no_data", lang))

    x = [dates[i] for i in indices]
    fitness_y = [fitness[i] for i in indices]
    fatigue_y = [fatigue[i] for i in indices]
    # Form (training stress balance): fitness minus fatigue, read against zero.
    form_y = [f - g for f, g in zip(fitness_y, fatigue_y)]

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
                color=_FITNESS_COLOR, width=_FITNESS_WIDTH,
                area=False, end_label=True,
            ),
            Trace(
                name=translate("plot.fitness_fatigue.fatigue", lang),
                x=x, y=fatigue_y, kind=TraceKind.LINE,
                color=_FATIGUE_COLOR, width=_FATIGUE_WIDTH,
                area=True, end_label=True,
            ),
            Trace(
                name=translate("plot.fitness_fatigue.form", lang),
                x=x, y=form_y, kind=TraceKind.BAR,
                color=_FORM_COLOR,
                point_opacity=[
                    _FORM_OPACITY_POSITIVE if v >= 0 else _FORM_OPACITY_NEGATIVE for v in form_y
                ],
            ),
        ],
        height=420,
        hover_mode="x unified",
        bargap=_FORM_BARGAP,
        family="oscillation",
        baseline=0.0,
        x_bucket="day",
    )
    return PlotOutput(charts=[chart], notes=notes)


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
