"""The weekly review: how hard it felt, how it went, and whether it paid.

Three answers about the same week, on one timeline:

* **average RPE** — the curve, on the only axis (1-10);
* **average feeling** — the colour of the week's slab of background;
* **the fitness trend** — the week summary's own tag ("Fitness ↑"), pinned in a
  row above the curve.

Three quantities, three scales, and one of them ordinal with three levels: a
second y-axis would have to pick an alignment between them, and the reader would
read a correlation out of where the lines happen to cross. So only the RPE gets an
axis; the other two get channels that carry no scale (a fill, a tag) and can't
imply one.

**Nothing here invents its own vocabulary.** The feeling colours are
``FEELING_COLOR`` from ``TrainingScreen.tsx``, the tag is the one
``WeekDetailColumn`` draws, and "up / stable / down" uses the same ±1 threshold on
the same Banister CTL as ``weekFitnessTrend``. A reader who learned the tag in
Training already knows what it means here — which is the whole reason to reuse it
rather than design a fourth encoding.

Like :mod:`src.domain.plots.fitness_fatigue`, this reads ``all_summaries()`` —
every activity, cross-sport — rather than the panel's own filtered selection:
training load has no running-vs-cycling comparability problem, and neither has a
rating the athlete typed in. That also keeps this plot's numbers identical to the
week summary's, which counts every sport too.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    Badge,
    Band,
    ChartData,
    PlotOutput,
    Trace,
    TraceKind,
    empty_output,
)
from src.domain.dataset.binning import bin_start, to_date
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.dataset.training_load import daily_training_load, fitness_fatigue_series
from src.domain.gap import theme
from src.domain.plots.base import PlotDefinition, display_window, register
from src.translations import translate

# Fallback display window when the data source doesn't define one (a hand-picked
# activity list, not a time window) — this plot is a continuous weekly timeline,
# which a discrete pick list doesn't naturally define.
_FALLBACK_DISPLAY_DAYS = 182  # ~6 months

# The athlete's three feeling levels, as scored and coloured by the app —
# FEELING_SCORE / FEELING_COLOR in web/components/TrainingScreen.tsx.
_FEELING_SCORE = {"faible": 1, "ok": 2, "fort": 3}
_FEELING_BY_SCORE = ("faible", "ok", "fort")
_FEELING_COLOR = {
    "faible": theme.DANGER,
    "ok": theme.SUNRISE,
    "fort": theme.PRIMARY,
}
# Light enough that the curve, the gridlines and the tags all still read over it.
_FEELING_OPACITY = 0.15

# The fitness tag: arrow, ink/border colour, fill, and the full wording. The
# wording and the ±1 threshold below are read straight from the `ui.*` keys the
# week summary uses, on purpose — two screens telling the same athlete two
# different things about the same week is the one failure mode worth this
# coupling.
_FITNESS_TAG = {
    "increasing": ("↑", theme.MOSS, theme.MOSS_TINT, "ui.training.week.fitness_increasing"),
    "stable": ("→", theme.SUNRISE, theme.SUNRISE_TINT, "ui.training.week.fitness_stable"),
    "decreasing": ("↓", theme.DANGER, theme.DANGER_TINT, "ui.training.week.fitness_decreasing"),
}
# CTL points gained or lost over the week before it counts as a trend — the same
# number as `weekFitnessTrend` in TrainingScreen.tsx.
_TREND_THRESHOLD = 1.0

_RPE_COLOR = "#3A6EA5"  # lake blue — theme.TIME_SCALE_CYCLE[3]
_RPE_MIN, _RPE_MAX = 1.0, 10.0
# Headroom above the RPE scale for the badge row to live in (see ir.Badge).
_BADGE_HEADROOM = 1.5


@dataclass
class _Week:
    """One week's three answers, already aggregated."""

    start: date
    avg_rpe: Optional[float] = None
    rated: int = 0
    feeling: Optional[str] = None
    delta: Optional[float] = None
    trend: Optional[str] = None


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    summaries = resolved.all_summaries()
    if not summaries:
        return empty_output(translate("plot.no_data", lang))

    today = date.today()
    lo, hi = display_window(
        resolved, fallback_end=today, fallback_days=_FALLBACK_DISPLAY_DAYS,
    )
    weeks = _weeks(summaries, lo, min(hi, today), today)
    if not weeks:
        return empty_output(translate("plot.no_data", lang))
    if not any(w.avg_rpe is not None or w.feeling is not None for w in weeks):
        # The tags alone would just restate the Fitness & Fatigue chart.
        return empty_output(translate("plot.weekly_feel.no_entries", lang))

    x = [w.start for w in weeks]
    return PlotOutput(charts=[ChartData(
        title=translate("plot.weekly_feel.label", lang),
        x_axis=Axis(title="", kind=AxisKind.DATE),
        y_axis=Axis(
            title=translate("plot.weekly_feel.y", lang),
            kind=AxisKind.LINEAR,
            tick_format=",.0f",
            range=[_RPE_MIN, _RPE_MAX + _BADGE_HEADROOM],
            dtick=2,
        ),
        traces=[
            Trace(
                name=translate("plot.weekly_feel.rpe", lang),
                x=x,
                y=[w.avg_rpe for w in weeks],
                kind=TraceKind.LINE,
                color=_RPE_COLOR,
                markers=True,
                marker_size=8,
                hover_text=[_hover(w, lang) for w in weeks],
                hover_template="%{x|%d/%m/%Y}<br>%{customdata}<extra></extra>",
            ),
            *_feeling_legend(lang, x[0]),
        ],
        bands=[
            Band(
                x0=_midweek(w.start, -3.5),
                x1=_midweek(w.start, 3.5),
                color=_FEELING_COLOR[w.feeling],
                opacity=_FEELING_OPACITY,
            )
            for w in weeks if w.feeling is not None
        ],
        badges=[_badge(w, lang) for w in weeks if w.trend is not None],
        height=340,
        caption=translate("plot.weekly_feel.caption", lang),
    )])


# --- Aggregation -----------------------------------------------------------

def _weeks(summaries, lo: date, hi: date, today: date) -> List[_Week]:
    """One :class:`_Week` per calendar week in ``[lo, hi]``, gaps included.

    Every week in the range is present even when nothing happened in it: a
    missing week has to read as a hole in the curve, not as a week that quietly
    disappeared and pulled its neighbours together.
    """
    rpe: Dict[date, List[int]] = {}
    feelings: Dict[date, List[int]] = {}
    for summary in summaries:
        if summary.rpe is None and summary.feeling is None:
            continue
        week = bin_start(to_date(summary.start_date), "week")
        if summary.rpe is not None:
            rpe.setdefault(week, []).append(int(summary.rpe))
        score = _FEELING_SCORE.get(summary.feeling or "")
        if score is not None:
            feelings.setdefault(week, []).append(score)

    fitness = _fitness_by_date(summaries, hi)
    out: List[_Week] = []
    week = bin_start(lo, "week")
    last = bin_start(hi, "week")
    while week <= last:
        values = rpe.get(week, [])
        scores = feelings.get(week, [])
        delta = _fitness_delta(fitness, week, today)
        out.append(_Week(
            start=week,
            avg_rpe=(sum(values) / len(values)) if values else None,
            rated=len(values),
            feeling=_average_feeling(scores),
            delta=delta,
            trend=_trend(delta),
        ))
        week += timedelta(days=7)
    return out


def _fitness_by_date(summaries, end: date) -> Dict[date, float]:
    """The Banister CTL, day by day, from the first day of load to ``end``.

    Fitted over the athlete's *whole* history — the 42-day time constant has to
    be warmed up long before the visible window starts — then read per day.
    """
    daily, _missing = daily_training_load(summaries)
    if not daily:
        return {}
    dates, fitness, _fatigue = fitness_fatigue_series(daily, min(daily), end)
    return dict(zip(dates, fitness))


def _fitness_delta(
    fitness: Dict[date, float], week: date, today: date,
) -> Optional[float]:
    """CTL gained or lost over the week, ``None`` when it can't be known yet.

    The week in progress is measured up to today rather than to Sunday — same as
    the week summary's ``effectiveEnd`` — so the current week has a trend from its
    first day instead of waiting for the weekend.
    """
    if week > today:
        return None
    end = min(week + timedelta(days=6), today)
    start_value, end_value = fitness.get(week), fitness.get(end)
    if start_value is None or end_value is None:
        return None
    return end_value - start_value


def _trend(delta: Optional[float]) -> Optional[str]:
    if delta is None:
        return None
    if delta > _TREND_THRESHOLD:
        return "increasing"
    if delta < -_TREND_THRESHOLD:
        return "decreasing"
    return "stable"


def _average_feeling(scores: List[int]) -> Optional[str]:
    """The week's feeling: the mean of its sessions', rounded back to a level.

    Rounded half *up*, matching the `Math.round` the week summary uses — Python's
    own ``round`` would send 2.5 down to "ok" where the app sends it up to "fort",
    and one card contradicting the other over the same week is exactly what this
    plot is meant not to do.
    """
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    level = min(max(int(mean + 0.5), 1), len(_FEELING_BY_SCORE))
    return _FEELING_BY_SCORE[level - 1]


# --- Rendering pieces ------------------------------------------------------

def _midweek(week: date, offset_days: float) -> datetime:
    """A band edge, as a datetime so half-day offsets survive.

    Bands are centred on their week's Monday, which is where the marker sits, so
    each point stands in the middle of its own colour rather than on its left edge.
    """
    return datetime.combine(week, time.min) + timedelta(days=offset_days)


def _badge(week: _Week, lang: str) -> Badge:
    """The week's fitness tag: "Fitness ↑", falling back to the arrow alone.

    Which of the two a reader gets is the renderer's call, on the pixels the
    figure actually got — thirty of these on a phone can only be arrows.

    Sentence case, where the CSS badge upper-cases itself: caps are ~15% wider,
    and that width is the whole reason a tag has to fall back to its arrow. The
    wording is what has to match the week summary, not the letter case.
    """
    arrow, ink, fill, _word_key = _FITNESS_TAG[week.trend]
    label = translate("ui.training.week.fitness_label", lang)
    return Badge(
        x=week.start,
        text=f"{label} {arrow}",
        color=ink,
        fill=fill,
        short=arrow,
    )


def _feeling_legend(lang: str, x0: date) -> List[Trace]:
    """Three marker-only traces carrying no point — the legend key for the bands.

    A band cannot legend itself, and a background that only means something if you
    already know the colour code means nothing. These draw nothing (their single
    y is null) and exist to put "Feeling: strong / ok / weak" in the legend, in
    the same words the athlete picked when rating the session.
    """
    label = translate("ui.training.session.feeling_short", lang)
    return [
        Trace(
            name=f"{label} · {translate(f'ui.training.session.feeling_{level}', lang)}",
            x=[x0],
            y=[None],
            kind=TraceKind.SCATTER,
            color=_FEELING_COLOR[level],
            marker_size=11,
            opacity=0.55,
        )
        for level in reversed(_FEELING_BY_SCORE)
    ]


def _hover(week: _Week, lang: str) -> str:
    """The week's three answers in one hover, whichever mark is under the cursor."""
    rpe_label = translate("ui.training.session.rpe_short", lang)
    feel_label = translate("ui.training.session.feeling_short", lang)
    fitness_label = translate("ui.training.week.fitness_label", lang)

    if week.avg_rpe is None:
        rpe = f"{rpe_label} —"
    else:
        rpe = translate("plot.weekly_feel.hover_rpe", lang).format(
            label=rpe_label, value=f"{week.avg_rpe:.1f}", count=week.rated,
        )
    feeling = (
        f"{feel_label} —" if week.feeling is None
        else f"{feel_label} {translate(f'ui.training.session.feeling_{week.feeling}', lang)}"
    )
    if week.trend is None:
        fitness = f"{fitness_label} —"
    else:
        # The wording already carries "Fitness" ("Fitness en hausse"), so only the
        # delta is added to it.
        word = translate(_FITNESS_TAG[week.trend][3], lang)
        fitness = f"{word} ({week.delta:+.1f})"
    return f"{rpe}<br>{feeling}<br>{fitness}"


register(PlotDefinition(
    key="weekly_feel",
    label_key="plot.weekly_feel.label",
    description_key="plot.weekly_feel.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=[],
    requires_streams=False,
    # Reads `all_summaries()`, not the panel's window/filter match — so, exactly
    # as for `fitness_fatigue`, `resolved.is_empty` is the wrong gate and
    # `compute()` carries its own empty guards instead.
    requires_data=False,
    # RPE and feeling are typed in after the import, on activities whose ids do
    # not change — so the render signature has to fold them in, or a session
    # rated this morning would keep showing this morning's cached chart.
    reads_ratings=True,
    category_key="plotcat.trends",
))
