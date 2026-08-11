"""Daily training load and the Banister fitness/fatigue impulse-response model.

Shared by two plots — the standalone :mod:`src.domain.plots.fitness_fatigue`
(always both curves together) and :mod:`src.domain.plots.metric_trend` (either
curve as an ordinary, individually-selectable metric) — which is why this lives
here rather than in either plot module: neither should depend on the other.

Two exponentially-weighted moving averages of the same daily training-load
series (Σ Relative Effort per calendar day, 0 on a rest day), differing only in
their time constant:

    fitness[d] = fitness[d-1] + (TL[d] - fitness[d-1]) / 42
    fatigue[d] = fatigue[d-1] + (TL[d] - fatigue[d-1]) / 7

Fatigue reacts ~6x faster than fitness by construction (1/7 vs 1/42 of the gap
per day) — the entire mechanism behind a taper: rest drains fatigue fast while
fitness barely erodes.
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple

from src.domain.dataset.binning import to_date

TAU_FITNESS_DAYS = 42.0
TAU_FATIGUE_DAYS = 7.0


def daily_training_load(summaries) -> Tuple[Dict[date, float], int]:
    """Σ relative_effort per calendar day, plus how many activities had none.

    Takes any iterable of objects with ``.start_date``/``.relative_effort``
    (an :class:`~src.domain.ports.activity_data.ActivitySummary`, cross-sport
    and unfiltered — see ``ResolvedPanelData.all_summaries``). An activity with
    no Relative Effort (no heart rate) contributes nothing to its day rather
    than being treated as a full rest day's worth of zero — that would be
    indistinguishable from an actual rest day, so it's dropped and disclosed
    via the returned count instead.
    """
    daily: Dict[date, float] = {}
    missing = 0
    for summary in summaries:
        if summary.relative_effort is None:
            missing += 1
            continue
        day = to_date(summary.start_date)
        daily[day] = daily.get(day, 0.0) + float(summary.relative_effort)
    return daily, missing


def fitness_fatigue_series(
    daily: Dict[date, float], start: date, end: date,
) -> Tuple[List[date], List[float], List[float]]:
    """The two EWMAs, day by day from ``start`` to ``end`` inclusive, both
    seeded at 0 — an athlete's history has to start somewhere."""
    dates: List[date] = []
    fitness_series: List[float] = []
    fatigue_series: List[float] = []
    fitness = fatigue = 0.0
    day = start
    while day <= end:
        load = daily.get(day, 0.0)
        fitness += (load - fitness) / TAU_FITNESS_DAYS
        fatigue += (load - fatigue) / TAU_FATIGUE_DAYS
        dates.append(day)
        fitness_series.append(fitness)
        fatigue_series.append(fatigue)
        day += timedelta(days=1)
    return dates, fitness_series, fatigue_series
