"""Optional pre-race load: how far recent Banister fatigue exceeds fitness.

    pre_race_exposure = max(0, ATL / CTL − 1)

ATL and CTL are the app's existing Banister fatigue (7-day) and fitness (42-day)
series over Strava Relative Effort (:mod:`src.domain.dataset.training_load`). The
term is disabled by default (see :class:`~src.domain.durability.config.PreRaceLoadConfig`)
and returns ``None`` — never a zero pretending to be data — unless enabled and
enough activities carry a Relative Effort.
"""

from datetime import date, timedelta
from typing import Optional, Sequence

from src.domain.dataset.binning import to_date
from src.domain.dataset.training_load import daily_training_load, fitness_fatigue_series
from src.domain.durability.config import PreRaceLoadConfig


def pre_race_exposure(summaries: Sequence, today: date, lookback_days: int,
                      config: PreRaceLoadConfig) -> Optional[float]:
    if not config.enabled:
        return None
    since = today - timedelta(days=lookback_days)
    recent = [s for s in summaries if since <= to_date(s.start_date) <= today]
    rated = [s for s in recent if s.relative_effort is not None]
    if len(rated) < config.min_rated_activities:
        return None
    daily, _ = daily_training_load(rated)
    _, fitness, fatigue = fitness_fatigue_series(daily, since, today)
    if not fitness or fitness[-1] <= 0:
        return None
    return max(0.0, fatigue[-1] / fitness[-1] - 1.0)
