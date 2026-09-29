"""One athlete's durability model from their past year of running.

Everything read here is limited to the last ``lookback_days`` (365) before
``today``: the reference speed comes from that year's best efforts, the segments
from that year's long runs. Older history says little about current durability.

Storage-free: it reads only through the
:class:`ActivityDataSource` port, and the caller decides how to cache (the race
plan memoizes per athlete, a panel render per memo dict).
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.domain.dataset.binning import to_date
from src.domain.dataset.sport import RUNNING_SPORT_TYPES
from src.domain.durability.capability import ReferenceSpeed, reference_from_best_efforts
from src.domain.durability.config import DEFAULT_CONFIG, DurabilityConfig
from src.domain.durability.personalization import AthleteDurabilityModel, fit_athlete
from src.domain.durability.pre_race import pre_race_exposure
from src.domain.durability.segments import SEGMENT_SPORTS, DurabilitySegment, extract_segments
from src.domain.ports.activity_data import ActivityDataSource, ActivitySummary


def past_year(summaries: Sequence[ActivitySummary], today: date, lookback_days: int
              ) -> List[ActivitySummary]:
    since = today - timedelta(days=lookback_days)
    return [s for s in summaries if since <= to_date(s.start_date) <= today]


def durability_activity_ids(summaries: Sequence[ActivitySummary], today: date,
                            config: DurabilityConfig = DEFAULT_CONFIG) -> Tuple[int, ...]:
    """The long outdoor runs of the past year a fit reads — also its cache key."""
    settings = config.personalization
    runs = [
        s for s in past_year(summaries, today, settings.lookback_days)
        if s.has_streams and s.sport_type in SEGMENT_SPORTS
        and (s.moving_s or 0) >= settings.min_activity_moving_s
    ]
    runs.sort(key=lambda s: s.start_date)
    return tuple(s.activity_id for s in runs[-settings.max_activities:])


def athlete_reference(data: ActivityDataSource, summaries: Sequence[ActivitySummary],
                      today: date, config: DurabilityConfig = DEFAULT_CONFIG) -> ReferenceSpeed:
    ids = [s.activity_id for s in past_year(summaries, today, config.personalization.lookback_days)
           if s.sport_type in RUNNING_SPORT_TYPES]
    features = data.features(ids) if ids else None
    return reference_from_best_efforts(features, config.capability)


def fit_athlete_durability(
    data: ActivityDataSource,
    today: date,
    config: DurabilityConfig = DEFAULT_CONFIG,
    activity_ids: Optional[Sequence[int]] = None,
    memo: Optional[Dict[Any, Any]] = None,
) -> AthleteDurabilityModel:
    """The athlete's durability model; the population model when history is absent.

    ``activity_ids`` narrows the evidence (a panel's selection), always intersected
    with the past-year long runs. ``memo`` caches per-activity segments.
    """
    memo = memo if memo is not None else {}
    summaries = data.summaries()
    reference = athlete_reference(data, summaries, today, config)
    eligible = durability_activity_ids(summaries, today, config)
    if activity_ids is not None:
        wanted = set(int(i) for i in activity_ids)
        eligible = tuple(i for i in eligible if i in wanted)

    segments: List[DurabilitySegment] = []
    excluded: Dict[str, int] = {}
    if reference.known:
        for activity_id in eligible:
            key = ("durability_segments", activity_id, round(reference.speed_m_per_s, 4))
            if key not in memo:
                stream = data.stream(activity_id)
                memo[key] = (
                    extract_segments(stream, reference, config.exposure, config.personalization)
                    if stream is not None else ([], "no_stream")
                )
            found, reason = memo[key]
            segments.extend(found)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1

    model = fit_athlete(segments, reference, config, excluded=excluded)
    model.pre_race_exposure = pre_race_exposure(
        summaries, today, config.personalization.lookback_days, config.pre_race
    )
    return model
