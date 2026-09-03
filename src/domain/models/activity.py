from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class ActivityStream:
    """Raw per-second telemetry for a single activity.

    When per-second streams are unavailable (manual entries, activities Strava
    won't serve streams for), ``has_streams`` is ``False`` and the arrays are
    empty; the ``summary_*`` scalars then carry the activity-level totals from
    Strava's summary so distance/elevation/time analyses can still use it.
    """
    activity_id: int
    sport_type: str
    time: np.ndarray
    distance: np.ndarray
    altitude: np.ndarray
    heartrate: np.ndarray
    # Power-meter watts, as Strava sends them (a metered ride, or a run whose
    # watch reported "running power"). NaN-filled like altitude/heartrate when
    # absent. Carried verbatim for every sport; whether they are *used* is the
    # feature pipeline's call, and it uses them for everything except running
    # (see build_activity_features). Never a modelled estimate of ours.
    watts: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))
    # When the activity took place. Optional so older callers/mocks stay valid;
    # used by the app to filter pre-fetched history by date without re-fetching.
    start_date: Optional[datetime] = None
    # False when no per-second streams were available (summary-only fallback).
    has_streams: bool = True
    # Activity-level totals from the Strava summary; used when has_streams is
    # False (and as a cross-check otherwise). ``None`` when unknown.
    summary_distance_m: Optional[float] = None
    summary_moving_time_s: Optional[float] = None
    summary_elevation_gain_m: Optional[float] = None
    # Strava's Relative Effort (``suffer_score``) — its training-load score for the
    # session. Carried on the summary rather than derived from the streams: it is
    # computed by Strava from heart rate against the athlete's own zones, which we
    # do not have. ``None`` for activities without heart rate.
    summary_relative_effort: Optional[float] = None
