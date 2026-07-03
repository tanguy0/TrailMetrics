from dataclasses import dataclass
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
