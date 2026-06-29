"""Data shapes for the long-term-progress analysis.

The page works on *every* loaded activity (run + trail run) and looks at season
-over-season trends. The expensive per-activity work — sliding-window best
efforts and per-second gradient classification — is done once and summarized
into :class:`ActivityProgress`; every plot then re-aggregates these cheap
summaries, so changing a UI option never re-touches the raw streams.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --- Personal-record distances --------------------------------------------
# (display label, distance in metres), shortest → longest. The label is a
# stable key used across the table, the plot legend and the translations.
PR_DISTANCES: List[Tuple[str, float]] = [
    ("1 km", 1_000.0),
    ("3 km", 3_000.0),
    ("5 km", 5_000.0),
    ("10 km", 10_000.0),
    ("Semi", 21_097.5),
    ("Marathon", 42_195.0),
    ("50 km", 50_000.0),
    ("100 km", 100_000.0),
    ("150 km", 150_000.0),
]

# --- Gradient bands --------------------------------------------------------
# (key, lower bound %, upper bound %), steepest descent → steepest ascent.
# Bounds are [lower, upper): a step counts in the band whose lower ≤ grad < upper
# (the open ±inf ends catch everything beyond the modelled range).
GRADIENT_BANDS: List[Tuple[str, float, float]] = [
    ("steep_descent", float("-inf"), -12.0),
    ("gentle_descent", -12.0, -3.0),
    ("flat", -3.0, 3.0),
    ("gentle_ascent", 3.0, 12.0),
    ("steep_ascent", 12.0, float("inf")),
]
GRADIENT_BAND_KEYS: List[str] = [band[0] for band in GRADIENT_BANDS]


@dataclass
class ActivityProgress:
    """Per-activity summary feeding every long-term-progress plot.

    ``band_seconds`` maps each :data:`GRADIENT_BANDS` key to the moving time
    (seconds) spent in that band. ``best_efforts`` maps each
    :data:`PR_DISTANCES` label to the fastest time (seconds) to cover that
    distance in this activity, or ``None`` when the activity was too short.
    ``power_to_hr`` is the session-average power-to-heart-rate ratio (W/bpm), or
    ``None`` when the runner's weight is unknown (power can't be modelled).
    """

    activity_id: int
    date: datetime
    sport_type: str
    distance_m: float
    elevation_gain_m: float
    moving_seconds: float
    band_seconds: Dict[str, float]
    best_efforts: Dict[str, Optional[float]]
    power_to_hr: Optional[float] = None
