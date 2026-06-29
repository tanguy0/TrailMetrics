"""Summarize a whole activity history into per-activity progress records.

Runs on the streams already loaded in memory (the app's "fetch once, analyse
many" flow) — never hits Strava. For every activity it computes the data the
long-term-progress plots need: sliding-window best efforts for the PR distances,
plus moving distance, elevation gain and per-gradient-band time derived from a
smoothed altitude trace. The heavy work happens here, once; the page then
re-aggregates the cheap :class:`ActivityProgress` summaries for each view.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from src.domain.models.activity import ActivityStream
from src.domain.progress.models import (
    ActivityProgress,
    GRADIENT_BANDS,
    PR_DISTANCES,
)
from src.domain.progress.records import best_effort_time
from src.domain.races.metrics import compute_power_series
from src.domain.races.smoothing import (
    FilterConfig,
    apply_signal_filters,
    default_smoothing_params,
)
from src.usecases.base import UseCase

# A time jump larger than this between samples means the watch was paused; such
# bridging steps add no real distance/time/climb, so they're excluded.
_PAUSE_THRESHOLD_S = 60.0


@dataclass
class AnalyzeLongTermProgressInput:
    streams: List[ActivityStream] = field(default_factory=list)
    # Runner weight (kg); enables the power-to-HR metric. Left None → power-to-HR
    # stays unavailable (the page gates that plot on it).
    mass_kg: Optional[float] = None


@dataclass
class AnalyzeLongTermProgressOutput:
    activities: List[ActivityProgress] = field(default_factory=list)


def _sport_name(sport_type) -> str:
    """Coerce stravalib's ``RelaxedSportType`` (has ``.root``) to a plain string."""
    return str(getattr(sport_type, "root", sport_type))


def _naive(dt: datetime) -> datetime:
    """Drop tzinfo so season binning/filtering never mixes aware & naive dates."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


class AnalyzeLongTermProgress(UseCase):
    """Turn raw activity streams into :class:`ActivityProgress` summaries.

    ``altitude_filter`` smooths the altitude trace before deriving gradient and
    elevation gain (raw per-second GPS altitude is far too noisy); it defaults
    to the same altitude smoothing the race comparator uses.
    """

    def __init__(
        self,
        altitude_filter: Optional[FilterConfig] = None,
        polyorder: int = 2,
    ):
        self.altitude_filter = altitude_filter or default_smoothing_params().altitude
        self.polyorder = polyorder

    def execute(
        self, params: AnalyzeLongTermProgressInput
    ) -> AnalyzeLongTermProgressOutput:
        activities = [
            ap
            for s in params.streams
            if (ap := self._process(s, params.mass_kg)) is not None
        ]
        return AnalyzeLongTermProgressOutput(activities=activities)

    def _process(
        self, stream: ActivityStream, mass_kg: Optional[float]
    ) -> Optional[ActivityProgress]:
        if not isinstance(stream.start_date, datetime):
            return None  # undated activities can't be placed on a timeline

        time = np.asarray(stream.time, dtype=float)
        distance = np.asarray(stream.distance, dtype=float)
        altitude = np.asarray(stream.altitude, dtype=float)
        heartrate = np.asarray(stream.heartrate, dtype=float)
        n = time.size
        if n < 2 or distance.size != n or altitude.size != n:
            return None

        # Best efforts run on the raw cumulative distance / time streams.
        best_efforts = {
            label: best_effort_time(distance, time, meters)
            for label, meters in PR_DISTANCES
        }

        # Gradient & elevation gain come from a smoothed altitude trace.
        altitude_s = apply_signal_filters(
            altitude,
            timestamps_s=time,
            distance_m=distance,
            config=self.altitude_filter,
            polyorder=self.polyorder,
        )
        delta_time = np.diff(time)
        delta_dist = np.diff(distance)
        delta_alt = np.diff(altitude_s)

        moving = (delta_time > 0) & (delta_time <= _PAUSE_THRESHOLD_S) & (delta_dist > 0)

        moving_distance = float(np.sum(delta_dist[moving]))
        elevation_gain = float(np.sum(delta_alt[moving & (delta_alt > 0)]))
        moving_seconds = float(np.sum(delta_time[moving]))

        gradient_pct = np.divide(
            delta_alt, delta_dist, out=np.zeros_like(delta_dist), where=delta_dist > 0
        ) * 100.0

        band_seconds = {}
        for key, lower, upper in GRADIENT_BANDS:
            in_band = moving & (gradient_pct >= lower) & (gradient_pct < upper)
            band_seconds[key] = float(np.sum(delta_time[in_band]))

        power_to_hr = self._avg_power_to_hr(
            delta_dist=delta_dist,
            delta_time=delta_time,
            gradient_pct=gradient_pct,
            heartrate=heartrate[1:],
            moving=moving,
            mass_kg=mass_kg,
        )

        return ActivityProgress(
            activity_id=stream.activity_id,
            date=_naive(stream.start_date),
            sport_type=_sport_name(stream.sport_type),
            distance_m=moving_distance,
            elevation_gain_m=elevation_gain,
            moving_seconds=moving_seconds,
            band_seconds=band_seconds,
            best_efforts=best_efforts,
            power_to_hr=power_to_hr,
        )

    @staticmethod
    def _avg_power_to_hr(
        *,
        delta_dist: np.ndarray,
        delta_time: np.ndarray,
        gradient_pct: np.ndarray,
        heartrate: np.ndarray,
        moving: np.ndarray,
        mass_kg: Optional[float],
    ) -> Optional[float]:
        """Session-average power-to-HR (W/bpm) = mean power ÷ mean HR.

        Uses the same mechanical power model as the race comparator. Returns
        ``None`` without a weight (power is unmodellable) or when no valid HR
        sample is available.
        """
        speed = np.divide(
            delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0
        )
        power = compute_power_series(
            speed_m_per_s=speed,
            gradient_m_per_km=gradient_pct * 10.0,  # % → m/km
            mass_kg=mass_kg,
        )
        if power is None:
            return None
        valid = moving & np.isfinite(power) & np.isfinite(heartrate) & (heartrate > 0)
        if not valid.any():
            return None
        mean_power = float(np.mean(power[valid]))
        mean_hr = float(np.mean(heartrate[valid]))
        return mean_power / mean_hr if mean_hr > 0 else None
