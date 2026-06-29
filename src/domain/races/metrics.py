"""Per-race metrics and time series for cross-race comparison.

Turns a single ``ActivityStream`` into:
  - ``RaceMetrics``: scalar summary stats (distance, D+, time, paces, power).
  - ``RaceSeries``: aligned per-step series for the evolution plots, switchable
    between a time (s) and a distance (m) x-axis.

Smoothing happens in two stages (see :mod:`src.domain.races.smoothing`):
  1. a time-domain rolling mean on each raw stream — 15 s for speed/pace, 60 s
     for heart rate and altitude;
  2. a distance-domain median → Savitzky–Golay pass on the altitude-derived
     gradient and on altitude (for total elevation gain).
GAP is derived pointwise from the smoothed pace and gradient. Power is derived
pointwise then passed through its own 15 s rolling mean. Gradient-adjusted pace
uses the Strava reference GAP model (the published "balanced runner" curve),
which maps gradient (m/km) to a speed-adjuster factor.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.domain.gap.reference_curves import balanced_runner
from src.domain.models.activity import ActivityStream
from src.domain.races.smoothing import (
    SmoothingParams,
    apply_signal_filters,
    default_smoothing_params,
)

# A jump larger than this between consecutive samples means the watch was paused.
# Such bridging steps are excluded from every series and every aggregate stat.
_PAUSE_THRESHOLD_S = 60.0

# Instantaneous speed is clipped to a plausible running range (3–20 km/h);
# anything outside is a GPS glitch or a stop. The bounds map directly to
# pace/GAP limits: 20 km/h → 180 s/km (fast), 3 km/h → 1200 s/km (slow).
_MAX_SPEED_M_PER_S = 20.0 / 3.6
_MIN_SPEED_M_PER_S = 3.0 / 3.6
_MIN_PACE_S_PER_KM = 1000.0 / _MAX_SPEED_M_PER_S  # = 180 s/km (20 km/h)
_MAX_PACE_S_PER_KM = 1000.0 / _MIN_SPEED_M_PER_S  # = 1200 s/km (3 km/h)

# Mechanical running-power model: P = m * v * (Cr + g * s).
_GRAVITY = 9.81  # m/s²
_COST_OF_TRANSPORT = 1.0  # J·kg⁻¹·m⁻¹ (typical range 0.9–1.1)
_MIN_SLOPE = -0.05  # cap downhill slope at -5% to avoid unrealistic power reduction


@dataclass
class RaceMetrics:
    """Scalar summary of one race."""
    label: str
    distance_km: float
    elevation_gain_m: float
    time_s: float
    avg_pace_s_per_km: float
    avg_gap_pace_s_per_km: float
    avg_power_w: Optional[float] = None


@dataclass
class RaceSeries:
    """Aligned per-step series for one race (length N-1 for an N-sample stream).

    ``time_s`` and ``distance_m`` are the two x-axes; the rest are y-series.
    Undefined points are NaN so the chart leaves a gap rather than spiking.
    """
    label: str
    time_s: np.ndarray
    distance_m: np.ndarray
    gap_pace_s_per_km: np.ndarray
    heartrate: np.ndarray
    power_w: Optional[np.ndarray] = None
    power_to_hr: Optional[np.ndarray] = None


def gradient_adjustment_factor(elevation_gain_m_per_km: np.ndarray) -> np.ndarray:
    """Strava GAP speed-adjuster for each gradient, via the reference curve.

    Linearly interpolates the published "balanced runner" curve and clamps to
    its endpoints for gradients beyond the modelled range.
    """
    curve = balanced_runner()
    return np.interp(
        np.asarray(elevation_gain_m_per_km, dtype=float),
        curve.bin_centers,
        curve.means,
    )


def compute_power_series(
    *,
    speed_m_per_s: np.ndarray,
    gradient_m_per_km: np.ndarray,
    mass_kg: Optional[float],
) -> Optional[np.ndarray]:
    """Mechanical running power per step, in watts.

    Uses ``P = m * v * (Cr + g * s)`` where ``s`` is the slope as a rise/run
    fraction (our gradient is m/km, i.e. ``slope = gradient / 1000``), capped at
    ``_MIN_SLOPE`` for steep downhills to avoid unrealistic power reduction.
    Per-kg power is floored at 0.

    Returns ``None`` when ``mass_kg`` is missing — power needs the runner's
    weight (set on the Home page), so the table/plots stay gated until then.
    """
    if mass_kg is None:
        return None
    speed = np.asarray(speed_m_per_s, dtype=float)
    slope = np.maximum(np.asarray(gradient_m_per_km, dtype=float) / 1000.0, _MIN_SLOPE)
    power_per_kg = np.maximum(speed * (_COST_OF_TRANSPORT + _GRAVITY * slope), 0.0)
    return mass_kg * power_per_kg


def compute_race(
    stream: ActivityStream,
    label: str,
    *,
    mass_kg: Optional[float] = None,
    smoothing: Optional[SmoothingParams] = None,
) -> tuple[RaceMetrics, RaceSeries]:
    """Compute summary metrics and plotting series for one race.

    ``smoothing`` selects, per signal, an optional rolling mean (time domain)
    and/or Savitzky–Golay pass (distance domain); see
    :func:`default_smoothing_params`. Gradient and elevation gain derive from
    the smoothed altitude; GAP from the smoothed pace and gradient; power is
    derived pointwise then runs through its own filters. Watch pauses (time
    jumps > 60 s) are cut out: the bridging step is excluded from every series
    and aggregate, and the x-axes use moving time/distance so pauses don't
    appear. ``mass_kg`` (runner weight) enables the power series.
    """
    smoothing = smoothing or default_smoothing_params()
    poly = smoothing.savgol_polyorder

    time = np.asarray(stream.time, dtype=float)
    distance = np.asarray(stream.distance, dtype=float)
    altitude = np.asarray(stream.altitude, dtype=float)
    heartrate = np.asarray(stream.heartrate, dtype=float)

    if time.size < 2:
        raise ValueError(f"Race '{label}' has too few samples to analyse.")

    delta_time = np.diff(time)
    delta_dist = np.diff(distance)
    step_distance = distance[1:]  # cumulative distance at each per-step sample
    timestamps = time[1:]

    # A step that spans a pause (big time jump) bridges across missing data —
    # exclude it everywhere so it can't create phantom pace/gradient/distance.
    moving_step = (delta_time > 0) & (delta_time <= _PAUSE_THRESHOLD_S)

    # --- Altitude (full grid) → gradient + elevation gain --------------------
    altitude_smoothed = apply_signal_filters(
        altitude, timestamps_s=time, distance_m=distance,
        config=smoothing.altitude, polyorder=poly,
    )
    delta_alt = np.diff(altitude_smoothed)
    moved = delta_dist >= 0.1
    gradient_pct = np.divide(  # rise/run × 100
        delta_alt, delta_dist, out=np.zeros_like(delta_dist), where=moved
    ) * 100.0
    gradient_pct[~moved] = 0.0
    # GAP reference curve and the power model work in m/km (= %·10).
    gradient_m_per_km = gradient_pct * 10.0
    factor = gradient_adjustment_factor(gradient_m_per_km)

    # --- Pace (clipped to 3–20 km/h) and heart rate --------------------------
    speed = np.divide(
        delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0
    )
    speed = np.clip(speed, _MIN_SPEED_M_PER_S, _MAX_SPEED_M_PER_S)
    pace = np.divide(
        1000.0, speed, out=np.full_like(speed, np.nan), where=moving_step
    )
    pace_smoothed = apply_signal_filters(
        pace, timestamps_s=timestamps, distance_m=step_distance,
        config=smoothing.pace, polyorder=poly,
    )
    heartrate_smoothed = apply_signal_filters(
        heartrate[1:], timestamps_s=timestamps, distance_m=step_distance,
        config=smoothing.heartrate, polyorder=poly,
    )

    # --- GAP — pointwise from smoothed pace/gradient; clipped to 180–1200 ----
    gap_pace = np.clip(
        pace_smoothed / factor, _MIN_PACE_S_PER_KM, _MAX_PACE_S_PER_KM
    )

    # --- Power — pointwise, then its own configured filters ------------------
    speed_smoothed = np.divide(
        1000.0, pace_smoothed, out=np.full_like(pace_smoothed, np.nan),
        where=pace_smoothed > 0,
    )
    power = compute_power_series(
        speed_m_per_s=speed_smoothed,
        gradient_m_per_km=gradient_m_per_km,
        mass_kg=mass_kg,
    )
    power_to_hr = None
    if power is not None:
        power = apply_signal_filters(
            power, timestamps_s=timestamps, distance_m=step_distance,
            config=smoothing.power, polyorder=poly,
        )
        power_to_hr = np.divide(
            power, heartrate_smoothed,
            out=np.full_like(power, np.nan), where=heartrate_smoothed > 0,
        )

    # Moving-time / moving-distance axes: paused steps contribute 0, collapsing
    # the gap so the curves stay continuous and the x-axes show only real effort.
    moving_time = np.cumsum(np.where(moving_step, delta_time, 0.0))
    moving_distance = np.cumsum(np.where(moving_step, delta_dist, 0.0))

    series = RaceSeries(
        label=label,
        time_s=moving_time,
        distance_m=moving_distance,
        gap_pace_s_per_km=gap_pace,
        heartrate=heartrate_smoothed,
        power_w=power,
        power_to_hr=power_to_hr,
    )

    metrics = _summary_metrics(
        label=label,
        altitude_smoothed=altitude_smoothed,
        delta_time=delta_time,
        delta_dist=delta_dist,
        factor=factor,
        power=power,
        moving_step=moving_step,
    )
    return metrics, series


def _summary_metrics(
    *,
    label: str,
    altitude_smoothed: np.ndarray,
    delta_time: np.ndarray,
    delta_dist: np.ndarray,
    factor: np.ndarray,
    power: Optional[np.ndarray],
    moving_step: np.ndarray,
) -> RaceMetrics:
    # Every aggregate sums over moving steps only, so paused intervals are
    # excluded from time, distance, GAP and elevation gain.
    total_time = float(np.sum(delta_time[moving_step]))
    total_dist = float(np.sum(delta_dist[moving_step]))

    avg_speed = total_dist / total_time if total_time > 0 else 0.0
    avg_pace = 1000.0 / avg_speed if avg_speed > 0 else float("nan")

    # GAP-adjusted distance = Σ (step distance × gradient factor).
    gap_dist = float(np.nansum((delta_dist * factor)[moving_step]))
    avg_gap_speed = gap_dist / total_time if total_time > 0 else 0.0
    avg_gap_pace = 1000.0 / avg_gap_speed if avg_gap_speed > 0 else float("nan")

    # D+: positive deltas on the smoothed altitude (whatever filters the user
    # chose), with paused/bridging steps dropped so a pause adds no phantom climb.
    gains = np.diff(altitude_smoothed)
    rising = moving_step & (gains > 0)
    elevation_gain = float(np.sum(gains[rising]))

    avg_power = float(np.nanmean(power)) if power is not None and power.size else None

    return RaceMetrics(
        label=label,
        distance_km=total_dist / 1000.0,
        elevation_gain_m=elevation_gain,
        time_s=total_time,
        avg_pace_s_per_km=avg_pace,
        avg_gap_pace_s_per_km=avg_gap_pace,
        avg_power_w=avg_power,
    )
