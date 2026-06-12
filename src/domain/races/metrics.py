"""Per-race metrics and time series for cross-race comparison.

Turns a single ``ActivityStream`` into:
  - ``RaceMetrics``: scalar summary stats (distance, D+, time, paces, power).
  - ``RaceSeries``: aligned per-step series for the evolution plots, switchable
    between a time (s) and a distance (m) x-axis.

Gradient-adjusted pace uses the Strava reference GAP model (the published
"balanced runner" curve), which maps gradient (m/km) to a speed-adjuster
factor: ``gap_speed = speed * factor``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.domain.gap.reference_curves import balanced_runner
from src.domain.models.activity import ActivityStream

# Minimum speed (m/s) below which pace is undefined (standing / walking pauses).
# ~0.5 m/s ≈ 33 min/km — slower than that we blank the line rather than spike it.
_MIN_PACE_SPEED = 0.5

# Mechanical running-power model: P = m * v * (Cr + g * s).
_GRAVITY = 9.81  # m/s²
_COST_OF_TRANSPORT = 1.0  # J·kg⁻¹·m⁻¹ (typical range 0.9–1.1)


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
    Undefined points are NaN so matplotlib leaves a gap rather than spiking.
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


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Symmetric rolling mean with edge padding; preserves length."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + values.size]


def compute_power_series(
    *,
    speed_m_per_s: np.ndarray,
    gradient_m_per_km: np.ndarray,
    mass_kg: Optional[float],
) -> Optional[np.ndarray]:
    """Mechanical running power per step, in watts.

    Uses ``P = m * v * (Cr + g * s)`` where ``s`` is the slope as a rise/run
    fraction (our gradient is m/km, i.e. ``slope = gradient / 1000``). Per-kg
    power is floored at 0, so steep descents drive power toward 0 rather than
    negative.

    Returns ``None`` when ``mass_kg`` is missing — power needs the runner's
    weight (set on the Home page), so the table/plots stay gated until then.
    """
    if mass_kg is None:
        return None
    speed = np.asarray(speed_m_per_s, dtype=float)
    slope = np.asarray(gradient_m_per_km, dtype=float) / 1000.0
    power_per_kg = np.maximum(speed * (_COST_OF_TRANSPORT + _GRAVITY * slope), 0.0)
    return mass_kg * power_per_kg


def compute_race(
    stream: ActivityStream,
    label: str,
    *,
    smoothing_window: int = 30,
    mass_kg: Optional[float] = None,
) -> tuple[RaceMetrics, RaceSeries]:
    """Compute summary metrics and plotting series for one race.

    ``smoothing_window`` is in samples (~seconds for 1 Hz Strava streams) and
    only smooths the series/derived gradient — the scalar averages are computed
    from raw totals, so they stay independent of the smoothing choice.
    ``mass_kg`` (runner weight) enables the power series; omit it and power
    stays ``None``.
    """
    time = np.asarray(stream.time, dtype=float)
    distance = np.asarray(stream.distance, dtype=float)
    altitude = np.asarray(stream.altitude, dtype=float)
    heartrate = np.asarray(stream.heartrate, dtype=float)

    if time.size < 2:
        raise ValueError(f"Race '{label}' has too few samples to analyse.")

    delta_time = np.diff(time)
    delta_dist = np.diff(distance)
    delta_alt = np.diff(altitude)

    # Instantaneous speed (m/s); 0 where the runner is stationary.
    speed = np.divide(
        delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0
    )
    # Gradient (m/km); 0 over near-stationary steps to avoid blow-ups.
    moved = delta_dist >= 0.1
    gradient = np.divide(
        delta_alt, delta_dist, out=np.zeros_like(delta_dist), where=moved
    ) * 1000.0
    gradient[~moved] = 0.0

    speed_sm = _moving_average(speed, smoothing_window)
    gradient_sm = _moving_average(gradient, smoothing_window)
    heartrate_sm = _moving_average(heartrate[1:], smoothing_window)

    factor = gradient_adjustment_factor(gradient_sm)
    gap_speed = speed_sm * factor

    gap_pace = np.where(
        gap_speed > _MIN_PACE_SPEED, (1000.0 / gap_speed) / 60.0, np.nan
    )

    power = compute_power_series(
        speed_m_per_s=speed_sm,
        gradient_m_per_km=gradient_sm,
        mass_kg=mass_kg,
    )
    power_to_hr = None
    if power is not None:
        power_to_hr = np.divide(
            power, heartrate_sm, out=np.full_like(power, np.nan), where=heartrate_sm > 0
        )

    series = RaceSeries(
        label=label,
        time_s=time[1:] - time[0],
        distance_m=distance[1:] - distance[0],
        gap_pace_s_per_km=gap_pace * 60.0,  # store seconds/km for a consistent unit
        heartrate=heartrate_sm,
        power_w=power,
        power_to_hr=power_to_hr,
    )

    metrics = _summary_metrics(
        label=label,
        time=time,
        distance=distance,
        altitude=altitude,
        delta_dist=delta_dist,
        factor=factor,
        power=power,
        smoothing_window=smoothing_window,
    )
    return metrics, series


def _summary_metrics(
    *,
    label: str,
    time: np.ndarray,
    distance: np.ndarray,
    altitude: np.ndarray,
    delta_dist: np.ndarray,
    factor: np.ndarray,
    power: Optional[np.ndarray],
    smoothing_window: int,
) -> RaceMetrics:
    total_time = float(time[-1] - time[0])
    total_dist = float(distance[-1] - distance[0])

    avg_speed = total_dist / total_time if total_time > 0 else 0.0
    avg_pace = 1000.0 / avg_speed if avg_speed > 0 else float("nan")

    # GAP-adjusted distance = Σ (step distance × gradient factor).
    gap_dist = float(np.nansum(delta_dist * factor))
    avg_gap_speed = gap_dist / total_time if total_time > 0 else 0.0
    avg_gap_pace = 1000.0 / avg_gap_speed if avg_gap_speed > 0 else float("nan")

    # D+: positive altitude diffs after light smoothing to tame barometric noise.
    altitude_sm = _moving_average(altitude, smoothing_window)
    gains = np.diff(altitude_sm)
    elevation_gain = float(np.sum(gains[gains > 0]))

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
