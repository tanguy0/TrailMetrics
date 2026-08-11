"""Modelled cycling power — a physics estimate for rides without a power meter.

Real power-meter watts always win when Strava has them (see
:mod:`src.domain.dataset.features` and :func:`src.domain.races.metrics.compute_race`,
which check an activity's ``watts`` stream before ever calling into this module).
This is the fallback: the classic cycling power-balance equation,

    P = (1/η) [ m·g·v·sinθ + Crr·m·g·v·cosθ + ½·ρ·CdA·v³ + m_eff·v·(dv/dt) ]

collapsed into population-average constants for CdA, Crr and drivetrain
efficiency, because that is all this app knows about most rides, then scaled by
the athlete's actual mass plus a fixed bike/kit allowance.

Like running's modelled power, this needs a weight to produce anything — no
weight on file means no modelled estimate, full stop (see
:func:`compute_cycling_power_series` returning ``None``). Unlike running's,
though, the aero term is *not* proportional to rider mass — it is a fixed
population CdA regardless of who's riding — which is why this model cannot be
cached "per kg at a 1 kg basis" and rescaled later the way running's is: see the
note on ``avg_power_w_modelled`` in :mod:`src.domain.dataset.features`.

Preprocessing matters more than the formula: raw GPS altitude and speed are far
too noisy to differentiate directly into gradient and acceleration, so both are
smoothed first via :mod:`src.domain.races.smoothing` — the same two-stage filter
running's GAP model uses, just with cycling-appropriate windows.
"""

from typing import Optional

import numpy as np

from src.domain.races.smoothing import FilterConfig, apply_signal_filters, rolling_mean_time

_BIKE_KIT_MASS_KG = 9.0  # added to the athlete's weight

# Mass-proportional coefficient v·(A·G + B + C·a), and the mass-independent aero
# coefficient v³·D, both at a reference air density ρ=1.2 kg/m³ (sea level,
# ~15°C) corrected per-ride by _air_density_ratio.
_GRAVITY_ROLLING_COEFF = 10.06  # per unit gradient fraction, per kg
_ROLLING_COEFF = 0.060  # per kg
_INERTIA_COEFF = 1.05  # per kg, per m/s²
_AERO_COEFF = 0.197  # W per (m/s)³, population CdA at ρ=1.2

# Clip the noisiest inputs before they blow up the model: a signed gradient
# beyond this is a data artefact, and forcing power to 0 below walking-pace
# speed keeps stoplights/traffic from reading as effort.
_MAX_GRADIENT_FRACTION = 0.25
_MIN_MOVING_SPEED_M_PER_S = 1.0
_MAX_POWER_W = 1500.0

# Cycling-tuned smoothing — gentler than running's GAP defaults (60 s / 500 m):
# a bike computer samples cleanly enough that a shorter window still kills GPS
# noise without eating real short climbs.
_ALTITUDE_FILTER = FilterConfig(rolling_window_s=30.0, savgol_window_m=200.0)
_SPEED_SMOOTHING_S = 4.0


def _air_density_ratio(altitude_m: np.ndarray) -> float:
    """ρ(altitude) ÷ 1.2 — the reference density the aero coefficient assumes.

    Uses the ride's mean altitude rather than a per-step value: altitude barely
    varies within one ride, and this correction is about *where* the athlete
    trains (a Boulder or Font-Romeu resident rides materially thinner air), not
    about elevation change during the ride itself.
    """
    mean_altitude = float(np.nanmean(altitude_m)) if altitude_m.size else 0.0
    if not np.isfinite(mean_altitude):
        mean_altitude = 0.0
    rho = 1.225 * (1.0 - 2.256e-5 * mean_altitude) ** 4.256
    return rho / 1.2


def compute_cycling_power_series(
    *,
    time: np.ndarray,
    distance: np.ndarray,
    altitude: np.ndarray,
    mass_kg: Optional[float],
) -> Optional[np.ndarray]:
    """Modelled cycling power per step, in watts (length ``len(time) - 1``).

    Returns ``None`` when ``mass_kg`` is missing — exactly like running's
    :func:`~src.domain.races.metrics.compute_power_series`, so a rider with no
    weight on file gets no modelled figure rather than a population-average
    guess. (A real power meter bypasses this function entirely, at the caller,
    so it is unaffected by this gate.)
    """
    if mass_kg is None:
        return None
    time = np.asarray(time, dtype=float)
    distance = np.asarray(distance, dtype=float)
    altitude = np.asarray(altitude, dtype=float)
    if time.size < 2:
        return np.zeros(0)

    delta_time = np.diff(time)
    delta_dist = np.diff(distance)

    altitude_smoothed = apply_signal_filters(
        altitude, timestamps_s=time, distance_m=distance,
        config=_ALTITUDE_FILTER, polyorder=2,
    )
    delta_alt = np.diff(altitude_smoothed)
    gradient_fraction = np.clip(
        np.divide(delta_alt, delta_dist, out=np.zeros_like(delta_dist), where=delta_dist > 0),
        -_MAX_GRADIENT_FRACTION, _MAX_GRADIENT_FRACTION,
    )

    speed = np.divide(
        delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0
    )
    speed_smoothed = rolling_mean_time(speed, time[1:], _SPEED_SMOOTHING_S)
    accel = np.divide(
        np.diff(speed_smoothed, prepend=speed_smoothed[0]),
        delta_time, out=np.zeros_like(delta_time), where=delta_time > 0,
    )

    air_ratio = _air_density_ratio(altitude_smoothed)
    aero_term = _AERO_COEFF * air_ratio * speed_smoothed ** 3

    total_mass = float(mass_kg) + _BIKE_KIT_MASS_KG
    mass_term = total_mass * speed_smoothed * (
        _GRAVITY_ROLLING_COEFF * gradient_fraction
        + _ROLLING_COEFF
        + _INERTIA_COEFF * accel
    )
    power = mass_term + aero_term

    power = np.where(speed_smoothed < _MIN_MOVING_SPEED_M_PER_S, 0.0, power)
    return np.clip(power, 0.0, _MAX_POWER_W)
