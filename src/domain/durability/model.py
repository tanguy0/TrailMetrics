"""The durability cost multiplier ``phi(t)`` along a sequence of points.

    effective energetic cost after exposure = phi(t) · effective cost while fresh

``phi`` is an *equivalent metabolic cost multiplier*: it says how much more the
same GAP speed costs after the exposures accumulated so far. It is not a claim to
measure VO2 or running economy.

    logPhi(t) = Σ_c coefficient_c · exposure_c(t)  (+ optional pre-race term)
    phi(t)    = exp(logPhi(t)),   phi >= 1

Exposures accumulate interval by interval, so irregular spacing, pauses (``dt``
large, ``u`` ≈ 0) and very low speeds need no special case:

====================  =====================================  =======
exposure              increment over one interval            unit
====================  =====================================  =======
duration              ``dt_h · u^p``                         h
severe_intensity      ``dt_h · max(0, u - 1)^q``             h
downhill              ``descent_km · u^r``                   km
thermal               ``dt_h · heat_stress``                 h
====================  =====================================  =======

``u`` is relative intensity (required power / the athlete's critical power —
see :mod:`src.domain.durability.solver` for how the planner derives it), ``dt_h`` the
interval's duration in hours and ``descent_km`` its elevation *loss* in km. The
downhill term is delayed eccentric muscle damage accumulating over the race — not
the instantaneous cost of descending, which the GAP curve already prices.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.domain.durability.config import (
    COMPONENTS,
    DOWNHILL,
    DURATION,
    PRE_RACE_LOAD,
    SEVERE_INTENSITY,
    THERMAL,
    ExposureConfig,
    DurabilityCoefficients,
)


# --- Weather ------------------------------------------------------------------

@dataclass(frozen=True)
class RaceWeather:
    """Race-day conditions: temperature interpolated over the race, humidity constant.

    ``relative_humidity_start_pct`` / ``relative_humidity_end_pct`` are accepted so
    a later version can interpolate humidity too without changing this API; when
    both are set they take precedence over the constant.
    """

    temperature_start_c: Optional[float] = None
    temperature_end_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    relative_humidity_start_pct: Optional[float] = None
    relative_humidity_end_pct: Optional[float] = None

    @property
    def known(self) -> bool:
        return self.temperature_start_c is not None or self.temperature_end_c is not None

    def temperature_at(self, fraction: np.ndarray) -> np.ndarray:
        """°C at each race fraction in ``[0, 1]``; NaN when no temperature is known."""
        start, end = self.temperature_start_c, self.temperature_end_c
        if start is None and end is None:
            return np.full(np.shape(fraction), np.nan)
        start = end if start is None else start
        end = start if end is None else end
        return _lerp(float(start), float(end), fraction)

    def humidity_at(self, fraction: np.ndarray, default_pct: float) -> np.ndarray:
        start, end = self.relative_humidity_start_pct, self.relative_humidity_end_pct
        if start is not None and end is not None:
            values = _lerp(float(start), float(end), fraction)
        else:
            constant = self.relative_humidity_pct
            values = np.full(np.shape(fraction), default_pct if constant is None else constant)
        return np.clip(values, 0.0, 100.0)


def _lerp(start: float, end: float, fraction: np.ndarray) -> np.ndarray:
    f = np.clip(np.asarray(fraction, dtype=float), 0.0, 1.0)
    return start + (end - start) * f


def apparent_temperature_c(temperature_c: np.ndarray, relative_humidity_pct: np.ndarray
                           ) -> np.ndarray:
    """Humidity-adjusted apparent temperature (Steadman, shade, no wind), °C.

        AT = T + 0.33·e − 4.0,   e = RH/100 · 6.105 · exp(17.27·T / (237.7 + T))  [hPa]

    Wind and solar radiation are unknown, so this is **not WBGT** — only a proxy that
    rises with both temperature and humidity.
    """
    t = np.asarray(temperature_c, dtype=float)
    rh = np.asarray(relative_humidity_pct, dtype=float)
    vapour_hpa = rh / 100.0 * 6.105 * np.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * vapour_hpa - 4.0


def heat_stress(temperature_c: np.ndarray, relative_humidity_pct: np.ndarray,
                config: ExposureConfig) -> np.ndarray:
    """Normalized heat stress: 0 in neutral/cool conditions, ``>= 0`` above them.

        heat_stress = clip((AT − thermal_reference_c) / thermal_scale_c, 0, thermal_cap)

    Unknown temperature (NaN) counts as neutral.
    """
    apparent = apparent_temperature_c(temperature_c, relative_humidity_pct)
    excess = (apparent - config.thermal_reference_c) / config.thermal_scale_c
    return np.nan_to_num(np.clip(excess, 0.0, config.thermal_cap), nan=0.0)


# --- Exposures and phi ----------------------------------------------------------

@dataclass
class DurabilityProfile:
    """``phi`` and its diagnostics at every point (``len = intervals + 1``)."""

    multiplier: np.ndarray                    # phi, >= 1
    log_multiplier: np.ndarray
    components: Dict[str, np.ndarray]         # log-cost contribution per exposure
    exposures: Dict[str, np.ndarray]          # cumulative exposure, in its unit
    clamped_points: int = 0
    pre_race_exposure: Optional[float] = None
    coefficients: Optional[DurabilityCoefficients] = None
    extra: dict = field(default_factory=dict)

    @property
    def clamped(self) -> bool:
        return self.clamped_points > 0

    def interval_multiplier(self) -> np.ndarray:
        """phi over each interval: the mean of its two end points."""
        return (self.multiplier[:-1] + self.multiplier[1:]) / 2

    def at(self, index: int) -> Dict[str, object]:
        """One point's output, in the shape the API documents."""
        components = {name: float(values[index]) for name, values in self.components.items()}
        return {
            "durability_cost_multiplier": float(self.multiplier[index]),
            "log_durability_cost": float(self.log_multiplier[index]),
            "components": components,
        }


def accumulate_exposures(
    elapsed_s: np.ndarray,
    intensity: np.ndarray,
    descent_m: np.ndarray,
    heat: np.ndarray,
    config: ExposureConfig,
) -> Dict[str, np.ndarray]:
    """Cumulative exposures at every point, from per-interval inputs.

    ``elapsed_s`` has one entry per point; ``intensity``, ``descent_m`` and ``heat``
    one per interval. Non-finite inputs contribute nothing, negative ``dt`` is
    treated as zero, and ``u`` is clipped to ``[0, max_intensity]``.
    """
    dt_h = np.clip(np.nan_to_num(np.diff(np.asarray(elapsed_s, dtype=float)), nan=0.0),
                   0.0, None) / 3600.0
    u = np.clip(np.nan_to_num(np.asarray(intensity, dtype=float), nan=0.0),
                0.0, config.max_intensity)
    descent_km = np.clip(np.nan_to_num(np.asarray(descent_m, dtype=float), nan=0.0),
                         0.0, None) / 1000.0
    h = np.clip(np.nan_to_num(np.asarray(heat, dtype=float), nan=0.0), 0.0, None)

    increments = {
        DURATION: dt_h * u ** config.duration_exponent,
        SEVERE_INTENSITY: dt_h * np.maximum(0.0, u - 1.0) ** config.severe_exponent,
        DOWNHILL: descent_km * u ** config.downhill_exponent,
        THERMAL: dt_h * h,
    }
    return {name: np.concatenate([[0.0], np.cumsum(inc)]) for name, inc in increments.items()}


def durability_profile(
    exposures: Dict[str, np.ndarray],
    coefficients: DurabilityCoefficients,
    config: ExposureConfig,
    pre_race_exposure: Optional[float] = None,
) -> DurabilityProfile:
    """``phi`` from cumulative exposures. ``phi(0) = 1`` unless the pre-race load term is set."""
    n = len(next(iter(exposures.values())))
    components = {name: coefficients.get(name) * exposures[name] for name in COMPONENTS}
    if pre_race_exposure is not None and np.isfinite(pre_race_exposure):
        components[PRE_RACE_LOAD] = np.full(
            n, coefficients.pre_race_load * max(0.0, float(pre_race_exposure))
        )
    raw = np.sum(list(components.values()), axis=0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=np.inf)
    ceiling = float(np.log(config.max_multiplier))
    log_phi = np.clip(raw, 0.0, ceiling)
    return DurabilityProfile(
        multiplier=np.exp(log_phi),
        log_multiplier=log_phi,
        components=components,
        exposures=exposures,
        clamped_points=int(np.sum(raw > ceiling)),
        pre_race_exposure=pre_race_exposure,
        coefficients=coefficients,
    )
