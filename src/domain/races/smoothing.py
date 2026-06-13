"""Configurable smoothing for the race-comparator signal pipeline.

Every signal (pace, altitude, heart rate, power) can be passed through two
independent, optional filters, in this order:

1. **Rolling mean** — a trailing, time-domain moving average (window in seconds).
   See :func:`rolling_mean_time`.
2. **Savitzky–Golay** — a distance-domain polynomial smoother (window in metres).
   See :func:`savgol_distance`.

A :class:`FilterConfig` says which filters are on (and their windows) for one
signal; :class:`SmoothingParams` bundles one config per signal. The windows are
left entirely to the caller (the page exposes them as inputs).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# --- Configuration ---------------------------------------------------------

@dataclass
class FilterConfig:
    """Smoothing for one signal. ``None`` (or ≤ 0) disables a filter.

    Filters are applied rolling-mean first, then Savitzky–Golay.
    """
    rolling_window_s: Optional[float] = None
    savgol_window_m: Optional[float] = None


@dataclass
class SmoothingParams:
    """One :class:`FilterConfig` per signal, plus the shared SavGol polyorder."""
    pace: FilterConfig = field(default_factory=FilterConfig)
    altitude: FilterConfig = field(default_factory=FilterConfig)
    heartrate: FilterConfig = field(default_factory=FilterConfig)
    power: FilterConfig = field(default_factory=FilterConfig)
    savgol_polyorder: int = 2


def default_smoothing_params() -> SmoothingParams:
    """Sensible defaults (also the page's initial control values)."""
    return SmoothingParams(
        pace=FilterConfig(rolling_window_s=15.0),
        altitude=FilterConfig(rolling_window_s=60.0, savgol_window_m=500.0),
        heartrate=FilterConfig(rolling_window_s=60.0),
        power=FilterConfig(rolling_window_s=15.0),
        savgol_polyorder=2,
    )


# --- Primitives ------------------------------------------------------------

def rolling_mean_time(
    values: np.ndarray, timestamps_s: np.ndarray, window_s: float
) -> np.ndarray:
    """Trailing (causal) time-windowed rolling mean; ``min_periods=1``.

    Each output sample is the mean of all samples within the preceding
    ``window_s`` seconds (inclusive), keyed on real timestamps so it adapts to
    the sampling rate and never bridges a pause. NaNs are skipped within a
    window; an all-NaN window yields NaN.
    """
    values = np.asarray(values, dtype=float)
    index = pd.to_timedelta(np.asarray(timestamps_s, dtype=float), unit="s")
    series = pd.Series(values, index=index)
    smoothed = series.rolling(pd.Timedelta(seconds=window_s), min_periods=1).mean()
    return smoothed.to_numpy()


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _distance_window_samples(
    distance_m: np.ndarray, window_m: float, min_samples: int
) -> int:
    """Convert a distance window (m) to an odd sample count via median spacing."""
    steps = np.diff(np.asarray(distance_m, dtype=float))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    median_step = float(np.median(steps)) if steps.size else 1.0
    if not np.isfinite(median_step) or median_step <= 0:
        median_step = 1.0
    return max(_odd(window_m / median_step), _odd(min_samples))


def savgol_distance(
    values: np.ndarray, distance_m: np.ndarray, window_m: float, polyorder: int = 2
) -> np.ndarray:
    """Distance-domain Savitzky–Golay smoothing (window in metres).

    The window is converted to an odd sample count from the median inter-sample
    spacing, clamped to ``[polyorder + 2, length]``. NaNs are linearly
    interpolated before filtering and restored afterward. Returns the input
    unchanged when there aren't enough samples to fit.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < polyorder + 2:
        return values

    largest_odd = n if n % 2 == 1 else n - 1
    w = min(_distance_window_samples(distance_m, window_m, polyorder + 2), largest_odd)
    if w <= polyorder:
        return values

    nan_mask = ~np.isfinite(values)
    filled = values.copy()
    if nan_mask.any():
        good = ~nan_mask
        if good.sum() < 2:
            return values
        idx = np.arange(n)
        filled[nan_mask] = np.interp(idx[nan_mask], idx[good], values[good])

    smoothed = savgol_filter(filled, window_length=w, polyorder=polyorder, mode="interp")
    smoothed[nan_mask] = np.nan
    return smoothed


def apply_signal_filters(
    values: np.ndarray,
    *,
    timestamps_s: np.ndarray,
    distance_m: np.ndarray,
    config: FilterConfig,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply a signal's configured filters: rolling mean, then Savitzky–Golay."""
    out = values
    if config.rolling_window_s and config.rolling_window_s > 0:
        out = rolling_mean_time(out, timestamps_s, config.rolling_window_s)
    if config.savgol_window_m and config.savgol_window_m > 0:
        out = savgol_distance(out, distance_m, config.savgol_window_m, polyorder)
    return out
