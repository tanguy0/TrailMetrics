"""The activity feature table — one tidy row per activity.

This is the substrate for every activity-level plot: a flat, exportable
``pandas`` frame whose columns are the quantities we know about a run. Trends,
records, distributions, scatter plots and comparison tables are all views of it,
which is why adding an analysable quantity means adding a column here plus one
entry in :mod:`src.domain.dataset.metrics` — and it immediately becomes plottable
everywhere.

The per-second pass that produces a row is the expensive part (altitude
smoothing, sliding-window best efforts, per-step power), so rows are memoized per
activity in a :class:`FeatureStore` whose cache is injected by the caller. The app
keeps that cache in session state; a deployed version can back it with a table and
compute each activity exactly once, ever.

Columns are deliberately *raw sums*, not averages: ``distance_m`` and ``moving_s``
rather than a pre-computed pace. Averages are derived by
:mod:`src.domain.dataset.metrics` as ratios, so they re-aggregate correctly over
any bin (Σ distance ÷ Σ time), which a mean of per-activity paces does not.

Power is stored **per kilogram**. The model is ``P = m·v·(Cr + g·s)``, exactly
linear in body mass, so a row computed once is valid for any weight:
:func:`apply_mass` multiplies through at read time. That keeps a stored row
weight-independent — changing your weight rescales every power figure instantly
instead of invalidating a decade of cached activities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.domain.dataset.binning import naive
from src.domain.models.activity import ActivityStream
from src.domain.progress.models import GRADIENT_BANDS, PR_DISTANCES
from src.domain.progress.records import best_effort_time
from src.domain.races.metrics import compute_power_series, gradient_adjustment_factor
from src.domain.races.smoothing import (
    FilterConfig,
    apply_signal_filters,
    default_smoothing_params,
)

# A time jump larger than this between samples means the watch was paused; the
# bridging step adds no real distance, time or climb, so it is excluded.
PAUSE_THRESHOLD_S = 60.0

# Column-name prefixes for the families of generated columns.
BAND_PREFIX = "time_"        # time_flat, time_steep_ascent, …  (seconds)
BEST_PREFIX = "best_"        # best_10km, best_marathon, …      (seconds)


def band_column(band_key: str) -> str:
    return f"{BAND_PREFIX}{band_key}"


def best_column(pr_label: str) -> str:
    """Column name for a PR distance's best effort (``"10 km"`` → ``best_10_km``)."""
    slug = pr_label.lower().replace(" ", "_")
    return f"{BEST_PREFIX}{slug}"


# Columns computed and stored per activity, independent of body weight.
STORED_COLUMNS: List[str] = (
    [
        "activity_id", "date", "sport_type", "has_streams",
        "distance_m", "elevation_gain_m", "moving_s", "elapsed_s",
        "gap_distance_m", "avg_hr", "max_hr",
        "avg_power_w_per_kg", "power_to_hr_per_kg",
        # Strava's Relative Effort. The one column that is *reported* rather than
        # computed here — it comes off the activity summary, because Strava derives
        # it from the athlete's own heart-rate zones, which its API does not expose.
        "relative_effort",
    ]
    + [band_column(key) for key, _, _ in GRADIENT_BANDS]
    + [best_column(label) for label, _ in PR_DISTANCES]
)

# Columns derived from the stored ones at read time (see :func:`apply_mass`).
MASS_SCALED_COLUMNS: Dict[str, str] = {
    "avg_power_w": "avg_power_w_per_kg",
    "power_to_hr": "power_to_hr_per_kg",
}

# What a plot sees.
FEATURE_COLUMNS: List[str] = STORED_COLUMNS + list(MASS_SCALED_COLUMNS)

# The generated column families, exposed so the storage layer can round-trip them
# without hard-coding the current list of bands and PR distances.
GENERATED_COLUMNS: List[str] = (
    [band_column(key) for key, _, _ in GRADIENT_BANDS]
    + [best_column(label) for label, _ in PR_DISTANCES]
)


class FeatureStore:
    """Memoized per-activity feature rows.

    ``cache`` is injected so it can outlive one render, or be replaced by a
    persistent store — the interface is just ``dict``-like access keyed by
    activity id. Rows are weight-independent, so nothing else enters the key.
    """

    def __init__(
        self,
        cache: Optional[Dict[Any, Dict[str, Any]]] = None,
        altitude_filter: Optional[FilterConfig] = None,
        polyorder: int = 2,
    ):
        self.cache = cache if cache is not None else {}
        self.altitude_filter = altitude_filter or default_smoothing_params().altitude
        self.polyorder = polyorder

    def row(self, stream: ActivityStream) -> Optional[Dict[str, Any]]:
        """Feature row for one activity, computing it at most once per cache."""
        key = int(stream.activity_id)
        if key not in self.cache:
            self.cache[key] = build_activity_features(
                stream,
                altitude_filter=self.altitude_filter,
                polyorder=self.polyorder,
            )
        return self.cache[key]

    def table(self, streams: Sequence[ActivityStream]) -> pd.DataFrame:
        """Feature table for ``streams``, undated/unusable activities dropped."""
        rows = [r for s in streams if (r := self.row(s)) is not None]
        return frame_from_rows(rows)


def frame_from_rows(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Build a feature frame from raw rows, with a stable column set."""
    return _frame(list(rows))


def apply_mass(frame: pd.DataFrame, mass_kg: Optional[float]) -> pd.DataFrame:
    """Materialize the weight-dependent power columns.

    Without a weight, power is unmodellable and the columns stay ``NaN`` — plots
    that need them say so rather than inventing a default body mass.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for target, source in MASS_SCALED_COLUMNS.items():
        if mass_kg and source in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce") * float(mass_kg)
        else:
            out[target] = np.nan
    return out


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the frame with a stable column set, even when empty."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FEATURE_COLUMNS})
    df = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[FEATURE_COLUMNS].sort_values("date").reset_index(drop=True)


def build_activity_features(
    stream: ActivityStream,
    *,
    altitude_filter: Optional[FilterConfig] = None,
    polyorder: int = 2,
) -> Optional[Dict[str, Any]]:
    """One per-second pass over an activity, producing its whole feature row.

    Returns ``None`` for activities that can't be placed on a timeline (no date)
    or carry no usable totals at all. Activities without per-second streams
    (manual entries) still get a row from Strava's summary — distance, elevation
    and time are known, the stream-derived columns stay ``NaN`` — so they show up
    in volume trends instead of silently vanishing.
    """
    if not isinstance(stream.start_date, datetime):
        return None

    altitude_filter = altitude_filter or default_smoothing_params().altitude

    time = np.asarray(stream.time, dtype=float)
    distance = np.asarray(stream.distance, dtype=float)
    altitude = np.asarray(stream.altitude, dtype=float)
    heartrate = np.asarray(stream.heartrate, dtype=float)
    n = time.size

    if n < 2 or distance.size != n or altitude.size != n:
        return _summary_only_row(stream)

    row: Dict[str, Any] = {
        "activity_id": int(stream.activity_id),
        "date": naive(stream.start_date),
        "sport_type": sport_name(stream.sport_type),
        "has_streams": True,
        "elapsed_s": float(time[-1] - time[0]),
        "relative_effort": _optional_float(stream.summary_relative_effort),
    }

    # Gradient, elevation gain and GAP all derive from a *smoothed* altitude
    # trace; raw per-second GPS altitude is far too noisy to differentiate.
    altitude_smoothed = apply_signal_filters(
        altitude, timestamps_s=time, distance_m=distance,
        config=altitude_filter, polyorder=polyorder,
    )

    delta_time = np.diff(time)
    delta_dist = np.diff(distance)
    delta_alt = np.diff(altitude_smoothed)
    moving = (delta_time > 0) & (delta_time <= PAUSE_THRESHOLD_S) & (delta_dist > 0)

    row["distance_m"] = float(np.sum(delta_dist[moving]))
    row["moving_s"] = float(np.sum(delta_time[moving]))
    row["elevation_gain_m"] = float(np.sum(delta_alt[moving & (delta_alt > 0)]))

    gradient_pct = np.divide(
        delta_alt, delta_dist, out=np.zeros_like(delta_dist), where=delta_dist > 0
    ) * 100.0
    gradient_m_per_km = gradient_pct * 10.0

    # GAP-adjusted distance: Σ (step distance × the reference speed adjuster).
    # Stored as a distance so avg GAP pace re-aggregates as Σtime ÷ Σgap-distance.
    factor = gradient_adjustment_factor(gradient_m_per_km)
    row["gap_distance_m"] = float(np.nansum((delta_dist * factor)[moving]))

    # Time per gradient band — the raw material of the gradient map, and of
    # "how much of my season was steep climbing" style questions.
    for key, lower, upper in GRADIENT_BANDS:
        in_band = moving & (gradient_pct >= lower) & (gradient_pct < upper)
        row[band_column(key)] = float(np.sum(delta_time[in_band]))

    step_hr = heartrate[1:] if heartrate.size == n else np.full(n - 1, np.nan)
    hr_valid = moving & np.isfinite(step_hr) & (step_hr > 0)
    row["avg_hr"] = float(np.mean(step_hr[hr_valid])) if hr_valid.any() else np.nan
    row["max_hr"] = float(np.max(step_hr[hr_valid])) if hr_valid.any() else np.nan

    speed = np.divide(
        delta_dist, delta_time, out=np.zeros_like(delta_dist), where=delta_time > 0
    )
    # Computed at 1 kg: the power model is linear in mass, so the row stays valid
    # for any body weight and :func:`apply_mass` scales it on read.
    power = compute_power_series(
        speed_m_per_s=speed, gradient_m_per_km=gradient_m_per_km, mass_kg=1.0,
    )
    row["avg_power_w_per_kg"] = np.nan
    row["power_to_hr_per_kg"] = np.nan
    if power is not None:
        power_valid = moving & np.isfinite(power)
        if power_valid.any():
            row["avg_power_w_per_kg"] = float(np.mean(power[power_valid]))
        both = power_valid & hr_valid
        if both.any():
            mean_hr = float(np.mean(step_hr[both]))
            if mean_hr > 0:
                row["power_to_hr_per_kg"] = float(np.mean(power[both])) / mean_hr

    # Best efforts run on the *raw* cumulative streams: elapsed time spans real
    # wall-clock, so a paused stretch inflates a segment and self-excludes.
    for label, meters in PR_DISTANCES:
        best = best_effort_time(distance, time, meters)
        row[best_column(label)] = float(best) if best is not None else np.nan

    return row


def _summary_only_row(stream: ActivityStream) -> Optional[Dict[str, Any]]:
    """Row built from Strava's activity summary, for streamless activities."""
    if stream.summary_distance_m is None:
        return None
    row: Dict[str, Any] = {
        "activity_id": int(stream.activity_id),
        "date": naive(stream.start_date),
        "sport_type": sport_name(stream.sport_type),
        "has_streams": False,
        "distance_m": float(stream.summary_distance_m or 0.0),
        "elevation_gain_m": float(stream.summary_elevation_gain_m or 0.0),
        "moving_s": float(stream.summary_moving_time_s or 0.0),
        "elapsed_s": float(stream.summary_moving_time_s or 0.0),
        "gap_distance_m": np.nan,
        "avg_hr": np.nan,
        "max_hr": np.nan,
        "avg_power_w_per_kg": np.nan,
        "power_to_hr_per_kg": np.nan,
        # Reported by Strava, so it survives even with no per-second data.
        "relative_effort": _optional_float(stream.summary_relative_effort),
    }
    for key, _, _ in GRADIENT_BANDS:
        row[band_column(key)] = np.nan
    for label, _ in PR_DISTANCES:
        row[best_column(label)] = np.nan
    return row


def _optional_float(value: Any) -> float:
    """A float, or ``NaN`` for anything unusable.

    NaN rather than ``None`` because the feature frame is numeric: a ``None`` in a
    float column makes pandas fall back to ``object`` dtype and every downstream
    aggregation on it silently changes behaviour.
    """
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def sport_name(sport_type) -> str:
    """Coerce stravalib's ``RelaxedSportType`` (carries ``.root``) to a string."""
    return str(getattr(sport_type, "root", sport_type))
