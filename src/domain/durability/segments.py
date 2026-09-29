"""Historical evidence for durability: steady 5-minute segments of long runs.

This is the **calibration-data interface**: every fit — one athlete's
personalization at runtime, or the offline population calibration — consumes
:class:`DurabilitySegment` rows and nothing else.

**What a segment observes.** For a steady segment, HR reserve tracks metabolic
effort (%HRR ≈ %VO2 reserve), and effort is ``phi · (GAP speed)`` in the project's
cost model. So

    log(HR - HR_rest) = log(phi) + log(v_gap) + drift_HR(t) + c_activity

HR rises with elapsed time *at constant metabolic output* (cardiovascular drift),
and that must not be read as extra cost. Its population rate is subtracted before
HR is used as an effort signal:

    observed_log_cost = log((HR - HR_rest) / v_gap) - hr_drift_per_hour · t_h
                      = log(phi(t)) + c_activity + noise

The per-activity constant ``c`` (fitness that day, sensor offset, heat) is removed
by the fit, which only reads changes *within* an activity. Historical temperature
is not in any stream, so thermal HR drift cannot be corrected — it is part of the
noise the robust fit downweights, and the thermal coefficient is never
personalized.

**What is filtered out**, with a reason recorded per activity: treadmill runs (no
real gradient), poor elevation coverage, missing HR, pauses inside a segment,
walking-speed or near-rest-HR segments, unsteady segments, extreme gradients, and
intermittent sessions — the app has no structured-workout metadata, so intervals
can only be recognised by their speed variability, and are dropped.

Running power is deliberately not used, consistently with the feature pipeline:
a watch's running power is an undocumented per-vendor model, not a measurement.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.domain.dataset.sport import RUNNING_SPORT_TYPES
from src.domain.durability.capability import ReferenceSpeed
from src.domain.durability.config import (
    COMPONENTS,
    DOWNHILL,
    DURATION,
    SEVERE_INTENSITY,
    THERMAL,
    ExposureConfig,
    PersonalizationConfig,
)
from src.domain.durability.model import accumulate_exposures
from src.domain.models.activity import ActivityStream
from src.domain.races.metrics import gradient_adjustment_factor
from src.domain.races.smoothing import apply_signal_filters, default_smoothing_params

# Same pause rule as the feature pipeline.
PAUSE_THRESHOLD_S = 60.0
# A treadmill's altitude is not a gradient.
SEGMENT_SPORTS = RUNNING_SPORT_TYPES - {"VirtualRun"}
_CHUNK_S = 60.0

# Why an activity contributed nothing.
REASON_SPORT = "sport"
REASON_TOO_SHORT = "too_short"
REASON_NO_HR = "no_heart_rate"
REASON_ELEVATION = "poor_elevation"
REASON_FEW_SEGMENTS = "few_valid_segments"
REASON_INTERMITTENT = "intermittent"


@dataclass(frozen=True)
class DurabilitySegment:
    """One steady segment of one run, with the exposures accumulated before it."""

    activity_id: int
    start_date: Optional[datetime]
    elapsed_s: float              # moving time at the segment midpoint
    gap_speed_m_per_s: float
    heartrate_bpm: float
    intensity: float              # u = v_gap / CS
    exposures: Dict[str, float]   # cumulative, at the midpoint (same units as the model)
    observed_log_cost: float      # see module docstring
    athlete_key: str = ""         # grouping key for the offline calibration


def extract_segments(
    stream: ActivityStream,
    reference: ReferenceSpeed,
    exposure: ExposureConfig,
    config: PersonalizationConfig,
    athlete_key: str = "",
) -> Tuple[List[DurabilitySegment], Optional[str]]:
    """Valid segments of one activity, or ``([], reason)``."""
    if str(getattr(stream.sport_type, "root", stream.sport_type)) not in SEGMENT_SPORTS:
        return [], REASON_SPORT
    time = np.asarray(stream.time, dtype=float)
    distance = np.asarray(stream.distance, dtype=float)
    altitude = np.asarray(stream.altitude, dtype=float)
    heartrate = np.asarray(stream.heartrate, dtype=float)
    n = time.size
    if n < 3 or distance.size != n or altitude.size != n:
        return [], REASON_TOO_SHORT
    if heartrate.size != n or not np.isfinite(heartrate).any():
        return [], REASON_NO_HR
    if np.isfinite(altitude).mean() < config.min_altitude_coverage:
        return [], REASON_ELEVATION

    dt = np.diff(time)
    dd = np.diff(distance)
    moving = (dt > 0) & (dt <= PAUSE_THRESHOLD_S) & (dd >= 0)
    step_dt = np.where(moving, dt, 0.0)
    cum_t = np.concatenate([[0.0], np.cumsum(step_dt)])
    if cum_t[-1] < config.warmup_s + config.min_segments_per_activity * config.segment_s:
        return [], REASON_TOO_SHORT

    smoothed = apply_signal_filters(
        altitude, timestamps_s=time, distance_m=distance,
        config=default_smoothing_params().altitude,
    )
    dalt = np.nan_to_num(np.diff(smoothed), nan=0.0)
    grade = np.divide(dalt, dd, out=np.zeros_like(dd), where=dd > 0) * 1000.0
    gap_step = np.where(moving, dd * gradient_adjustment_factor(grade), 0.0)
    step_hr = heartrate[1:]

    cs = float(reference.speed_m_per_s)
    gap_speed = np.divide(gap_step, step_dt, out=np.zeros_like(step_dt), where=step_dt > 0)
    intensity = gap_speed / cs
    descent = np.where(moving & (dalt < 0), -dalt, 0.0)
    # No temperature in historical streams: the thermal exposure is zero.
    exposures = accumulate_exposures(cum_t, intensity, descent, np.zeros_like(dt), exposure)

    segments: List[DurabilitySegment] = []
    start = config.warmup_s
    while start + config.segment_s <= cum_t[-1]:
        end = start + config.segment_s
        i0 = int(np.searchsorted(cum_t, start, side="left"))
        i1 = int(np.searchsorted(cum_t, end, side="left"))
        segment = _segment(
            stream, i0, i1, time, cum_t, step_dt, dd, dalt, gap_step, step_hr,
            exposures, cs, config, athlete_key,
        )
        if segment is not None:
            segments.append(segment)
        start = end

    if len(segments) < config.min_segments_per_activity:
        return [], REASON_FEW_SEGMENTS
    speeds = np.array([s.gap_speed_m_per_s for s in segments])
    if np.std(speeds) / np.mean(speeds) > config.max_activity_speed_cv:
        return [], REASON_INTERMITTENT
    return segments, None


def _segment(stream, i0, i1, time, cum_t, step_dt, dd, dalt, gap_step, step_hr,
             exposures, cs, config: PersonalizationConfig, athlete_key: str
             ) -> Optional[DurabilitySegment]:
    if i1 - i0 < 2:
        return None
    moving_s = float(step_dt[i0:i1].sum())
    wall_s = float(time[i1] - time[i0])
    if wall_s <= 0 or moving_s / wall_s < config.min_moving_fraction:
        return None
    hr = step_hr[i0:i1]
    weights = step_dt[i0:i1]
    hr_ok = np.isfinite(hr) & (hr > 0)
    if weights[hr_ok].sum() < config.min_hr_coverage * moving_s:
        return None
    hr_mean = float(np.sum(hr[hr_ok] * weights[hr_ok]) / weights[hr_ok].sum())
    reserve = hr_mean - config.hr_rest_bpm
    if reserve < config.min_hr_reserve_bpm:
        return None

    v_gap = float(gap_step[i0:i1].sum() / moving_s)
    if v_gap < config.min_gap_speed:
        return None
    covered = float(dd[i0:i1].sum())
    if covered <= 0 or abs(dalt[i0:i1].sum() / covered * 1000.0) > config.max_abs_grade:
        return None
    if _speed_cv(cum_t[i0:i1 + 1], gap_step[i0:i1]) > config.max_segment_speed_cv:
        return None

    mid = (cum_t[i0] + cum_t[i1]) / 2
    at_mid = {name: float(np.interp(mid, cum_t, exposures[name])) for name in COMPONENTS}
    observed = float(np.log(reserve / v_gap) - config.hr_drift_per_hour * mid / 3600.0)
    return DurabilitySegment(
        activity_id=int(stream.activity_id),
        start_date=stream.start_date,
        elapsed_s=float(mid),
        gap_speed_m_per_s=v_gap,
        heartrate_bpm=hr_mean,
        intensity=v_gap / cs,
        exposures=at_mid,
        observed_log_cost=observed,
        athlete_key=athlete_key,
    )


def _speed_cv(cum_t: np.ndarray, gap_step: np.ndarray) -> float:
    """Coefficient of variation of GAP speed over ~60 s chunks (1 s GPS is too noisy)."""
    edges = np.arange(cum_t[0], cum_t[-1] + 1e-9, _CHUNK_S)
    if edges.size < 3:
        return 0.0
    chunk = np.clip(np.searchsorted(edges, cum_t[:-1], side="right") - 1, 0, edges.size - 2)
    dist = np.bincount(chunk, weights=gap_step, minlength=edges.size - 1)
    dur = np.bincount(chunk, weights=np.diff(cum_t), minlength=edges.size - 1)
    ok = dur > 0.5 * _CHUNK_S
    if ok.sum() < 2:
        return 0.0
    speeds = dist[ok] / dur[ok]
    return float(np.std(speeds) / np.mean(speeds)) if np.mean(speeds) > 0 else np.inf


def design_matrix(segments: List[DurabilitySegment], names=(DURATION, SEVERE_INTENSITY, DOWNHILL)
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y, X, activity_index)`` for a list of segments."""
    y = np.array([s.observed_log_cost for s in segments], dtype=float)
    X = np.array([[s.exposures[name] for name in names] for s in segments], dtype=float)
    _, groups = np.unique([s.activity_id for s in segments], return_inverse=True)
    return y, X.reshape(len(segments), len(names)), groups


IDENTIFIABLE = (DURATION, SEVERE_INTENSITY, DOWNHILL)
NOT_IDENTIFIABLE = (THERMAL,)
