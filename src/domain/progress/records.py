"""Personal records via sliding-window best efforts.

A *best effort* for a target distance D is the fastest time the runner ever
covered D contiguous metres inside a single activity. For each start sample we
find the first later sample that is D metres further along the cumulative
distance stream (interpolating the crossing for sub-sample accuracy) and keep
the smallest elapsed time. Walking the per-activity best efforts in date order
then gives the *record progression*: one point each time the all-time best
improves.
"""

from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np


def best_effort_time(
    distance_m: Sequence[float],
    time_s: Sequence[float],
    target_m: float,
) -> Optional[float]:
    """Fastest elapsed time (s) to cover ``target_m`` contiguous metres.

    Returns ``None`` when the activity never covers ``target_m``. Elapsed time
    spans real (wall-clock) time, so a paused stretch inflates the segment and
    therefore can never be the fastest — pauses self-exclude.
    """
    distance = np.asarray(distance_m, dtype=float)
    time = np.asarray(time_s, dtype=float)
    n = distance.size
    if n < 2 or time.size != n or target_m <= 0:
        return None

    # Cumulative distance should be monotonic; GPS jitter can break that, which
    # would corrupt the searchsorted below — clamp it to non-decreasing.
    distance = np.maximum.accumulate(distance)
    if distance[-1] - distance[0] < target_m:
        return None

    # For every start i, the first index j whose distance reaches distance[i]+D.
    targets = distance + target_m
    j = np.searchsorted(distance, targets, side="left")
    valid = j < n
    if not valid.any():
        return None

    i_idx = np.nonzero(valid)[0]
    j_idx = j[valid]
    # targets[i] > distance[i] (D > 0) so j_idx ≥ 1 and j_idx-1 ≥ i_idx ≥ 0.
    d_cur = distance[j_idx]
    d_prev = distance[j_idx - 1]
    t_cur = time[j_idx]
    t_prev = time[j_idx - 1]

    # Interpolate the time at exactly distance[i]+D inside the (j-1, j] step.
    span = d_cur - d_prev
    frac = np.where(span > 0, (targets[i_idx] - d_prev) / span, 0.0)
    t_at = t_prev + frac * (t_cur - t_prev)

    elapsed = t_at - time[i_idx]
    elapsed = elapsed[np.isfinite(elapsed) & (elapsed > 0)]
    return float(elapsed.min()) if elapsed.size else None


def record_progression(
    samples: Sequence[Tuple[datetime, float]],
) -> List[Tuple[datetime, float]]:
    """Running-best step series from dated best efforts.

    ``samples`` is ``(date, time_s)`` pairs. Returns, in date order, one point
    each time the all-time best strictly improves — the data behind the stepped
    record-evolution line. The last point is the current record.
    """
    ordered = sorted(
        (d, t) for d, t in samples if d is not None and t is not None
    )
    progression: List[Tuple[datetime, float]] = []
    best: Optional[float] = None
    for date, value in ordered:
        if best is None or value < best:
            best = value
            progression.append((date, value))
    return progression
