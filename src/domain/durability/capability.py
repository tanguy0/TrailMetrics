"""The athlete's reference speed: the denominator of relative intensity ``u``.

The project's running cost model is ``P = m·v·Cr·a(g)`` (see
:func:`src.domain.races.metrics.compute_power_series`), so at equal mass and
``Cr`` a power ratio is a GAP-speed ratio:

    u = P_required / P_critical = (v · a(g)) / CS

No critical speed, VDOT or threshold is stored anywhere in the app, and none is
invented here. Two sources, in order:

1. **The athlete's own best efforts of the past year.** Every activity's best time
   over each PR distance is a candidate, preferably its *gradient-adjusted* best
   (``best_gap_*``: the fastest time to cover D metres of GAP distance), so a
   trail runner's hilly 10 km counts as the flat-equivalent effort it was rather
   than as a slow flat one. Rows featurized before those columns existed fall
   back to the raw best (``best_*``), which can only *under*estimate on hills —
   and the target floor below catches that. Each candidate of distance ``d`` in
   time ``t`` implies a critical speed ``(d / t) / fraction(t)``; outliers are
   removed (see :func:`_consensus`) and the highest surviving implication wins,
   since the most maximal effort is the most informative.
2. **The race target itself**, for a visitor or an athlete without usable
   efforts: a target is by definition an effort the runner expects to sustain for
   its whole duration, so it implies ``u = fraction(T)``. This is a documented,
   low-confidence population fallback, and it is also used as a *floor* for (1):
   stale or sub-maximal training efforts would otherwise make an achievable target
   look supra-critical.

    fraction(t) = (t / cs_reference_duration_s) ^ (1 - riegel_exponent)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.dataset.features import best_column, gap_best_column
from src.domain.durability.config import CapabilityConfig
from src.domain.progress.models import PR_DISTANCES

BEST_EFFORTS = "best_efforts"
TARGET_TIME = "target_time"
NONE = "none"


@dataclass(frozen=True)
class ReferenceSpeed:
    """A critical-speed estimate (GAP m/s) and where it came from."""

    speed_m_per_s: Optional[float]
    source: str = NONE
    detail: dict = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.speed_m_per_s is not None and np.isfinite(self.speed_m_per_s) \
            and self.speed_m_per_s > 0


def sustainable_fraction(duration_s: float, config: CapabilityConfig) -> float:
    """Share of critical speed sustainable for ``duration_s`` (Riegel)."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        return 1.0
    fraction = (duration_s / config.cs_reference_duration_s) ** (1.0 - config.riegel_exponent)
    return float(np.clip(fraction, config.min_fraction, config.max_fraction))


@dataclass(frozen=True)
class _Candidate:
    implied: float     # critical-speed implication, m/s
    label: str         # PR distance label
    time_s: float
    activity_id: int
    gap_adjusted: bool


def reference_from_best_efforts(features: pd.DataFrame, config: CapabilityConfig
                                ) -> ReferenceSpeed:
    """Critical speed implied by the best efforts in ``features`` (already date-filtered)."""
    candidates = _candidates(features, config)
    if not candidates:
        return ReferenceSpeed(None, NONE, {"reason": "no_efforts"})
    kept, rejected = _consensus(candidates, config)
    if not kept:
        return ReferenceSpeed(None, NONE, {"reason": "no_efforts"})
    best = max(kept, key=lambda c: c.implied)
    return ReferenceSpeed(best.implied, BEST_EFFORTS, {
        "distance": best.label,
        "time_s": best.time_s,
        "activity_id": best.activity_id,
        "gap_adjusted": best.gap_adjusted,
        "outliers": [
            {"distance": c.label, "activity_id": c.activity_id, "implied": c.implied}
            for c in rejected
        ],
    })


def _candidates(features: pd.DataFrame, config: CapabilityConfig) -> List[_Candidate]:
    """Every plausible (activity, distance) best effort, GAP-adjusted where available."""
    if features is None or features.empty:
        return []
    ids = (features["activity_id"].to_numpy() if "activity_id" in features.columns
           else np.arange(len(features)))
    out: List[_Candidate] = []
    for label, metres in PR_DISTANCES:
        raw = _column(features, best_column(label))
        gap = _column(features, gap_best_column(label))
        use_gap = np.isfinite(gap) & (gap > 0)
        times = np.where(use_gap, gap, raw)
        for t, adjusted, activity_id in zip(times, use_gap, ids):
            if not (np.isfinite(t) and config.min_effort_s <= t <= config.max_effort_s):
                continue
            speed = metres / t
            if speed > config.max_plausible_speed:
                continue
            out.append(_Candidate(speed / sustainable_fraction(t, config), label,
                                  float(t), int(activity_id), bool(adjusted)))
    return out


def _column(features: pd.DataFrame, name: str) -> np.ndarray:
    if name not in features.columns:
        return np.full(len(features), np.nan)
    return pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)


def _consensus(candidates: List[_Candidate], config: CapabilityConfig
               ) -> Tuple[List[_Candidate], List[_Candidate]]:
    """Drop season bests that the other distances do not corroborate.

    For each distance, its best candidate is compared with the median of the
    *other* distances' current bests (leave-one-out, so the suspect never votes
    for itself). Above ``outlier_log_tolerance`` it is rejected and that distance's
    next-best candidate — usually another activity — takes its place, until the
    best passes or the distance runs out. Repeated until nothing changes, since
    removing one outlier can lower the median another was judged against.

    A tunnel on one run makes that run's 1 km implausibly fast next to every 3, 5
    and 10 km the athlete has run; the check drops it and keeps their true best
    1 km from another run. With fewer than ``min_distances_for_consensus``
    distances there is nothing to compare against, and candidates are kept as is.
    """
    by_distance: Dict[str, List[_Candidate]] = {}
    for c in sorted(candidates, key=lambda c: -c.implied):
        by_distance.setdefault(c.label, []).append(c)
    rejected: List[_Candidate] = []
    if len(by_distance) < config.min_distances_for_consensus:
        return candidates, rejected

    changed = True
    while changed:
        changed = False
        for label in list(by_distance):
            others = [np.log(v[0].implied) for k, v in by_distance.items() if k != label and v]
            if len(others) < config.min_distances_for_consensus - 1:
                continue
            reference = float(np.median(others))
            queue = by_distance[label]
            while queue and np.log(queue[0].implied) - reference > config.outlier_log_tolerance:
                rejected.append(queue.pop(0))
                changed = True
            if not queue:
                del by_distance[label]
    kept = [c for queue in by_distance.values() for c in queue]
    return kept, rejected


def race_reference(
    athlete: Optional[ReferenceSpeed],
    fresh_gap_pace_s_per_km: float,
    target_time_s: float,
    config: CapabilityConfig,
) -> ReferenceSpeed:
    """The reference speed a race plan uses: the athlete's, floored by the target's."""
    implied = (1000.0 / fresh_gap_pace_s_per_km) / sustainable_fraction(target_time_s, config)
    if athlete is not None and athlete.known and athlete.speed_m_per_s >= implied:
        return athlete
    detail = {"target_implied": implied}
    if athlete is not None and athlete.known:
        # The athlete's efforts say this target is beyond them; trust the target.
        detail["athlete_speed"] = athlete.speed_m_per_s
        detail["reason"] = "athlete_reference_below_target"
    return ReferenceSpeed(implied, TARGET_TIME, detail)
