"""Three ways in, one estimate out (design/specs/level.md § Les trois entrées).

Every test goes through the VDOT pivot so the three agree with each other and
with the race paces elsewhere in the app; what is specific to a test (the
field-test VMA of a half-Cooper, the critical speed and D' of a two-distance
test, the spread of a set of records) rides along as notes.

Notes are translation keys with format parameters, never prose: the domain does
not know the reader's language.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain.level import vdot as model
from src.domain.level.zones import PaceZone, pace_zones

HALF_COOPER = "half_cooper"
CRITICAL_SPEED = "critical_speed"
RECORDS = "records"
METHODS = (HALF_COOPER, CRITICAL_SPEED, RECORDS)

# Plausibility bounds: outside them the input is a typo, not a runner.
_HALF_COOPER_M = (600.0, 2600.0)       # 6 km/h .. 26 km/h over 6 minutes
_D_PRIME_M = (50.0, 500.0)
_CS_M_PER_S = (1.5, 7.0)
_RECORD_PACE_S_PER_KM = (120.0, 720.0)  # 2:00 .. 12:00 /km

# Records within this many VDOT points of the median read as consistent.
_CONSISTENT_SPREAD = 1.5
_INCONSISTENT_SPREAD = 3.0


class LevelInputError(ValueError):
    """Input the model cannot use; ``key`` names the message to show."""

    def __init__(self, key: str, **params):
        super().__init__(key)
        self.key = key
        self.params = params


@dataclass
class Note:
    key: str
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class LevelEstimate:
    method: str
    vma_kmh: float
    vma_pace_s_per_km: float
    vdot: float
    confidence: str  # "high" | "medium" | "low"
    notes: List[Note] = field(default_factory=list)
    zones: List[PaceZone] = field(default_factory=list)
    # Test-specific figures shown beside the VMA (critical speed, D', field VMA).
    extras: Dict[str, float] = field(default_factory=dict)


def _estimate(method: str, vdot_value: float, confidence: str,
              notes: List[Note], extras: Dict[str, float]) -> LevelEstimate:
    vma = model.vma_kmh(vdot_value)
    pace = model.pace_s_per_km(vma)
    return LevelEstimate(
        method=method,
        vma_kmh=round(vma, 2),
        vma_pace_s_per_km=round(pace, 1),
        vdot=round(vdot_value, 1),
        confidence=confidence,
        notes=notes,
        zones=pace_zones(pace),
        extras=extras,
    )


def from_half_cooper(distance_m: float) -> LevelEstimate:
    """Six minutes all out. The VDOT route is the result; the field rule
    (VMA = distance / 100 km/h) is given as a note — they differ by a few %."""
    low, high = _HALF_COOPER_M
    if not low <= distance_m <= high:
        raise LevelInputError("level.error.half_cooper_range", low=int(low), high=int(high))
    field_vma = distance_m / 100.0
    estimate = _estimate(
        HALF_COOPER, model.vdot(distance_m, 6.0), "medium",
        [Note("level.note.field_vma", {"vma": round(field_vma, 1)})],
        {"field_vma_kmh": round(field_vma, 2)},
    )
    return estimate


def from_critical_speed(d3_m: float, d12_m: float) -> LevelEstimate:
    """3 min and 12 min all out. The 12-minute effort goes through VDOT; the
    critical speed and D' are this test's own figures, shown beside it."""
    if d3_m <= 0 or d12_m <= d3_m:
        raise LevelInputError("level.error.cs_inconsistent")
    cs = (d12_m - d3_m) / (720.0 - 180.0)
    d_prime = d3_m - cs * 180.0
    if not (_CS_M_PER_S[0] <= cs <= _CS_M_PER_S[1] and _D_PRIME_M[0] <= d_prime <= _D_PRIME_M[1]):
        raise LevelInputError("level.error.cs_inconsistent")
    estimate = _estimate(
        CRITICAL_SPEED, model.vdot(d12_m, 12.0),
        "high" if 100.0 <= d_prime <= 350.0 else "medium",
        [],
        {"critical_pace_s_per_km": round(1000.0 / cs, 1), "d_prime_m": round(d_prime)},
    )
    # CS ≈ 0.9 × VMA is a coherence check, not part of the computation.
    ratio = cs * 3.6 / estimate.vma_kmh
    estimate.notes.append(Note("level.note.cs_ratio", {"ratio": round(ratio * 100)}))
    return estimate


def from_records(records: Sequence[Tuple[float, float]]) -> LevelEstimate:
    """``(distance_m, seconds)`` pairs. Median VDOT from three records, mean of
    two, the one otherwise; the spread says how far to trust it."""
    if not records:
        raise LevelInputError("level.error.records_empty")
    values = []
    for distance_m, seconds in records:
        minutes = seconds / 60.0
        if distance_m <= 0 or seconds <= 0:
            raise LevelInputError("level.error.record_invalid")
        pace = seconds / (distance_m / 1000.0)
        if not _RECORD_PACE_S_PER_KM[0] <= pace <= _RECORD_PACE_S_PER_KM[1]:
            raise LevelInputError("level.error.record_pace")
        if not model.MIN_MINUTES <= minutes <= model.MAX_MINUTES:
            raise LevelInputError("level.error.record_duration")
        values.append((distance_m, model.vdot(distance_m, minutes)))

    vdots = [v for _, v in values]
    if len(vdots) >= 3:
        pivot = statistics.median(vdots)
    else:
        pivot = statistics.fmean(vdots)

    notes: List[Note] = []
    if len(vdots) == 1:
        confidence = "medium"
    else:
        spread = max(abs(v - pivot) for v in vdots)
        if spread <= _CONSISTENT_SPREAD:
            confidence = "high"
            notes.append(Note("level.note.records_consistent", {"spread": round(spread, 1)}))
        else:
            confidence = "low" if spread > _INCONSISTENT_SPREAD else "medium"
            # Name the outlier pair the reader can act on: the shortest vs the
            # longest distance, which is where the story usually is.
            by_distance = sorted(values)
            short, long = by_distance[0], by_distance[-1]
            if short[1] > long[1]:
                notes.append(Note("level.note.records_short_better", {
                    "short_m": int(short[0]), "long_m": int(long[0]),
                }))
            else:
                notes.append(Note("level.note.records_long_better", {
                    "short_m": int(short[0]), "long_m": int(long[0]),
                }))
    return _estimate(RECORDS, pivot, confidence, notes, {"records": float(len(vdots))})


def estimate(method: str, inputs: Dict[str, object]) -> LevelEstimate:
    """Dispatch on ``method`` with the raw inputs the form sends."""
    try:
        if method == HALF_COOPER:
            return from_half_cooper(float(inputs["distance_m"]))
        if method == CRITICAL_SPEED:
            return from_critical_speed(float(inputs["d3_m"]), float(inputs["d12_m"]))
        if method == RECORDS:
            rows = inputs.get("records") or []
            return from_records([
                (float(row["distance_m"]), float(row["seconds"])) for row in rows  # type: ignore[index]
            ])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, LevelInputError):
            raise
        raise LevelInputError("level.error.invalid")
    raise LevelInputError("level.error.method")


def hr_max_or_none(raw: Optional[object]) -> Optional[int]:
    if raw in (None, ""):
        return None
    value = int(float(raw))  # type: ignore[arg-type]
    if not 120 <= value <= 230:
        raise LevelInputError("level.error.hr_max")
    return value
