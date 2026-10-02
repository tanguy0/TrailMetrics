"""The training zones Home and the level tool show — one definition, served by the API.

Pace zones are a %VMA range; the pace for the low end comes first in the table
most plans quote (lower %VMA is the slower pace). Heart-rate zones are ceilings
as a %HRmax, so HRmax is the only number to keep up to date. ``HR_PACE_ZONES``
says where each pace zone's effort sits on that same %HRmax scale — what Home's
heart-rate map draws.

These used to live in the web app (HomeScreen); they moved here so the level
estimate, Home and later the coaching pages read the same numbers.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PercentRange:
    key: str
    low_pct: float
    high_pct: float


VMA_PACE_ZONES: List[PercentRange] = [
    PercentRange("z2", 60, 65),
    PercentRange("endurance", 70, 75),
    PercentRange("threshold", 85, 90),
    PercentRange("intervals", 95, 100),
    PercentRange("reps", 105, 115),
]

# Each heart-rate zone's ceiling, as a fraction of HRmax.
HR_ZONE_MAX_PCT = [("z1", 0.70), ("z2", 0.77), ("z3", 0.87), ("z4", 0.91)]

HR_PACE_ZONES: List[PercentRange] = [
    PercentRange("z2", 68, 73),
    PercentRange("endurance", 75, 81),
    PercentRange("threshold", 84, 89),
    PercentRange("intervals", 91, 94),
    PercentRange("reps", 96, 100),
]


@dataclass(frozen=True)
class PaceZone:
    key: str
    low_pct: float
    high_pct: float
    # Fastest first, as density.md writes an interval: `fast–slow`.
    fast_s_per_km: float
    slow_s_per_km: float


def pace_zones(vma_pace_s_per_km: float) -> List[PaceZone]:
    return [
        PaceZone(
            zone.key, zone.low_pct, zone.high_pct,
            fast_s_per_km=vma_pace_s_per_km / (zone.high_pct / 100),
            slow_s_per_km=vma_pace_s_per_km / (zone.low_pct / 100),
        )
        for zone in VMA_PACE_ZONES
    ]


def hr_zone_ceilings(hr_max: Optional[int]) -> List[dict]:
    """Each zone's ceiling in bpm, truncated like a monitor reads it."""
    if not hr_max:
        return []
    return [{"key": key, "bpm": int(hr_max * pct)} for key, pct in HR_ZONE_MAX_PCT]


def definitions() -> dict:
    """The zone tables as the web app reads them."""
    return {
        "vma_pace": [vars(z) for z in VMA_PACE_ZONES],
        "hr_max_pct": [{"key": k, "pct": p} for k, p in HR_ZONE_MAX_PCT],
        "hr_pace": [vars(z) for z in HR_PACE_ZONES],
    }
