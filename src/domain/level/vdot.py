"""The common pivot: Daniels & Gilbert's VDOT, and the VMA it implies.

For a performance of duration ``t`` (minutes) at speed ``v`` (m/min)::

    VO2(v) = -4.60 + 0.182258·v + 0.000104·v²
    pct(t) = 0.8 + 0.1894393·e^(-0.012778·t) + 0.2989558·e^(-0.1932605·t)
    VDOT   = VO2(v) / pct(t)

The VMA is the speed whose VO2 equals the VDOT — the positive root of the
quadratic above.
"""

import math

# The model is only meaningful between these durations (level.md § Records).
MIN_MINUTES = 3.0
MAX_MINUTES = 6 * 60.0


def vo2_at(speed_m_per_min: float) -> float:
    return -4.60 + 0.182258 * speed_m_per_min + 0.000104 * speed_m_per_min ** 2


def sustainable_fraction(minutes: float) -> float:
    """The fraction of VO2max that can be held for ``minutes``."""
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * minutes)
        + 0.2989558 * math.exp(-0.1932605 * minutes)
    )


def vdot(distance_m: float, minutes: float) -> float:
    if distance_m <= 0 or minutes <= 0:
        raise ValueError("distance and duration must be positive")
    return vo2_at(distance_m / minutes) / sustainable_fraction(minutes)


def vma_kmh(vdot_value: float) -> float:
    """Speed at VO2max, in km/h: the root of ``VO2(v) = VDOT``."""
    a, b, c = 0.000104, 0.182258, -(4.60 + vdot_value)
    speed_m_per_min = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return speed_m_per_min * 60.0 / 1000.0


def pace_s_per_km(speed_kmh: float) -> float:
    return 3600.0 / speed_kmh
