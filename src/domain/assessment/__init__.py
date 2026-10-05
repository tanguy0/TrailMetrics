"""Where a runner stands against a reference runner, in five words.

The GAP and durability profiles both turn a curve the runner does not read into
one level per terrain or effort: how much more (or less) it costs *them* than the
reference runner, as a percentage, then bucketed. Positive means it costs more,
which is worse; the same scale serves both profiles.

    > +15 %          poor
    +5 % … +15 %     limited
    −5 % … +5 %      average
    −15 % … −5 %     good
    < −15 %          excellent
"""

from dataclasses import dataclass
from typing import Optional

POOR = "poor"
LIMITED = "limited"
AVERAGE = "average"
GOOD = "good"
EXCELLENT = "excellent"
INSUFFICIENT = "insufficient"

LEVELS = (EXCELLENT, GOOD, AVERAGE, LIMITED, POOR)

# Symmetric around the reference; a bound belongs to the level nearer average.
AVERAGE_WITHIN_PCT = 5.0
CLEAR_BEYOND_PCT = 15.0


def rate(extra_cost_pct: Optional[float]) -> str:
    """The level of an extra cost against the reference; ``insufficient`` for none."""
    if extra_cost_pct is None:
        return INSUFFICIENT
    size = abs(extra_cost_pct)
    if size <= AVERAGE_WITHIN_PCT:
        return AVERAGE
    if extra_cost_pct > 0:
        return LIMITED if size <= CLEAR_BEYOND_PCT else POOR
    return GOOD if size <= CLEAR_BEYOND_PCT else EXCELLENT


@dataclass(frozen=True)
class Assessment:
    """One terrain or effort of a profile, rated."""

    key: str
    # ``None`` when there is nothing to compare against.
    extra_cost_pct: Optional[float]
    level: str

    @staticmethod
    def of(key: str, extra_cost_pct: Optional[float]) -> "Assessment":
        return Assessment(key, extra_cost_pct, rate(extra_cost_pct))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "extra_cost_pct": None if self.extra_cost_pct is None else round(self.extra_cost_pct, 1),
            "level": self.level,
        }
