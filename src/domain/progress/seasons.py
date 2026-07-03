"""User-defined *seasons* — arbitrary named time periods — for the long-term page.

By default a season is a calendar year, but the page lets the user define any set
of named periods (e.g. "Marathon block", "Summer base"). Every season overlay
then aligns each season to its own start: the x-axis is *months elapsed since the
season started*, so seasons of different lengths stack from a common 0 and each
simply stops at its own end.

This module holds the :class:`Season` model plus the pure binning helpers shared
by the aggregation layer — both the *overlay* (elapsed-since-start) axis and the
*continuous* (real calendar) axis, at day / week / month / quarter granularity.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple

# Mean Gregorian month length; used to express an elapsed duration in "months"
# on the overlay axis regardless of the aggregation granularity.
AVG_DAYS_PER_MONTH = 30.4375

GRANULARITIES = ("day", "week", "month", "quarter")


@dataclass
class Season:
    """A named, inclusive ``[start, end]`` date period treated as one season."""

    name: str
    start: date
    end: date


def to_date(value) -> date:
    """Coerce a ``datetime`` (or ``date``) to a plain ``date``."""
    return value.date() if isinstance(value, datetime) else value


def calendar_year_seasons(activities: Sequence) -> List[Season]:
    """Default seasons: one per calendar year present in the activities."""
    years = sorted({to_date(a.date).year for a in activities})
    return [
        Season(name=str(y), start=date(y, 1, 1), end=date(y, 12, 31)) for y in years
    ]


def find_overlaps(seasons: Sequence[Season]) -> List[Tuple[str, str]]:
    """Return the (name, name) pairs of seasons whose date ranges overlap."""
    overlaps: List[Tuple[str, str]] = []
    for i in range(len(seasons)):
        for j in range(i + 1, len(seasons)):
            a, b = seasons[i], seasons[j]
            if a.start <= b.end and b.start <= a.end:
                overlaps.append((a.name, b.name))
    return overlaps


def assign_season_index(d: date, seasons: Sequence[Season]) -> Optional[int]:
    """Index of the first season containing ``d``, or ``None`` if in none."""
    for i, s in enumerate(seasons):
        if s.start <= d <= s.end:
            return i
    return None


def season_length_months(s: Season) -> float:
    """Season duration in months (inclusive of both ends)."""
    return ((s.end - s.start).days + 1) / AVG_DAYS_PER_MONTH


# --- Continuous (real calendar) binning ------------------------------------

def continuous_bin_start(d: date, granularity: str) -> date:
    """Real calendar start of ``d``'s bin (day / Monday / 1st / quarter 1st)."""
    if granularity == "day":
        return d
    if granularity == "week":
        return d - timedelta(days=d.weekday())
    if granularity == "month":
        return date(d.year, d.month, 1)
    if granularity == "quarter":
        q_first_month = ((d.month - 1) // 3) * 3 + 1
        return date(d.year, q_first_month, 1)
    raise ValueError(f"unknown granularity: {granularity!r}")


# --- Overlay (elapsed-since-season-start) binning --------------------------

def _months_between(start: date, d: date) -> int:
    """Whole calendar months from ``start`` to ``d`` (0-based, floor)."""
    months = (d.year - start.year) * 12 + (d.month - start.month)
    if d.day < start.day:
        months -= 1
    return months


def overlay_bin_index(d: date, start: date, granularity: str) -> int:
    """Index of ``d``'s bin measured from ``start`` (0-based) at ``granularity``."""
    if granularity == "day":
        return (d - start).days
    if granularity == "week":
        return (d - start).days // 7
    if granularity == "month":
        return _months_between(start, d)
    if granularity == "quarter":
        return _months_between(start, d) // 3
    raise ValueError(f"unknown granularity: {granularity!r}")


def overlay_bin_months(bin_index: int, granularity: str) -> float:
    """Elapsed-months x-position of a bin, so every granularity shares one axis."""
    if granularity == "day":
        return bin_index / AVG_DAYS_PER_MONTH
    if granularity == "week":
        return bin_index * 7 / AVG_DAYS_PER_MONTH
    if granularity == "month":
        return float(bin_index)
    if granularity == "quarter":
        return float(bin_index * 3)
    raise ValueError(f"unknown granularity: {granularity!r}")
