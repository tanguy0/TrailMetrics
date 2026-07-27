"""Time binning — the two x-axes every trend can use.

* **calendar** — real dates, binned to the day / week / month / quarter / year.
* **elapsed** — each group aligned to its own start, x = months since it began,
  so windows of different lengths stack from a common 0 and each stops at its own
  end. This is what used to be the long-term page's "season overlay"; it now
  works for any group of any plot.

Pure date arithmetic, no domain knowledge — generalized from the old
``progress/seasons`` helpers so the binning is shared by every plot instead of
belonging to one page.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple

from src.domain.spec.datasource import TimeWindow

# Mean Gregorian month; lets any granularity share one "months elapsed" axis.
AVG_DAYS_PER_MONTH = 30.4375

# "activity" keeps every activity as its own point (no binning at all).
GRANULARITIES: Tuple[str, ...] = (
    "activity", "day", "week", "month", "quarter", "year",
)


def to_date(value) -> date:
    """Coerce a ``datetime`` (or ``date``) to a plain ``date``."""
    return value.date() if isinstance(value, datetime) else value


def naive(value: datetime) -> datetime:
    """Drop tzinfo so binning never mixes aware and naive datetimes."""
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


# --- Calendar binning ------------------------------------------------------

def bin_start(d: date, granularity: str) -> date:
    """Calendar start of ``d``'s bin (the day, Monday, 1st, quarter, or Jan 1)."""
    if granularity in ("activity", "day"):
        return d
    if granularity == "week":
        return d - timedelta(days=d.weekday())
    if granularity == "month":
        return date(d.year, d.month, 1)
    if granularity == "quarter":
        return date(d.year, ((d.month - 1) // 3) * 3 + 1, 1)
    if granularity == "year":
        return date(d.year, 1, 1)
    raise ValueError(f"unknown granularity: {granularity!r}")


# --- Elapsed (since a group's start) binning --------------------------------

def _months_between(start: date, d: date) -> int:
    """Whole calendar months from ``start`` to ``d`` (0-based, floored)."""
    months = (d.year - start.year) * 12 + (d.month - start.month)
    if d.day < start.day:
        months -= 1
    return months


def elapsed_bin_index(d: date, start: date, granularity: str) -> int:
    """Index of ``d``'s bin counted from ``start`` (0-based) at ``granularity``."""
    if granularity in ("activity", "day"):
        return (d - start).days
    if granularity == "week":
        return (d - start).days // 7
    if granularity == "month":
        return _months_between(start, d)
    if granularity == "quarter":
        return _months_between(start, d) // 3
    if granularity == "year":
        return _months_between(start, d) // 12
    raise ValueError(f"unknown granularity: {granularity!r}")


def elapsed_bin_months(bin_index: int, granularity: str) -> float:
    """A bin's x-position in elapsed months, so all granularities share an axis."""
    if granularity in ("activity", "day"):
        return bin_index / AVG_DAYS_PER_MONTH
    if granularity == "week":
        return bin_index * 7 / AVG_DAYS_PER_MONTH
    if granularity == "month":
        return float(bin_index)
    if granularity == "quarter":
        return float(bin_index * 3)
    if granularity == "year":
        return float(bin_index * 12)
    raise ValueError(f"unknown granularity: {granularity!r}")


def window_length_months(window: TimeWindow) -> float:
    """A window's duration in months, both ends inclusive."""
    return window.length_days / AVG_DAYS_PER_MONTH


def max_window_months(windows: Sequence[TimeWindow]) -> Optional[float]:
    """Longest window in months — bounds the elapsed axis so it isn't ragged."""
    lengths = [window_length_months(w) for w in windows]
    return max(lengths) if lengths else None


# --- Default windows -------------------------------------------------------

def calendar_year_windows(dates: Sequence[datetime]) -> List[TimeWindow]:
    """One window per calendar year present in ``dates`` — the usual starting point."""
    years = sorted({to_date(d).year for d in dates if d is not None})
    return [
        TimeWindow(name=str(y), start=date(y, 1, 1), end=date(y, 12, 31))
        for y in years
    ]


def find_overlaps(windows: Sequence[TimeWindow]) -> List[Tuple[str, str]]:
    """Name pairs of windows whose ranges overlap.

    Overlap is legal — an activity simply lands in both groups — but it is almost
    always a typo, so the editor warns.
    """
    overlaps: List[Tuple[str, str]] = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            a, b = windows[i], windows[j]
            if a.start <= b.end and b.start <= a.end:
                overlaps.append((a.name, b.name))
    return overlaps
