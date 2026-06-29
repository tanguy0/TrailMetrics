"""Re-aggregate per-activity summaries into plot-ready season trends.

Pure functions over a list of :class:`ActivityProgress` — no I/O, no plotting —
so the page can recompute any view instantly when a UI option changes, without
touching the raw streams again.

Overlay plots (mileage, elevation, average gradient) place every season on a
shared **reference year** x-axis so the curves stack on top of each other. The
gradient map instead runs along the real, continuous timeline.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain.progress.models import (
    ActivityProgress,
    GRADIENT_BAND_KEYS,
    PR_DISTANCES,
)
from src.domain.progress.records import record_progression

# A leap year, so both Feb 29 and day 366 exist when remapping dates for overlay.
REFERENCE_YEAR = 2000


# --- Personal records ------------------------------------------------------

def pr_progressions(
    activities: Sequence[ActivityProgress],
) -> Dict[str, List[Tuple[datetime, float]]]:
    """For each PR distance, the stepped record progression ``[(date, time_s)]``."""
    progressions: Dict[str, List[Tuple[datetime, float]]] = {}
    for label, _ in PR_DISTANCES:
        samples = [
            (a.date, a.best_efforts.get(label))
            for a in activities
            if a.best_efforts.get(label) is not None
        ]
        progressions[label] = record_progression(samples)
    return progressions


def current_records(
    activities: Sequence[ActivityProgress],
) -> Dict[str, Optional[Tuple[datetime, float]]]:
    """For each PR distance, the current record ``(date, time_s)`` or ``None``.

    The record is the *fastest* effort ever; its date is the day it was set.
    """
    records: Dict[str, Optional[Tuple[datetime, float]]] = {}
    for label, _ in PR_DISTANCES:
        best: Optional[Tuple[datetime, float]] = None
        for a in activities:
            t = a.best_efforts.get(label)
            if t is None:
                continue
            if best is None or t < best[1]:
                best = (a.date, t)
        records[label] = best
    return records


# --- Annual cumulative (distance / elevation) ------------------------------

@dataclass
class CumulativeYear:
    """One season's cumulative curve, on the reference-year x-axis."""

    year: int
    x: List[datetime] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    total: float = 0.0


def annual_cumulative(
    activities: Sequence[ActivityProgress],
    attr: str,
    scale: float = 1.0,
) -> List[CumulativeYear]:
    """Cumulative ``getattr(a, attr) * scale`` per calendar year over the year.

    Each season starts from a Jan-1 zero baseline and accrues at every activity
    (a staircase). Returns one :class:`CumulativeYear` per year, newest first.
    """
    by_year: Dict[int, List[ActivityProgress]] = defaultdict(list)
    for a in activities:
        by_year[a.date.year].append(a)

    series: List[CumulativeYear] = []
    for year in sorted(by_year, reverse=True):
        acts = sorted(by_year[year], key=lambda a: a.date)
        cy = CumulativeYear(year=year)
        cy.x.append(datetime(REFERENCE_YEAR, 1, 1))
        cy.y.append(0.0)
        cumulative = 0.0
        for a in acts:
            cumulative += float(getattr(a, attr)) * scale
            cy.x.append(_to_reference_date(a.date))
            cy.y.append(cumulative)
        cy.total = cumulative
        series.append(cy)
    return series


# --- Average gradient per season -------------------------------------------

@dataclass
class GradientYear:
    """One season's per-bin average-gradient curve (ΣD+ / Σdistance, %)."""

    year: int
    x: List[datetime] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    season_avg: float = 0.0


def avg_gradient_series(
    activities: Sequence[ActivityProgress],
    granularity: str = "week",
) -> List[GradientYear]:
    """Average gradient (%) per week/month, one curve per season.

    The average for a bin is ``Σ elevation_gain / Σ distance × 100`` over that
    bin's activities — always positive. ``season_avg`` applies the same ratio
    over the whole year. Returns one :class:`GradientYear` per year, newest first.
    """
    # (year, bin_index) -> [dist_sum, elev_sum]
    bins: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    year_totals: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0])

    for a in activities:
        idx = _bin_index(a.date, granularity)
        bins[(a.date.year, idx)][0] += a.distance_m
        bins[(a.date.year, idx)][1] += a.elevation_gain_m
        year_totals[a.date.year][0] += a.distance_m
        year_totals[a.date.year][1] += a.elevation_gain_m

    by_year: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for (year, idx), (dist, elev) in bins.items():
        by_year[year].append((idx, dist, elev))

    series: List[GradientYear] = []
    for year in sorted(by_year, reverse=True):
        gy = GradientYear(year=year)
        for idx, dist, elev in sorted(by_year[year]):
            if dist <= 0:
                continue
            gy.x.append(_bin_reference_date(idx, granularity))
            gy.y.append(elev / dist * 100.0)
        ytot = year_totals[year]
        gy.season_avg = (ytot[1] / ytot[0] * 100.0) if ytot[0] > 0 else 0.0
        series.append(gy)
    return series


# --- Power-to-HR efficiency -------------------------------------------------

@dataclass
class PowerHrSeries:
    """One season's weekly power-to-HR points, on the *real* timeline.

    Unlike the season overlays, ``x`` keeps real calendar dates so the seasons
    sit end-to-end on one continuous axis; splitting into one series per year is
    purely so each season draws in its own color (the color flips at Jan 1).
    """

    year: int
    x: List[datetime] = field(default_factory=list)
    y: List[float] = field(default_factory=list)


def power_hr_weekly(
    activities: Sequence[ActivityProgress],
    granularity: str = "week",
) -> List[PowerHrSeries]:
    """Average power-to-HR per bin, split into one real-timeline series per year.

    Each session contributes its session-average ratio; the bin value is the
    mean across that bin's sessions. Sessions without power-to-HR (no weight, no
    HR) are skipped. Returns one :class:`PowerHrSeries` per year, oldest first
    so the timeline reads left to right.
    """
    # bin start date -> list of session ratios
    bins: Dict[date, List[float]] = defaultdict(list)
    for a in activities:
        if a.power_to_hr is None:
            continue
        bins[_bin_start(a.date, granularity)].append(a.power_to_hr)

    by_year: Dict[int, PowerHrSeries] = {}
    for start in sorted(bins):
        values = bins[start]
        if not values:
            continue
        series = by_year.setdefault(start.year, PowerHrSeries(year=start.year))
        series.x.append(datetime(start.year, start.month, start.day))
        series.y.append(sum(values) / len(values))

    return [by_year[year] for year in sorted(by_year)]


# --- Gradient map ----------------------------------------------------------

@dataclass
class GradientMap:
    """% of moving time per gradient band, per bin, along the real timeline."""

    x: List[datetime] = field(default_factory=list)
    band_pct: Dict[str, List[float]] = field(default_factory=dict)


def gradient_map(
    activities: Sequence[ActivityProgress],
    *,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    granularity: str = "week",
) -> GradientMap:
    """Per-bin share of moving time in each gradient band (sums to 100%).

    Activities outside ``[from_date, to_date]`` (inclusive, ``None`` = open) are
    dropped. Each bin's band seconds are summed across its activities and
    normalised to percentages. Bins are ordered along the real calendar timeline.
    """
    # bin start date -> {band key: seconds}
    bins: Dict[date, Dict[str, float]] = defaultdict(
        lambda: {k: 0.0 for k in GRADIENT_BAND_KEYS}
    )
    for a in activities:
        if from_date is not None and a.date < from_date:
            continue
        if to_date is not None and a.date > to_date:
            continue
        start = _bin_start(a.date, granularity)
        for key in GRADIENT_BAND_KEYS:
            bins[start][key] += a.band_seconds.get(key, 0.0)

    result = GradientMap(band_pct={k: [] for k in GRADIENT_BAND_KEYS})
    for start in sorted(bins):
        total = sum(bins[start].values())
        if total <= 0:
            continue
        result.x.append(datetime(start.year, start.month, start.day))
        for key in GRADIENT_BAND_KEYS:
            result.band_pct[key].append(bins[start][key] / total * 100.0)
    return result


# --- Date / binning helpers ------------------------------------------------

def _to_reference_date(d: datetime) -> datetime:
    """Map a date onto the reference year (keep month/day) for season overlay."""
    return datetime(REFERENCE_YEAR, d.month, d.day)


def _week_of_year(d: datetime) -> int:
    """Simple 1-based week index within the calendar year (1…53)."""
    return (d.timetuple().tm_yday - 1) // 7 + 1


def _bin_index(d: datetime, granularity: str) -> int:
    """Index of the activity's bin within its own year (week 1…53 or month 1…12)."""
    if granularity == "month":
        return d.month
    return _week_of_year(d)


def _bin_reference_date(idx: int, granularity: str) -> datetime:
    """Representative reference-year date for a within-year bin index."""
    if granularity == "month":
        return datetime(REFERENCE_YEAR, idx, 1)
    return datetime(REFERENCE_YEAR, 1, 1) + timedelta(weeks=idx - 1)


def _bin_start(d: datetime, granularity: str) -> date:
    """Real calendar start of the activity's bin (month 1st or week Monday)."""
    if granularity == "month":
        return date(d.year, d.month, 1)
    day = date(d.year, d.month, d.day)
    return day - timedelta(days=day.weekday())
