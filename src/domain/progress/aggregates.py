"""Re-aggregate per-activity summaries into plot-ready season trends.

Pure functions over a list of :class:`ActivityProgress` — no I/O, no plotting —
so the page can recompute any view instantly when a UI option changes, without
touching the raw streams again.

Every trend is grouped by user-defined :class:`Season` periods and can be drawn
two ways (see :func:`metric_series`):

* **overlay** — each season aligned to its own start, x = months elapsed, so the
  seasons stack from a common 0 (activities outside every season are dropped);
* **continuous** — the real calendar timeline, one colored curve per season plus
  a greyed "unassigned" curve for activities outside every season.

Both support day / week / month / quarter granularity. The gradient map keeps its
own real, continuous timeline.
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
from src.domain.progress.seasons import (
    Season,
    assign_season_index,
    continuous_bin_start,
    overlay_bin_index,
    overlay_bin_months,
    to_date,
)
from src.domain.races.smoothing import smooth_uniform_series


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


# --- Season curves (mileage / elevation / gradient / power-to-HR) -----------

# Season index used for the "outside every season" curve (continuous mode only).
UNASSIGNED_INDEX = -1


@dataclass
class SeasonCurve:
    """One season's curve for an overlay/continuous plot.

    ``index`` is the season's position in the user's list (or
    :data:`UNASSIGNED_INDEX` for the greyed out-of-season curve); the plotting
    layer maps it to a stable color. ``x`` holds elapsed-months floats (overlay)
    or real ``datetime``s (continuous). ``total`` is the season's own aggregate
    (sum for mileage/elevation, average % for gradient), for the side table.
    """

    name: str
    index: int
    x: List = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    total: float = 0.0


def metric_series(
    activities: Sequence[ActivityProgress],
    attr: str,
    scale: float,
    seasons: Sequence[Season],
    *,
    mode: str,
    granularity: str,
    cumulative: bool,
    unassigned_label: str = "—",
) -> List[SeasonCurve]:
    """Per-season curves of a summed metric (``getattr(a, attr) * scale``).

    ``mode`` is ``"overlay"`` or ``"continuous"``; ``cumulative`` picks a running
    total vs. each bin's own total.
    """
    if mode == "overlay":
        return _overlay_sum(activities, attr, scale, seasons, granularity, cumulative)
    return _continuous_sum(
        activities, attr, scale, seasons, granularity, cumulative, unassigned_label
    )


def _overlay_sum(activities, attr, scale, seasons, granularity, cumulative):
    per_season: List[Dict[int, float]] = [defaultdict(float) for _ in seasons]
    totals: List[float] = [0.0] * len(seasons)
    for a in activities:
        d = to_date(a.date)
        si = assign_season_index(d, seasons)
        if si is None:
            continue  # overlay drops out-of-season activities
        value = float(getattr(a, attr)) * scale
        per_season[si][overlay_bin_index(d, seasons[si].start, granularity)] += value
        totals[si] += value

    curves: List[SeasonCurve] = []
    for si, bins in enumerate(per_season):
        if not bins:
            continue
        curve = SeasonCurve(name=seasons[si].name, index=si, total=totals[si])
        if cumulative:
            curve.x.append(0.0)
            curve.y.append(0.0)
            running = 0.0
            for bi in sorted(bins):
                running += bins[bi]
                curve.x.append(overlay_bin_months(bi, granularity))
                curve.y.append(running)
        else:
            for bi in sorted(bins):
                curve.x.append(overlay_bin_months(bi, granularity))
                curve.y.append(bins[bi])
        curves.append(curve)
    return curves


def _continuous_sum(
    activities, attr, scale, seasons, granularity, cumulative, unassigned_label
):
    bins: Dict[date, float] = defaultdict(float)
    for a in activities:
        d = to_date(a.date)
        bins[continuous_bin_start(d, granularity)] += float(getattr(a, attr)) * scale

    curves: Dict[int, SeasonCurve] = {}
    running = 0.0
    prev: Optional[Tuple[datetime, float, int]] = None
    for start in sorted(bins):
        increment = bins[start]
        running += increment
        y = running if cumulative else increment
        si = assign_season_index(start, seasons)
        key = si if si is not None else UNASSIGNED_INDEX
        curve = curves.get(key)
        if curve is None:
            name = seasons[si].name if si is not None else unassigned_label
            curve = SeasonCurve(name=name, index=key)
            curves[key] = curve
        x = datetime(start.year, start.month, start.day)
        prev_key = prev[2] if prev is not None else None
        if key == UNASSIGNED_INDEX:
            _break_reentry(curve, key, prev_key)
        elif cumulative and prev is not None and prev_key != key:
            # Keep the colored cumulative line unbroken across season changes.
            curve.x.append(prev[0])
            curve.y.append(prev[1])
        curve.x.append(x)
        curve.y.append(y)
        curve.total += increment
        prev = (x, y, key)

    return _ordered_curves(curves)


def gradient_series(
    activities: Sequence[ActivityProgress],
    seasons: Sequence[Season],
    *,
    mode: str,
    granularity: str,
    unassigned_label: str = "—",
) -> List[SeasonCurve]:
    """Per-season average-gradient (%) curves = Σ elevation ÷ Σ distance per bin."""
    if mode == "overlay":
        return _overlay_gradient(activities, seasons, granularity)
    return _continuous_gradient(activities, seasons, granularity, unassigned_label)


def _overlay_gradient(activities, seasons, granularity):
    # per season: bin_index -> [dist_sum, elev_sum]; plus season totals.
    per_season: List[Dict[int, List[float]]] = [
        defaultdict(lambda: [0.0, 0.0]) for _ in seasons
    ]
    totals: List[List[float]] = [[0.0, 0.0] for _ in seasons]
    for a in activities:
        d = to_date(a.date)
        si = assign_season_index(d, seasons)
        if si is None:
            continue
        bi = overlay_bin_index(d, seasons[si].start, granularity)
        per_season[si][bi][0] += a.distance_m
        per_season[si][bi][1] += a.elevation_gain_m
        totals[si][0] += a.distance_m
        totals[si][1] += a.elevation_gain_m

    curves: List[SeasonCurve] = []
    for si, bins in enumerate(per_season):
        if not bins:
            continue
        curve = SeasonCurve(name=seasons[si].name, index=si)
        for bi in sorted(bins):
            dist, elev = bins[bi]
            if dist <= 0:
                continue
            curve.x.append(overlay_bin_months(bi, granularity))
            curve.y.append(elev / dist * 100.0)
        curve.total = totals[si][1] / totals[si][0] * 100.0 if totals[si][0] > 0 else 0.0
        curves.append(curve)
    return curves


def _continuous_gradient(activities, seasons, granularity, unassigned_label):
    bins: Dict[date, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for a in activities:
        start = continuous_bin_start(to_date(a.date), granularity)
        bins[start][0] += a.distance_m
        bins[start][1] += a.elevation_gain_m

    curves: Dict[int, SeasonCurve] = {}
    prev_key: Optional[int] = None
    for start in sorted(bins):
        dist, elev = bins[start]
        if dist <= 0:
            continue
        si = assign_season_index(start, seasons)
        key = si if si is not None else UNASSIGNED_INDEX
        curve = curves.get(key)
        if curve is None:
            name = seasons[si].name if si is not None else unassigned_label
            curve = SeasonCurve(name=name, index=key)
            curves[key] = curve
        _break_reentry(curve, key, prev_key)
        curve.x.append(datetime(start.year, start.month, start.day))
        curve.y.append(elev / dist * 100.0)
        prev_key = key
    return _ordered_curves(curves)


def power_hr_series(
    activities: Sequence[ActivityProgress],
    seasons: Sequence[Season],
    *,
    granularity: str = "week",
    from_date: Optional[datetime] = None,
    to_date_bound: Optional[datetime] = None,
    rolling_window: Optional[int] = None,
    savgol_window: Optional[int] = None,
    savgol_polyorder: int = 2,
    unassigned_label: str = "—",
) -> List[SeasonCurve]:
    """Per-bin average power-to-HR on the continuous timeline, colored per season.

    Sessions without power-to-HR (no weight / no HR) or outside
    ``[from_date, to_date_bound]`` (inclusive) are skipped. ``rolling_window`` /
    ``savgol_window`` smooth the whole binned curve (windows in points/bins)
    before it is split per season for coloring, so smoothing carries across
    season boundaries.
    """
    bins: Dict[date, List[float]] = defaultdict(list)
    for a in activities:
        if a.power_to_hr is None:
            continue
        if from_date is not None and a.date < from_date:
            continue
        if to_date_bound is not None and a.date > to_date_bound:
            continue
        bins[continuous_bin_start(to_date(a.date), granularity)].append(a.power_to_hr)

    starts = sorted(bins)
    if not starts:
        return []
    values = [sum(bins[s]) / len(bins[s]) for s in starts]
    values = smooth_uniform_series(
        values,
        rolling_window=rolling_window,
        savgol_window=savgol_window,
        polyorder=savgol_polyorder,
    )

    curves: Dict[int, SeasonCurve] = {}
    prev_key: Optional[int] = None
    for start, value in zip(starts, values):
        si = assign_season_index(start, seasons)
        key = si if si is not None else UNASSIGNED_INDEX
        curve = curves.get(key)
        if curve is None:
            name = seasons[si].name if si is not None else unassigned_label
            curve = SeasonCurve(name=name, index=key)
            curves[key] = curve
        _break_reentry(curve, key, prev_key)
        curve.x.append(datetime(start.year, start.month, start.day))
        curve.y.append(value)
        prev_key = key
    return _ordered_curves(curves)


def _ordered_curves(curves: Dict[int, SeasonCurve]) -> List[SeasonCurve]:
    """Seasons in order, with the unassigned (greyed) curve last."""
    return [
        curves[k]
        for k in sorted(curves, key=lambda k: (k == UNASSIGNED_INDEX, k))
    ]


def _break_reentry(curve: SeasonCurve, key: int, prev_key: Optional[int]) -> None:
    """Insert a gap so the grey (unassigned) line never bridges an assigned season.

    Called before adding an unassigned point: if the previous point belonged to a
    season, this unassigned point starts a fresh run, so a ``None`` break is added
    to keep the two sides of the season disconnected.
    """
    if key == UNASSIGNED_INDEX and curve.x and prev_key not in (None, UNASSIGNED_INDEX):
        curve.x.append(None)
        curve.y.append(None)


# --- Per-season side tables (mode-independent) ------------------------------

def season_totals(
    activities: Sequence[ActivityProgress],
    attr: str,
    scale: float,
    seasons: Sequence[Season],
) -> List[Tuple[str, float]]:
    """``(season name, Σ attr·scale)`` per season, newest first."""
    totals = [0.0] * len(seasons)
    for a in activities:
        si = assign_season_index(to_date(a.date), seasons)
        if si is not None:
            totals[si] += float(getattr(a, attr)) * scale
    rows = [(seasons[i].name, totals[i]) for i in range(len(seasons))]
    return list(reversed(rows))


def season_gradient_averages(
    activities: Sequence[ActivityProgress],
    seasons: Sequence[Season],
) -> List[Tuple[str, float]]:
    """``(season name, Σ elevation ÷ Σ distance %)`` per season, newest first."""
    sums = [[0.0, 0.0] for _ in seasons]
    for a in activities:
        si = assign_season_index(to_date(a.date), seasons)
        if si is not None:
            sums[si][0] += a.distance_m
            sums[si][1] += a.elevation_gain_m
    rows = [
        (seasons[i].name, (sums[i][1] / sums[i][0] * 100.0) if sums[i][0] > 0 else 0.0)
        for i in range(len(seasons))
    ]
    return list(reversed(rows))


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
    to_date_bound: Optional[datetime] = None,
    granularity: str = "week",
) -> GradientMap:
    """Per-bin share of moving time in each gradient band (sums to 100%).

    Activities outside ``[from_date, to_date_bound]`` (inclusive, ``None`` =
    open) are dropped. Each bin's band seconds are summed across its activities
    and normalised to percentages, ordered along the real calendar timeline.
    """
    bins: Dict[date, Dict[str, float]] = defaultdict(
        lambda: {k: 0.0 for k in GRADIENT_BAND_KEYS}
    )
    for a in activities:
        if from_date is not None and a.date < from_date:
            continue
        if to_date_bound is not None and a.date > to_date_bound:
            continue
        start = continuous_bin_start(to_date(a.date), granularity)
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
