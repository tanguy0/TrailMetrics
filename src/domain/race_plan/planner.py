"""The race plan itself: grid, pacing, sections and aid-station legs.

**Pacing.** A GAP curve gives, for each gradient, the adjuster
``a(g) = GAP / speed`` — how much slower than flat a runner goes on that gradient
at equal effort. Holding a constant GAP pace ``P`` (seconds per km) therefore
means running each stretch at ``P · a(g)``, and the finish time is

    T(P) = P · Σ ds_i · a(g_i)

which is *linear* in ``P``. So the constant GAP pace that yields exactly the target
time is not searched for, it is solved: ``P = T / Σ ds_i · a(g_i)``.

**Durability.** Optionally each stretch's cost is also multiplied by ``phi(t) >= 1``
(the cost drift of a long effort, :mod:`src.domain.durability`), which keeps the
equation linear in ``P`` — ``P = T / Σ ds_i · a(g_i) · phi_i`` — but makes ``phi``
depend on the elapsed time it produces, hence a short fixed-point iteration.

**Two smoothings.** GPX elevation is noisy at the metre scale, and that noise both
inflates D+ and turns every 10 m step into a spurious 15 % ramp. The pacing reads a
light smoothing (a stride-scale ~150 m window) so short real ramps still show on the
pace graph; section detection reads a heavy one (~500 m) because a "climb" is a
landscape feature, not a bump.

**Sections.** Each grid step is labelled climb / descent / flat by the heavily
smoothed gradient, the labels are run-length encoded, and the runs are then
consolidated until every section is worth a row in a race plan: long enough to
matter, and — for climbs and descents — with enough elevation change to deserve the
name. See :func:`detect_sections`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

from src.domain.durability.solver import (
    DurabilitySolution,
    RouteDurability,
    solve_pacing,
    solve_route,
)
from src.domain.models.gap import GapCurve
from src.domain.race_plan.gpx import CoursePoints, GpxError

# Grid spacing, in metres. Fine enough that a 50 m ramp is several points, coarse
# enough that a 170 km ultra is ~17k points.
STEP_M = 10.0

PACING_SMOOTHING_M = 150.0
SECTION_SMOOTHING_M = 500.0

# Gradients are in m/km (the GAP curves' unit): 30 m/km is 3 %.
FLAT_GRADE = 30.0
# Clamp before evaluating the adjuster: past ±60 % a GPX gradient is a cliff or a
# glitch, and a linearly extrapolated curve should not be asked about it.
MAX_GRADE = 600.0

# A climb or descent must gain or lose at least this much to be called one.
MIN_SECTION_ELEVATION_M = 25.0
# A flat stretch this steep over this much elevation is really a gentle climb.
GENTLE_GRADE = 20.0
GENTLE_ELEVATION_M = 50.0

CLIMB = "climb"
DESCENT = "descent"
FLAT = "flat"
_LABELS = {1: CLIMB, -1: DESCENT, 0: FLAT}


class PlanError(ValueError):
    """Inputs that cannot produce a plan, with a translatable reason key."""

    def __init__(self, reason_key: str):
        super().__init__(reason_key)
        self.reason_key = reason_key


@dataclass
class Course:
    """The course on a regular distance grid."""

    distance: np.ndarray          # m, nodes
    elevation: np.ndarray         # m, raw (resampled)
    elevation_smooth: np.ndarray  # m, light smoothing — pacing, D+/D-
    grade: np.ndarray             # m/km, per interval (len = nodes - 1)
    section_grade: np.ndarray     # m/km, per interval, heavy smoothing

    @property
    def total_m(self) -> float:
        return float(self.distance[-1])

    def elevation_gain(self, start: int = 0, end: Optional[int] = None) -> Tuple[float, float]:
        """``(D+, D-)`` between two node indices, on the smoothed profile."""
        diffs = np.diff(self.elevation_smooth[start:(end if end is not None else None)])
        return float(diffs[diffs > 0].sum()), float(-diffs[diffs < 0].sum())


@dataclass
class Stretch:
    """One row of a plan: a section, or a leg between aid stations."""

    index: int
    kind: str                 # climb | descent | flat, or "leg"
    start_m: float
    end_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    start_elevation_m: float
    end_elevation_m: float
    start_time_s: float
    end_time_s: float
    label: str = ""

    @property
    def distance_m(self) -> float:
        return self.end_m - self.start_m

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def pace_s_per_km(self) -> float:
        return self.duration_s / (self.distance_m / 1000) if self.distance_m > 0 else float("nan")

    @property
    def average_grade_pct(self) -> float:
        if self.distance_m <= 0:
            return 0.0
        return (self.end_elevation_m - self.start_elevation_m) / self.distance_m * 100


@dataclass
class RacePlan:
    course: Course
    target_time_s: float
    # The fresh GAP pace: the effort-equivalent pace at the start line.
    gap_pace_s_per_km: float
    pace: np.ndarray          # s/km, per interval
    elapsed: np.ndarray       # s, per node
    sections: List[Stretch] = field(default_factory=list)
    legs: List[Stretch] = field(default_factory=list)
    # Aid-station distances that were dropped (outside the course), in km.
    ignored_aid_stations_km: List[float] = field(default_factory=list)
    # Set when the plan accounts for durability (see src.domain.durability.solver).
    durability: Optional[DurabilitySolution] = None

    @property
    def average_pace_s_per_km(self) -> float:
        return self.target_time_s / (self.course.total_m / 1000)

    @property
    def gap_pace_profile(self) -> np.ndarray:
        """Effort-equivalent GAP pace per interval: ``P · phi`` (constant without durability)."""
        multiplier = (self.durability.interval_multiplier if self.durability is not None
                      else np.ones_like(self.pace))
        return self.gap_pace_s_per_km * multiplier


# --- Course ---------------------------------------------------------------

def build_course(points: CoursePoints) -> Course:
    cumulative = np.concatenate([[0.0], np.cumsum(_haversine_m(points.lat, points.lon))])
    # Consecutive duplicates (a paused recording, a doubled point) would make the
    # distance non-increasing, which `np.interp` silently mishandles.
    keep = np.concatenate([[True], np.diff(cumulative) > 0.01])
    cumulative, elevation = cumulative[keep], points.elevation[keep]

    total = float(cumulative[-1])
    if total < 20 * STEP_M:
        raise GpxError("race_plan.error.gpx_too_short")

    n = int(np.floor(total / STEP_M))
    grid = np.arange(n + 1) * STEP_M
    if total - grid[-1] > 1:
        grid = np.append(grid, total)
    resampled = np.interp(grid, cumulative, elevation)

    light = _smooth(resampled, PACING_SMOOTHING_M)
    heavy = _smooth(resampled, SECTION_SMOOTHING_M)
    ds = np.diff(grid)
    return Course(
        distance=grid,
        elevation=resampled,
        elevation_smooth=light,
        grade=np.diff(light) / ds * 1000,
        section_grade=np.diff(heavy) / ds * 1000,
    )


def _haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    phi = np.radians(lat)
    dphi = np.diff(phi)
    dlmb = np.radians(np.diff(lon))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dlmb / 2) ** 2
    return 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _smooth(values: np.ndarray, window_m: float) -> np.ndarray:
    window = min(int(round(window_m / STEP_M)), len(values))
    if window % 2 == 0:
        window -= 1  # savgol wants an odd window no longer than the series
    if window < 5:
        return values.copy()
    return savgol_filter(values, window, 2, mode="interp")


# --- Pacing ---------------------------------------------------------------

def adjuster(curve: GapCurve, reference: Optional[GapCurve] = None):
    """``grade (m/km) → a(g)``, normalized so ``a(0) = 1``.

    Inside the curve's own range the adjuster is its linear interpolation. Outside
    it, a curve *with* a ``reference`` continues along the reference's shape, scaled
    to meet the curve at its edge — a personalized curve fitted on an athlete who
    never ran a 30 % slope has nothing to say about one, but the course may have it.
    Without a reference (the reference curves themselves), the end segments are
    extended linearly.
    """
    centers, means = _clean(curve)
    ref = _clean(reference) if reference is not None else None

    def raw(g: np.ndarray) -> np.ndarray:
        g = np.clip(np.asarray(g, dtype=float), -MAX_GRADE, MAX_GRADE)
        inside = np.interp(g, centers, means)
        if ref is not None:
            ref_c, ref_m = ref
            ref_at = lambda x: _extrapolate(x, ref_c, ref_m)  # noqa: E731
            low = ref_at(g) * (means[0] / ref_at(centers[0]))
            high = ref_at(g) * (means[-1] / ref_at(centers[-1]))
        else:
            low = high = _extrapolate(g, centers, means)
        return np.where(g < centers[0], low, np.where(g > centers[-1], high, inside))

    flat = float(raw(np.array([0.0]))[0])
    return lambda g: raw(g) / flat


def _clean(curve: GapCurve) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(curve.bin_centers, dtype=float)
    means = np.asarray(curve.means, dtype=float)
    ok = np.isfinite(centers) & np.isfinite(means) & (means > 0)
    centers, means = centers[ok], means[ok]
    order = np.argsort(centers)
    return centers[order], means[order]


def _extrapolate(g: np.ndarray, centers: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Linear interpolation, continued linearly past both ends."""
    g = np.asarray(g, dtype=float)
    out = np.interp(g, centers, means)
    low_slope = (means[1] - means[0]) / (centers[1] - centers[0])
    high_slope = (means[-1] - means[-2]) / (centers[-1] - centers[-2])
    out = np.where(g < centers[0], means[0] + (g - centers[0]) * low_slope, out)
    out = np.where(g > centers[-1], means[-1] + (g - centers[-1]) * high_slope, out)
    # A downhill slope steep enough to extrapolate below zero would mean "faster
    # than infinitely fast"; floor it well above that.
    return np.maximum(out, 0.3)


def usable_curve(curve: Optional[GapCurve]) -> bool:
    """Enough of a curve to plan on: several points, spanning the flat."""
    if curve is None:
        return False
    centers, _ = _clean(curve)
    return centers.size >= 4 and centers[0] < 0 < centers[-1]


def plan_race(
    course: Course,
    target_time_s: float,
    adjust,
    aid_stations_km: Sequence[float] = (),
    aid_station_names: Optional[Sequence[str]] = None,
    durability: Optional[RouteDurability] = None,
) -> RacePlan:
    """Pace the course for the target time.

    Without ``durability`` the cost of each stretch is ``a(g)`` and the GAP pace is
    constant. With it, the cost is ``phi(t) · a(g)``: the same closed form, solved
    iteratively because ``phi`` depends on elapsed time — the plan then starts
    faster than average and finishes slower, for the same target.
    """
    if not np.isfinite(target_time_s) or target_time_s <= 0:
        raise PlanError("race_plan.error.target_time")

    ds_km = np.diff(course.distance) / 1000
    factors = adjust(course.grade)
    solution = None
    if durability is not None:
        descent_m = np.maximum(-np.diff(course.elevation_smooth), 0.0)
        solution = solve_route(durability, ds_km, factors, descent_m, target_time_s)
        gap_pace, pace, elapsed = solution.gap_pace_s_per_km, solution.pace, solution.elapsed
    else:
        gap_pace, pace, elapsed = solve_pacing(ds_km, factors, target_time_s,
                                               np.ones_like(factors))

    plan = RacePlan(
        course=course,
        target_time_s=target_time_s,
        gap_pace_s_per_km=gap_pace,
        pace=pace,
        elapsed=elapsed,
        durability=solution,
    )
    plan.sections = [
        _stretch(plan, i, _LABELS[label], start, end)
        for i, (start, end, label) in enumerate(detect_sections(course), start=1)
    ]
    plan.legs, plan.ignored_aid_stations_km = _legs(
        plan, aid_stations_km, aid_station_names or []
    )
    return plan


def _stretch(plan: RacePlan, index: int, kind: str, start: int, end: int,
             label: str = "") -> Stretch:
    course = plan.course
    gain, loss = course.elevation_gain(start, end + 1)
    return Stretch(
        index=index,
        kind=kind,
        start_m=float(course.distance[start]),
        end_m=float(course.distance[end]),
        elevation_gain_m=gain,
        elevation_loss_m=loss,
        start_elevation_m=float(course.elevation_smooth[start]),
        end_elevation_m=float(course.elevation_smooth[end]),
        start_time_s=float(plan.elapsed[start]),
        end_time_s=float(plan.elapsed[end]),
        label=label,
    )


def _legs(
    plan: RacePlan, aid_stations_km: Sequence[float], names: Sequence[str]
) -> Tuple[List[Stretch], List[float]]:
    course = plan.course
    total = course.total_m
    stations: List[Tuple[int, str]] = []
    ignored: List[float] = []
    named = list(zip(aid_stations_km, list(names) + [""] * len(aid_stations_km)))
    for km, name in sorted(named, key=lambda item: item[0]):
        metres = float(km) * 1000
        if not np.isfinite(metres) or metres <= 0 or metres >= total:
            ignored.append(float(km))
            continue
        node = int(np.argmin(np.abs(course.distance - metres)))
        if node == 0 or node == len(course.distance) - 1:
            ignored.append(float(km))
            continue
        if stations and stations[-1][0] == node:
            continue  # the same station entered twice
        stations.append((node, (name or "").strip()))

    boundaries = [0] + [node for node, _ in stations] + [len(course.distance) - 1]
    labels = [label for _, label in stations] + [""]
    legs = [
        _stretch(plan, i + 1, "leg", boundaries[i], boundaries[i + 1], labels[i])
        for i in range(len(boundaries) - 1)
    ]
    return legs, ignored


# --- Sections -------------------------------------------------------------

def min_section_length_m(total_m: float) -> float:
    """Shortest section worth its own row — longer on a longer course.

    A 300 m ramp matters on a 10 km; on a 170 km ultra it is noise, and a plan with
    two hundred rows is not a plan.
    """
    return float(np.clip(total_m / 60, 300.0, 2500.0))


def detect_sections(course: Course) -> List[Tuple[int, int, int]]:
    """Climb / descent / flat sections as ``(start_node, end_node, label)``.

    ``label`` is ``1`` (climb), ``-1`` (descent) or ``0`` (flat). The algorithm:

    1. Label every grid interval by its heavily smoothed gradient against
       ±``FLAT_GRADE``, and run-length encode the labels.
    2. Repeat until stable:

       * merge neighbours that share a label;
       * absorb the **shortest** section under the minimum length into a neighbour
         — the one that makes both sides collapse together when they agree (a short
         flat between two climbs), otherwise the longer one;
       * once no section is too short, **relabel** by what each section actually
         does end to end: a "climb" gaining less than ``MIN_SECTION_ELEVATION_M``
         is flat; one that nets a loss is a descent. A flat is first split at the
         summits and valleys inside it (a zigzag filter, see
         :func:`_turning_points`), and any piece that steadily gains or loses
         ``GENTLE_ELEVATION_M`` at ``GENTLE_GRADE`` or more becomes a climb or
         descent — so rolling terrain under the 3 % bar still shows its hills.

    Every merge removes a section and every relabel moves a section towards the
    label its own endpoints imply, so this terminates; the iteration cap is a
    guard, not a tuning knob.
    """
    labels = np.where(
        course.section_grade > FLAT_GRADE, 1,
        np.where(course.section_grade < -FLAT_GRADE, -1, 0),
    )
    # Intervals [i, i+1] → node ranges. `segments` holds [start_node, end_node, label].
    segments: List[List[int]] = []
    for i, label in enumerate(labels):
        if segments and segments[-1][2] == label:
            segments[-1][1] = i + 1
        else:
            segments.append([i, i + 1, int(label)])

    distance, elevation = course.distance, _section_elevation(course)
    min_len = min_section_length_m(course.total_m)
    turns = _turning_points(elevation, MIN_SECTION_ELEVATION_M)

    def length(seg) -> float:
        return float(distance[seg[1]] - distance[seg[0]])

    def delta(seg) -> float:
        return float(elevation[seg[1]] - elevation[seg[0]])

    for _ in range(10 * len(segments) + 10):
        segments = _merge_same(segments)
        if len(segments) == 1:
            break

        short = [i for i, seg in enumerate(segments) if length(seg) < min_len]
        if short:
            i = min(short, key=lambda k: length(segments[k]))
            _absorb(segments, i, length)
            continue

        changed = False
        relabelled_segments: List[List[int]] = []
        for seg in segments:
            # A flat run is split at the summits and valleys inside it first:
            # "flat" over 40 km of rolling hills nets out near zero, while each
            # gentle climb inside it, taken alone, clearly is one.
            pieces = _split_at(seg, turns) if seg[2] == 0 else [seg]
            for piece in pieces:
                label = _relabel(piece[2], delta(piece), length(piece))
                if piece[2] == 0 and length(piece) < min_len:
                    label = 0  # too short to stand alone; it would only be re-absorbed
                changed |= label != piece[2]
                relabelled_segments.append([piece[0], piece[1], label])
        segments = relabelled_segments
        if not changed:
            break

    return [(seg[0], seg[1], seg[2]) for seg in _merge_same(segments)]


def _section_elevation(course: Course) -> np.ndarray:
    """The heavily smoothed elevation, rebuilt from its gradient."""
    ds = np.diff(course.distance)
    return course.elevation_smooth[0] + np.concatenate(
        [[0.0], np.cumsum(course.section_grade * ds / 1000)]
    )


def _turning_points(elevation: np.ndarray, hysteresis: float) -> List[int]:
    """Summits and valleys: extremes followed by a reversal of at least ``hysteresis``.

    The classic zigzag filter. Wobbles smaller than the hysteresis never register
    as a reversal, so this finds the landscape's real tops and bottoms.
    """
    turns: List[int] = []
    direction = 0  # 0 until the first reversal, then +1 rising / -1 falling
    high = low = 0
    for i in range(1, len(elevation)):
        if direction == 0:
            high = i if elevation[i] > elevation[high] else high
            low = i if elevation[i] < elevation[low] else low
            if elevation[high] - elevation[i] >= hysteresis:
                turns += [high] if high > 0 else []
                direction, low = -1, i
            elif elevation[i] - elevation[low] >= hysteresis:
                turns += [low] if low > 0 else []
                direction, high = 1, i
        elif direction == 1:
            if elevation[i] > elevation[high]:
                high = i
            elif elevation[high] - elevation[i] >= hysteresis:
                turns.append(high)
                direction, low = -1, i
        else:
            if elevation[i] < elevation[low]:
                low = i
            elif elevation[i] - elevation[low] >= hysteresis:
                turns.append(low)
                direction, high = 1, i
    return turns


def _split_at(seg: List[int], turns: List[int]) -> List[List[int]]:
    cuts = [t for t in turns if seg[0] < t < seg[1]]
    bounds = [seg[0]] + cuts + [seg[1]]
    return [[bounds[k], bounds[k + 1], seg[2]] for k in range(len(bounds) - 1)]


def _merge_same(segments: List[List[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for seg in segments:
        if out and out[-1][2] == seg[2]:
            out[-1][1] = seg[1]
        else:
            out.append(list(seg))
    return out


def _absorb(segments: List[List[int]], i: int, length) -> None:
    """Fold ``segments[i]`` into a neighbour, in place."""
    left = segments[i - 1] if i > 0 else None
    right = segments[i + 1] if i + 1 < len(segments) else None
    if left is not None and right is not None:
        if left[2] == right[2]:
            target = left
        else:
            target = left if length(left) >= length(right) else right
    else:
        target = left if left is not None else right
    if target is left:
        left[1] = segments[i][1]
    else:
        right[0] = segments[i][0]
    del segments[i]


def _relabel(label: int, delta_m: float, length_m: float) -> int:
    grade = delta_m / length_m * 1000 if length_m > 0 else 0.0
    if label != 0:
        if abs(delta_m) < MIN_SECTION_ELEVATION_M:
            return 0
        return 1 if delta_m > 0 else -1
    if abs(delta_m) >= GENTLE_ELEVATION_M and abs(grade) >= GENTLE_GRADE:
        return 1 if delta_m > 0 else -1
    return 0
