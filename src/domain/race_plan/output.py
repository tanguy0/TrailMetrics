"""A :class:`RacePlan` as chart IR — one block per question the runner asks.

1. *How fast, here?* — the target pace along the whole course over its profile.
2. *What does each climb / descent / flat cost?* — the sections, charted and tabled.
3. *When am I at each aid station?* — the legs, charted and tabled.
4. *How much does durability cost me, and why?* — ``phi`` along the course, split
   into its components, with the model's coefficients and confidence.

Every chart shares the same backdrop: the elevation profile on the left axis, pace
on the right. The profile is the thing the reader navigates by ("the second big
climb"), so it stays put from chart to chart while the pace overlay changes grain.
The elevation goes on the *primary* axis on purpose: Plotly draws an overlaying
axis's traces above the primary's, and the pace line has to sit on top of the
filled profile, not under it.
"""

from typing import Dict, List, Optional

import numpy as np

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    Badge,
    Band,
    CellFormat,
    ChartData,
    Column,
    Marker,
    PlotOutput,
    TableData,
    Trace,
    TraceKind,
)
from src.domain.durability.config import (
    COMPONENTS,
    DOWNHILL,
    DURATION,
    PRE_RACE_LOAD,
    SEVERE_INTENSITY,
    THERMAL,
)
from src.domain.durability.personalization import AthleteDurabilityModel
from src.domain.durability.report import durability_notes, durability_table
from src.domain.gap import theme
from src.domain.plotting_common import fmt_hms, fmt_pace
from src.domain.race_plan.planner import CLIMB, DESCENT, FLAT, STEP_M, RacePlan, Stretch
from src.translations import translate

ELEVATION_COLOR = theme.BALANCED_RUNNER
PACE_COLOR = theme.PRIMARY
SECTION_COLORS = {CLIMB: theme.TERRACOTTA, DESCENT: theme.CHART_YOU_4, FLAT: theme.MOSS}
LEG_COLORS = [theme.SUNRISE, theme.MOSS]
# charts.md § v1.2 — Plan de course. A section's pace bar reads its gap to the
# plan's average pace: slower than +3 % terra, faster than −3 % moss, forest in
# between. Bars rise from the slow end of the (reversed) pace axis.
_PACE_TOLERANCE = 0.03
_PACE_BAR_FLOOR = 1.08
_PACE_BAR_OPACITY = 0.6  # light enough for the course profile to read through

# Points drawn on the detailed chart. The plan itself runs on a 10 m grid; a
# browser does not need 17k points to draw a 170 km line.
MAX_CHART_POINTS = 2500
# The detailed pace line is a rolling mean over this distance, for display only.
DISPLAY_SMOOTHING_M = 200.0
# Past this many, a badge per section is a smear rather than a label.
MAX_BADGES = 40


DURABILITY_COLOR = theme.TERRACOTTA
COMPONENT_COLORS = {
    DURATION: theme.SUNRISE,
    SEVERE_INTENSITY: theme.DANGER,
    DOWNHILL: theme.CHART_YOU_4,
    THERMAL: theme.TERRACOTTA,
    PRE_RACE_LOAD: theme.KILIAN,
}


def build_outputs(plan: RacePlan, lang: str, start_clock_s: Optional[float] = None,
                  durability: Optional[AthleteDurabilityModel] = None
                  ) -> Dict[str, PlotOutput]:
    outputs = {
        "profile": _profile_output(plan, lang),
        "sections": _sections_output(plan, lang),
        "aid_stations": _legs_output(plan, lang, start_clock_s),
    }
    if plan.durability is not None and durability is not None:
        outputs["durability"] = _durability_output(plan, durability, lang)
    return outputs


def summary(plan: RacePlan, durability: Optional[AthleteDurabilityModel] = None
            ) -> Dict[str, object]:
    gain, loss = plan.course.elevation_gain()
    out: Dict[str, object] = {
        "distance_m": plan.course.total_m,
        "elevation_gain_m": gain,
        "elevation_loss_m": loss,
        "target_time_s": plan.target_time_s,
        "gap_pace_s_per_km": plan.gap_pace_s_per_km,
        "average_pace_s_per_km": plan.average_pace_s_per_km,
        "section_count": len(plan.sections),
        "aid_station_count": max(len(plan.legs) - 1, 0),
    }
    solution = plan.durability
    if solution is not None and durability is not None:
        profile = plan.gap_pace_profile
        out.update({
            "durability_enabled": solution.fallback != "disabled",
            "durability_multiplier_finish": float(solution.profile.multiplier[-1]),
            "gap_pace_finish_s_per_km": float(profile[-1]),
            "durability_confidence": durability.confidence,
            "durability_status": durability.coefficients.status,
            "reference_source": solution.reference.source,
        })
    return out


# --- Shared backdrop ------------------------------------------------------

def _chart_indices(n: int) -> np.ndarray:
    stride = max(1, int(np.ceil(n / MAX_CHART_POINTS)))
    idx = np.arange(0, n, stride)
    return idx if idx[-1] == n - 1 else np.append(idx, n - 1)


def _elevation_trace(plan: RacePlan, lang: str) -> Trace:
    course = plan.course
    idx = _chart_indices(len(course.distance))
    return Trace(
        name=translate("race_plan.series.elevation", lang),
        x=(course.distance[idx] / 1000).round(3).tolist(),
        y=course.elevation_smooth[idx].round(1).tolist(),
        kind=TraceKind.LINE,
        color=ELEVATION_COLOR,
        # The course profile is the backdrop of every plan chart: flat, drawn
        # first, never "the area" (charts.md § v1.2).
        background=True,
        hover_template="%{y:.0f} m<extra>%{fullData.name}</extra>",
    )


def _elevation_axis(plan: RacePlan, lang: str) -> Axis:
    low = float(plan.course.elevation_smooth.min())
    high = float(plan.course.elevation_smooth.max())
    span = max(high - low, 20.0)
    # Headroom above for the badge row; a floor below so the fill has a base
    # without the axis starting at sea level.
    return Axis(
        title=translate("race_plan.axis.elevation", lang),
        kind=AxisKind.LINEAR,
        tick_format=",.0f",
        range=[low - 0.15 * span, high + 0.3 * span],
        color=ELEVATION_COLOR,
    )


def _pace_axis(lang: str) -> Axis:
    return Axis(
        title=translate("race_plan.axis.pace", lang),
        kind=AxisKind.DURATION,
        reversed=True,
        tick_format="%M:%S",
        color=PACE_COLOR,
    )


def _distance_axis(lang: str) -> Axis:
    return Axis(title=translate("race_plan.axis.distance", lang),
                kind=AxisKind.LINEAR, tick_format=",.1f")


def _step_trace(stretches: List[Stretch], name: str, lang: str) -> Trace:
    """Average pace per stretch as a staircase on the pace axis."""
    x = [s.start_m / 1000 for s in stretches] + [stretches[-1].end_m / 1000]
    y = [s.pace_s_per_km for s in stretches] + [stretches[-1].pace_s_per_km]
    return Trace(
        name=name,
        x=[round(v, 3) for v in x],
        y=[round(v, 1) for v in y],
        kind=TraceKind.STEP,
        color=PACE_COLOR,
        axis="y2",
        width=2.0,
        hover_text=[fmt_pace(v) for v in y],
        hover_template="%{customdata}<extra>%{fullData.name}</extra>",
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean with edge-shrinking windows (no zero padding)."""
    if window <= 1:
        return values
    kernel = np.ones(window)
    return np.convolve(values, kernel, mode="same") / np.convolve(
        np.ones_like(values), kernel, mode="same"
    )


def _elevation_at(plan: RacePlan, metres: float) -> float:
    return float(np.interp(metres, plan.course.distance, plan.course.elevation_smooth))


# --- 1. Detailed pace profile ---------------------------------------------

def _profile_output(plan: RacePlan, lang: str) -> PlotOutput:
    course = plan.course
    # Pace is per interval; chart it at interval midpoints, averaged over a short
    # window (and at least the chart stride), so the line shows the terrain rather
    # than GPS elevation jitter, and a downsampled point is the mean of what it
    # hides rather than one arbitrary sample of it.
    mids = (course.distance[:-1] + course.distance[1:]) / 2
    idx = _chart_indices(len(mids))
    stride = max(1, int(np.ceil(len(mids) / MAX_CHART_POINTS)))
    window = max(stride, int(round(DISPLAY_SMOOTHING_M / STEP_M)))
    pace = _rolling_mean(plan.pace, window)[idx]
    elapsed = np.interp(mids[idx], course.distance, plan.elapsed)

    pace_trace = Trace(
        name=translate("race_plan.series.target_pace", lang),
        x=(mids[idx] / 1000).round(3).tolist(),
        y=pace.round(1).tolist(),
        kind=TraceKind.LINE,
        color=PACE_COLOR,
        axis="y2",
        width=2.0,
        hover_text=[
            f"{fmt_pace(p)} · {translate('race_plan.hover.elapsed', lang)} {fmt_hms(t)}"
            for p, t in zip(pace, elapsed)
        ],
        hover_template="%{customdata}<extra>%{fullData.name}</extra>",
    )
    if plan.durability is not None:
        # Effort-equivalent GAP pace drifts slower as durability costs accumulate.
        gap_x = (mids[idx] / 1000).round(3).tolist()
        gap_y = plan.gap_pace_profile[idx].round(1)
        gap_name = translate("race_plan.series.gap_pace_durability", lang)
    else:
        gap_x = [0.0, round(course.total_m / 1000, 3)]
        gap_y = np.array([round(plan.gap_pace_s_per_km, 1)] * 2)
        gap_name = translate("race_plan.series.gap_pace", lang)
    gap_trace = Trace(
        name=gap_name,
        x=gap_x,
        y=gap_y.tolist(),
        kind=TraceKind.LINE,
        color=theme.KILIAN,
        axis="y2",
        dash="--",
        width=1.5,
        hover_text=[fmt_pace(v) for v in gap_y],
        hover_template="%{customdata}<extra>%{fullData.name}</extra>",
    )
    chart = ChartData(
        family="function",
        title=translate("race_plan.chart.profile", lang),
        x_axis=_distance_axis(lang),
        y_axis=_elevation_axis(plan, lang),
        y2_axis=_pace_axis(lang),
        traces=[_elevation_trace(plan, lang), pace_trace, gap_trace],
        height=480,
        caption=translate("race_plan.caption.profile", lang).format(
            gap=fmt_pace(plan.gap_pace_s_per_km),
            avg=fmt_pace(plan.average_pace_s_per_km),
        ),
    )
    return PlotOutput(charts=[chart])


# --- 2. Sections ----------------------------------------------------------

def _kind_label(kind: str, lang: str) -> str:
    return translate(f"race_plan.section.{kind}", lang)


def _sections_output(plan: RacePlan, lang: str) -> PlotOutput:
    sections = plan.sections
    traces = [_elevation_trace(plan, lang)]

    # One marker series per kind at each section's midpoint: the legend for the
    # coloured bands, and a hover that says what the section is.
    for kind in (CLIMB, DESCENT, FLAT):
        members = [s for s in sections if s.kind == kind]
        if not members:
            continue
        mids = [(s.start_m + s.end_m) / 2 for s in members]
        traces.append(Trace(
            name=_kind_label(kind, lang),
            x=[round(m / 1000, 3) for m in mids],
            y=[round(_elevation_at(plan, m), 1) for m in mids],
            kind=TraceKind.SCATTER,
            color=SECTION_COLORS[kind],
            marker_size=10,
            hover_text=[_section_hover(s, lang) for s in members],
            hover_template="%{customdata}<extra></extra>",
        ))
    traces.append(_pace_bars(plan, sections, translate("race_plan.series.section_pace", lang)))

    chart = ChartData(
        family="function",
        title=translate("race_plan.chart.sections", lang),
        x_axis=_distance_axis(lang),
        y_axis=_elevation_axis(plan, lang),
        y2_axis=_pace_axis(lang),
        traces=traces,
        # Section limits as thin rules; the hovered section tints in the browser.
        markers=[
            Marker(kind="boundary", x=round(s.start_m / 1000, 3))
            for s in sections[1:]
        ],
        badges=[
            Badge(x=round((s.start_m + s.end_m) / 2000, 3), text=str(s.index),
                  color=SECTION_COLORS[s.kind])
            for s in sections
        ] if len(sections) <= MAX_BADGES else [],
        height=440,
        caption=translate("race_plan.caption.sections", lang),
    )

    table = TableData(
        title=translate("race_plan.table.sections", lang),
        columns=[
            Column("index", "#", CellFormat("integer")),
            Column("type", translate("race_plan.col.type", lang)),
            Column("start_km", translate("race_plan.col.start_km", lang), CellFormat("number", 1)),
            Column("end_km", translate("race_plan.col.end_km", lang), CellFormat("number", 1)),
            Column("distance_km", translate("race_plan.col.distance", lang),
                   CellFormat("number", 2, "km")),
            Column("gain_m", "D+", CellFormat("number", 0, "m")),
            Column("loss_m", "D−", CellFormat("number", 0, "m")),
            Column("grade_pct", translate("race_plan.col.grade", lang), CellFormat("percent", 1)),
            Column("pace", translate("race_plan.col.pace", lang), CellFormat("pace")),
            Column("duration", translate("race_plan.col.duration", lang), CellFormat("duration")),
            Column("elapsed", translate("race_plan.col.elapsed_end", lang), CellFormat("duration")),
        ],
        rows=[
            {
                "index": s.index,
                "type": _kind_label(s.kind, lang),
                "start_km": s.start_m / 1000,
                "end_km": s.end_m / 1000,
                "distance_km": s.distance_m / 1000,
                "gain_m": s.elevation_gain_m,
                "loss_m": s.elevation_loss_m,
                "grade_pct": s.average_grade_pct,
                "pace": s.pace_s_per_km,
                "duration": s.duration_s,
                "elapsed": s.end_time_s,
            }
            for s in sections
        ],
        download_name="race_plan_sections",
    )
    return PlotOutput(charts=[chart], tables=[table])


def _pace_bars(plan: RacePlan, sections: List[Stretch], name: str) -> Trace:
    """One bar per section, as wide as the section, coloured by its gap to the
    plan's average pace."""
    average = plan.average_pace_s_per_km
    paces = [s.pace_s_per_km for s in sections]

    def color(pace: float) -> str:
        gap = pace / average - 1 if average else 0.0
        if gap > _PACE_TOLERANCE:
            return theme.TERRA
        if gap < -_PACE_TOLERANCE:
            return theme.MOSS
        return theme.FOREST

    return Trace(
        name=name,
        x=[round((s.start_m + s.end_m) / 2000, 3) for s in sections],
        y=[round(p, 1) for p in paces],
        kind=TraceKind.BAR,
        color=PACE_COLOR,
        axis="y2",
        opacity=_PACE_BAR_OPACITY,
        point_colors=[color(p) for p in paces],
        point_widths=[round(s.distance_m / 1000, 3) for s in sections],
        bar_base=round(max(paces) * _PACE_BAR_FLOOR, 1),
        hover_text=[fmt_pace(p) for p in paces],
        hover_template="%{customdata}<extra>%{fullData.name}</extra>",
    )


def _section_hover(s: Stretch, lang: str) -> str:
    return (
        f"<b>{s.index}. {_kind_label(s.kind, lang)}</b><br>"
        f"km {s.start_m / 1000:.1f} → {s.end_m / 1000:.1f} · "
        f"+{s.elevation_gain_m:.0f} / −{s.elevation_loss_m:.0f} m · "
        f"{s.average_grade_pct:+.1f} %<br>"
        f"{fmt_pace(s.pace_s_per_km)} · {fmt_hms(s.duration_s)}"
    )


# --- 3. Aid-station legs ----------------------------------------------------

def _station_name(leg: Stretch, lang: str, is_finish: bool) -> str:
    if is_finish:
        return translate("race_plan.finish", lang)
    return leg.label or translate("race_plan.aid_station_n", lang).format(n=leg.index)


def _clock(elapsed_s: float, start_clock_s: Optional[float], lang: str) -> Optional[str]:
    if start_clock_s is None:
        return None
    total = int(round(start_clock_s + elapsed_s))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    text = f"{hours:02d}:{rest // 60:02d}"
    return f"{text} {translate('race_plan.next_day', lang).format(n=days)}" if days else text


def _legs_output(plan: RacePlan, lang: str, start_clock_s: Optional[float]) -> PlotOutput:
    legs = plan.legs
    names = [_station_name(leg, lang, i == len(legs) - 1) for i, leg in enumerate(legs)]
    arrivals = [leg.end_m for leg in legs]

    chart = ChartData(
        family="function",
        title=translate("race_plan.chart.aid_stations", lang),
        x_axis=_distance_axis(lang),
        y_axis=_elevation_axis(plan, lang),
        y2_axis=_pace_axis(lang),
        traces=[
            _elevation_trace(plan, lang),
            _step_trace(legs, translate("race_plan.series.leg_pace", lang), lang),
        ],
        # Aid stations drawn like a race on a calendar: a terra dot on the x-axis,
        # named; the arrival time stays in the badge row above.
        markers=[
            Marker(kind="aid", x=round(m / 1000, 3), label=name)
            for name, m in zip(names, arrivals)
        ],
        bands=[
            Band(x0=round(leg.start_m / 1000, 3), x1=round(leg.end_m / 1000, 3),
                 color=LEG_COLORS[i % 2], opacity=0.12)
            for i, leg in enumerate(legs)
        ],
        badges=[
            Badge(x=round(leg.end_m / 1000, 3), text=fmt_hms(leg.end_time_s),
                  color=theme.TERRACOTTA, fill=theme.DANGER_TINT, short=str(i + 1))
            for i, leg in enumerate(legs)
        ] if len(legs) <= MAX_BADGES else [],
        height=440,
        caption=translate("race_plan.caption.aid_stations", lang),
    )

    columns = [
        Column("index", "#", CellFormat("integer")),
        Column("station", translate("race_plan.col.station", lang)),
        Column("km", translate("race_plan.col.km", lang), CellFormat("number", 1)),
        Column("distance_km", translate("race_plan.col.leg_distance", lang),
               CellFormat("number", 2, "km")),
        Column("gain_m", "D+", CellFormat("number", 0, "m")),
        Column("loss_m", "D−", CellFormat("number", 0, "m")),
        Column("pace", translate("race_plan.col.pace", lang), CellFormat("pace")),
        Column("duration", translate("race_plan.col.leg_time", lang), CellFormat("duration")),
        Column("elapsed", translate("race_plan.col.arrival", lang), CellFormat("duration")),
    ]
    if start_clock_s is not None:
        columns.append(Column("clock", translate("race_plan.col.clock", lang)))

    rows = []
    for i, (leg, name) in enumerate(zip(legs, names)):
        row = {
            "index": i + 1,
            "station": name,
            "km": leg.end_m / 1000,
            "distance_km": leg.distance_m / 1000,
            "gain_m": leg.elevation_gain_m,
            "loss_m": leg.elevation_loss_m,
            "pace": leg.pace_s_per_km,
            "duration": leg.duration_s,
            "elapsed": leg.end_time_s,
        }
        if start_clock_s is not None:
            row["clock"] = _clock(leg.end_time_s, start_clock_s, lang)
        rows.append(row)

    notes = []
    if plan.ignored_aid_stations_km:
        notes.append(translate("race_plan.note.ignored_stations", lang).format(
            km=", ".join(f"{km:g}" for km in plan.ignored_aid_stations_km),
            total=f"{plan.course.total_m / 1000:.1f}",
        ))
    table = TableData(
        title=translate("race_plan.table.aid_stations", lang),
        columns=columns,
        rows=rows,
        download_name="race_plan_aid_stations",
    )
    return PlotOutput(charts=[chart], tables=[table], notes=notes)


# --- 4. Durability ----------------------------------------------------------

def _durability_output(plan: RacePlan, model: AthleteDurabilityModel, lang: str) -> PlotOutput:
    """``phi`` along the course over the profile, its components stacked beneath.

    Components are additive in log-cost, so they are drawn as log-cost × 100 —
    which reads as "percent" for the few-percent values a race produces — and the
    total line is the exact ``(phi − 1) × 100``.
    """
    solution = plan.durability
    profile = solution.profile
    course = plan.course
    idx = _chart_indices(len(course.distance))
    x = (course.distance[idx] / 1000).round(3).tolist()
    elapsed = plan.elapsed[idx]

    traces = [_elevation_trace(plan, lang)]
    names = [name for name in COMPONENTS + (PRE_RACE_LOAD,) if name in profile.components]
    for name in names:
        values = profile.components[name][idx] * 100
        if not np.any(values > 1e-6):
            continue
        traces.append(Trace(
            name=translate(f"durability.component.{name}", lang),
            x=x,
            y=values.round(3).tolist(),
            kind=TraceKind.AREA,
            color=COMPONENT_COLORS[name],
            axis="y2",
            stack_group="components",
            opacity=0.55,
            hover_template="%{y:.2f} %<extra>%{fullData.name}</extra>",
        ))
    total = (profile.multiplier[idx] - 1) * 100
    traces.append(Trace(
        name=translate("race_plan.series.durability_total", lang),
        x=x,
        y=total.round(3).tolist(),
        kind=TraceKind.LINE,
        color=DURABILITY_COLOR,
        axis="y2",
        width=2.0,
        hover_text=[
            f"+{v:.1f} % · {translate('race_plan.hover.elapsed', lang)} {fmt_hms(t)}"
            for v, t in zip(total, elapsed)
        ],
        hover_template="%{customdata}<extra>%{fullData.name}</extra>",
    ))
    chart = ChartData(
        title=translate("race_plan.chart.durability", lang),
        x_axis=_distance_axis(lang),
        y_axis=_elevation_axis(plan, lang),
        y2_axis=Axis(
            title=translate("race_plan.axis.durability", lang),
            kind=AxisKind.LINEAR,
            tick_format=",.1f",
            suffix=" %",
            color=DURABILITY_COLOR,
        ),
        traces=traces,
        height=420,
        caption=translate("race_plan.caption.durability", lang).format(
            start=fmt_pace(plan.gap_pace_s_per_km),
            finish=fmt_pace(float(plan.gap_pace_profile[-1])),
            pct=f"{(profile.multiplier[-1] - 1) * 100:.1f}",
        ),
    )
    return PlotOutput(
        charts=[chart],
        tables=[durability_table(model, profile, lang)],
        notes=durability_notes(model, lang, solution=solution),
    )

