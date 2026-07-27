"""Personal records: the stepped progression, and the current-best table.

Two plot types over the ``best_*`` feature columns. The progression line holds the
previous record flat until the day a new one is set, then jumps — and the y-axis is
reversed so every improvement moves *up*, which is the only orientation that reads
as progress.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    CellFormat,
    ChartData,
    Column,
    PlotOutput,
    TableData,
    Trace,
    TraceKind,
    empty_output,
)
from src.domain.dataset.features import best_column
from src.domain.dataset.resolved import DataLevel, ResolvedGroup, ResolvedPanelData
from src.domain.plots.base import PlotDefinition, register, series_color
from src.domain.plotting_common import fmt_hms, fmt_pace
from src.domain.progress.models import PR_DISTANCES
from src.domain.progress.records import record_progression
from src.domain.spec.params import Choice, ParamSpec, boolean, choice, multichoice
from src.translations import translate

_ALL_DISTANCES = [label for label, _ in PR_DISTANCES]
_METERS_BY_LABEL = {label: meters for label, meters in PR_DISTANCES}

# Distances that most histories actually contain, so the default isn't a mess of
# empty traces.
_DEFAULT_DISTANCES = ["5 km", "10 km", "Semi", "Marathon"]


# --- 1. Record progression -------------------------------------------------

PROGRESSION_PARAMS: List[ParamSpec] = [
    multichoice("distances", "param.distances", _DEFAULT_DISTANCES,
                choices_from="pr_distances"),
    choice("display", "param.record_display", "pace", choices=[
        Choice("pace", "param.record_display.pace"),
        Choice("time", "param.record_display.time"),
    ], help_key="param.record_display.help"),
    boolean("extend_to_last", "param.extend_to_last", True,
            help_key="param.extend_to_last.help"),
    boolean("split_by_group", "param.split_by_group", False,
            help_key="param.split_by_group.help"),
]


def compute_progression(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    labels = [l for l in (params.get("distances") or _DEFAULT_DISTANCES)
              if l in _METERS_BY_LABEL] or _DEFAULT_DISTANCES
    as_pace = (params.get("display") or "pace") == "pace"
    split = bool(params.get("split_by_group")) and resolved.has_multiple_groups

    series: List[Tuple[str, Any, Optional[ResolvedGroup]]] = (
        [(g.label, resolved.group_features(g), g) for g in resolved.groups]
        if split else [("", frame, None)]
    )

    end_date = None
    if params.get("extend_to_last", True):
        dates = [d for d in frame["date"] if isinstance(d, datetime)]
        end_date = max(dates) if dates else None

    traces: List[Trace] = []
    color_index = 0
    for prefix, subset, _ in series:
        if subset is None or subset.empty:
            continue
        for label in labels:
            trace = _progression_trace(
                subset, label, prefix, as_pace, end_date, color_index, lang
            )
            color_index += 1
            if trace is not None:
                traces.append(trace)

    if not traces:
        return empty_output(translate("plot.records.none", lang))

    y_title = translate(
        "plot.ltp.records.y_pace" if as_pace else "plot.ltp.records.y_time", lang
    )
    chart = ChartData(
        title=translate("plot.ltp.records.title", lang),
        x_axis=Axis(title=translate("plot.x.time", lang), kind=AxisKind.DATE),
        y_axis=Axis(title=y_title, kind=AxisKind.DURATION, reversed=True,
                    tick_format="%M:%S" if as_pace else "%H:%M:%S"),
        traces=traces,
    )
    return PlotOutput(charts=[chart])


def _progression_trace(
    frame, label: str, prefix: str, as_pace: bool,
    end_date: Optional[datetime], color_index: int, lang: str,
) -> Optional[Trace]:
    column = best_column(label)
    if column not in frame.columns:
        return None
    samples = [
        (row["date"], float(row[column]))
        for _, row in frame.iterrows()
        if isinstance(row.get("date"), datetime) and np.isfinite(_as_float(row.get(column)))
    ]
    progression = record_progression(samples)
    if not progression:
        return None

    dates = [d for d, _ in progression]
    times = [t for _, t in progression]
    kilometres = _METERS_BY_LABEL[label] / 1000.0
    paces = [t / kilometres for t in times]

    if end_date is not None and end_date > dates[-1]:
        # Carry the current record flat to the edge of the plot, so the line does
        # not stop short and imply the record lapsed.
        dates = dates + [end_date]
        times = times + [times[-1]]
        paces = paces + [paces[-1]]

    values = paces if as_pace else times
    hover = [
        f"{translate('plot.ltp.records.hover_record', lang)}: {fmt_hms(t)}"
        f" · {translate('plot.ltp.records.hover_pace', lang)}: {fmt_pace(p)}"
        for t, p in zip(times, paces)
    ]
    name = f"{prefix} · {label}" if prefix else label
    return Trace(
        name=name,
        x=dates,
        y=values,
        kind=TraceKind.STEP,
        color=series_color(color_index),
        markers=True,
        marker_size=7,
        hover_text=hover,
        hover_template="%{x|%Y-%m-%d}<br>%{customdata}<extra>%{fullData.name}</extra>",
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# --- 2. Current records table ----------------------------------------------

TABLE_PARAMS: List[ParamSpec] = [
    multichoice("distances", "param.distances", _ALL_DISTANCES,
                choices_from="pr_distances"),
    boolean("per_group", "param.per_group", False, help_key="param.per_group.help"),
]


def compute_table(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    labels = [l for l in (params.get("distances") or _ALL_DISTANCES)
              if l in _METERS_BY_LABEL] or _ALL_DISTANCES
    per_group = bool(params.get("per_group")) and resolved.has_multiple_groups

    columns = [
        Column(key="distance", label=translate("ltp.records.col.distance", lang),
               format=CellFormat(kind="text")),
    ]
    if per_group:
        columns.append(Column(key="group", label=translate("panel.group", lang),
                              format=CellFormat(kind="text")))
    columns += [
        Column(key="record", label=translate("ltp.records.col.record", lang),
               format=CellFormat(kind="duration")),
        Column(key="pace", label=translate("ltp.records.col.pace", lang),
               format=CellFormat(kind="pace")),
        Column(key="date", label=translate("ltp.records.col.date", lang),
               format=CellFormat(kind="date")),
    ]

    scopes = (
        [(g.label, resolved.group_features(g)) for g in resolved.groups]
        if per_group else [("", frame)]
    )
    rows: List[Dict[str, Any]] = []
    for group_label, subset in scopes:
        if subset is None or subset.empty:
            continue
        for label in labels:
            best = _best_effort(subset, label)
            kilometres = _METERS_BY_LABEL[label] / 1000.0
            rows.append({
                "distance": label,
                "group": group_label,
                "record": best[1] if best else None,
                "pace": (best[1] / kilometres) if best else None,
                "date": best[0] if best else None,
            })

    return PlotOutput(tables=[TableData(
        title=translate("plot.records_table.title", lang),
        columns=columns,
        rows=rows,
        download_name="records",
    )])


def _best_effort(frame, label: str) -> Optional[Tuple[datetime, float]]:
    """Fastest ``(date, seconds)`` for one distance across ``frame``."""
    column = best_column(label)
    if column not in frame.columns:
        return None
    best: Optional[Tuple[datetime, float]] = None
    for _, row in frame.iterrows():
        value = _as_float(row.get(column))
        when = row.get("date")
        if not np.isfinite(value) or not isinstance(when, datetime):
            continue
        if best is None or value < best[1]:
            best = (when, value)
    return best


register(PlotDefinition(
    key="pr_progression",
    label_key="plot.pr_progression.label",
    description_key="plot.pr_progression.description",
    level=DataLevel.ACTIVITY,
    compute=compute_progression,
    params=PROGRESSION_PARAMS,
    requires_streams=True,
    category_key="plotcat.records",
))

register(PlotDefinition(
    key="records_table",
    label_key="plot.records_table.label",
    description_key="plot.records_table.description",
    level=DataLevel.ACTIVITY,
    compute=compute_table,
    params=TABLE_PARAMS,
    requires_streams=True,
    category_key="plotcat.records",
))
