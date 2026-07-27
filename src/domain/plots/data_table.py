"""The feature table itself — one row per activity, or one per group.

Deliberately one plot type rather than two: the race comparator's "best value per
column" stats table, a recap of the activities a panel selected, and per-window
totals are the same object with different parameters. It doubles as the escape
hatch of a data-science tool — pick your columns, download the CSV, do whatever
you want elsewhere.
"""

from datetime import datetime
from typing import Any, Dict, List

from src.domain.charts.ir import CellFormat, Column, PlotOutput, TableData, empty_output
from src.domain.dataset.metrics import ACTIVITY_METRICS, metric_or_default
from src.domain.dataset.resolved import GROUP_COLUMN, DataLevel, ResolvedPanelData
from src.domain.plots.base import (
    PlotDefinition,
    metric_cell_format,
    metric_label,
    register,
)
from src.domain.spec.params import Choice, ParamSpec, boolean, choice, integer, multichoice
from src.translations import translate

_DEFAULT_COLUMNS = ["distance_km", "elevation_gain_m", "moving_time", "avg_pace"]

PARAMS: List[ParamSpec] = [
    choice("rows", "param.rows", "activity", choices=[
        Choice("activity", "param.rows.activity"),
        Choice("group", "param.rows.group"),
    ], help_key="param.rows.help"),
    multichoice("columns", "param.columns", _DEFAULT_COLUMNS,
                choices_from="activity_metrics"),
    boolean("highlight_best", "param.highlight_best", False,
            help_key="param.highlight_best.help"),
    choice("sort_by", "param.sort_by", "date", choices_from="sortable_columns"),
    boolean("descending", "param.descending", True),
    integer("limit", "param.limit", 0, min=0, max=1000, help_key="param.limit.help"),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    metrics = [
        ACTIVITY_METRICS[key]
        for key in (params.get("columns") or _DEFAULT_COLUMNS)
        if key in ACTIVITY_METRICS
    ]
    if not metrics:
        metrics = [metric_or_default(None)]

    by_group = (params.get("rows") or "activity") == "group"
    highlight = bool(params.get("highlight_best"))

    columns = _label_columns(resolved, lang, by_group)
    for metric in metrics:
        columns.append(Column(
            key=metric.key,
            label=metric_label(metric, lang),
            format=metric_cell_format(metric),
            highlight=_highlight_for(metric, highlight),
        ))

    rows = _group_rows(resolved, metrics) if by_group else _activity_rows(resolved, metrics)
    rows = _sorted(rows, params, metrics)
    limit = int(params.get("limit") or 0)
    if limit:
        rows = rows[:limit]

    return PlotOutput(tables=[TableData(
        title=translate("plot.data_table.title", lang),
        columns=columns,
        rows=rows,
        download_name="activities" if not by_group else "groups",
    )])


def _label_columns(resolved: ResolvedPanelData, lang: str, by_group: bool) -> List[Column]:
    if by_group:
        return [Column(key=GROUP_COLUMN, label=translate("panel.group", lang),
                       format=CellFormat(kind="text"))]
    columns = [
        Column(key="date", label=translate("races.col.date", lang),
               format=CellFormat(kind="date")),
        Column(key="sport_type", label=translate("races.col.sport", lang),
               format=CellFormat(kind="text")),
    ]
    # The group column only earns its place when there is more than one.
    if resolved.has_multiple_groups:
        columns.insert(1, Column(key=GROUP_COLUMN,
                                 label=translate("panel.group", lang),
                                 format=CellFormat(kind="text")))
    return columns


def _highlight_for(metric, enabled: bool):
    if not enabled or metric.higher_is_better is None:
        return None
    return "max" if metric.higher_is_better else "min"


def _activity_rows(resolved: ResolvedPanelData, metrics) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    frame = resolved.features
    for _, record in frame.iterrows():
        single = frame.loc[[record.name]]
        row: Dict[str, Any] = {
            "date": record.get("date"),
            "sport_type": record.get("sport_type"),
            GROUP_COLUMN: record.get(GROUP_COLUMN),
            "_activity_id": int(record.get("activity_id")),
        }
        for metric in metrics:
            # Row-wise for ratios (this activity's own pace), aggregate otherwise —
            # both collapse to the same thing on a single row.
            row[metric.key] = float(metric.values(single).iloc[0]) if metric.is_ratio \
                else metric.aggregate(single)
        rows.append(row)
    return rows


def _group_rows(resolved: ResolvedPanelData, metrics) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in resolved.groups:
        group_frame = resolved.group_features(group)
        if group_frame.empty:
            continue
        row: Dict[str, Any] = {GROUP_COLUMN: group.label, "date": None}
        for metric in metrics:
            row[metric.key] = metric.aggregate(group_frame)
        rows.append(row)
    return rows


def _sorted(rows: List[Dict[str, Any]], params: Dict[str, Any], metrics) -> List[Dict[str, Any]]:
    """Sort by the chosen column, falling back to date; unknown/NaN values last."""
    if not rows:
        return rows
    key = params.get("sort_by") or "date"
    if key not in rows[0]:
        key = "date" if "date" in rows[0] else None
    if key is None:
        return rows
    descending = bool(params.get("descending", True))

    def sortable(row: Dict[str, Any]):
        value = row.get(key)
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number else None  # NaN is not sortable

    ranked = [(sortable(r), r) for r in rows]
    ordered = sorted(
        (v for v in ranked if v[0] is not None),
        key=lambda pair: pair[0],
        reverse=descending,
    )
    # Rows with no value for the sort column always land at the bottom, whichever
    # direction was asked for — reversing them into first place reads as a bug.
    return [r for _, r in ordered] + [r for v, r in ranked if v is None]


register(PlotDefinition(
    key="data_table",
    label_key="plot.data_table.label",
    description_key="plot.data_table.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    category_key="plotcat.tables",
))
