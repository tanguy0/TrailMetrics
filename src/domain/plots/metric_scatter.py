"""Any activity metric against any other — one point per activity.

The relationship plot: pace vs gradient, power-to-HR vs distance, elevation vs
duration. This is the plot that most rewards the metric registry — every pair of
registered metrics is available without writing anything, and an optional
least-squares line says whether the eye is seeing a real slope.
"""

from typing import Any, Dict, List

import numpy as np

from src.domain.charts.ir import ChartData, PlotOutput, Trace, TraceKind, empty_output
from src.domain.dataset.metrics import metric_or_default
from src.domain.dataset.resolved import GROUP_COLUMN, DataLevel, ResolvedPanelData
from src.domain.plots.base import (
    PlotDefinition,
    group_color,
    metric_axis,
    metric_label,
    register,
    series_color,
)
from src.domain.spec.params import Choice, ParamSpec, boolean, choice
from src.translations import translate

# Minimum points before a trendline means anything.
_MIN_TREND_POINTS = 3

PARAMS: List[ParamSpec] = [
    choice("x_metric", "param.x_metric", "distance_km", choices_from="activity_metrics"),
    choice("y_metric", "param.y_metric", "avg_pace", choices_from="activity_metrics"),
    choice("color_by", "param.color_by", "group", choices=[
        Choice("group", "param.color_by.group"),
        Choice("sport_type", "param.color_by.sport"),
    ]),
    boolean("trendline", "param.trendline", False, help_key="param.trendline.help"),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    x_metric = metric_or_default(params.get("x_metric"), "distance_km")
    y_metric = metric_or_default(params.get("y_metric"), "avg_pace")
    color_by = params.get("color_by") or "group"

    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    traces: List[Trace] = []
    notes: List[str] = []
    for index, (label, subset) in enumerate(_series(frame, color_by)):
        x = x_metric.values(subset).to_numpy(dtype=float)
        y = y_metric.values(subset).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        x, y = x[valid], y[valid]
        labels = [
            resolved.activity_label(int(aid))
            for aid in subset["activity_id"].to_numpy()[valid]
        ]
        color = (
            group_color(_group_index(resolved, label)) if color_by == "group"
            else series_color(index)
        )
        traces.append(Trace(
            name=label,
            x=x.tolist(),
            y=y.tolist(),
            kind=TraceKind.SCATTER,
            color=color,
            marker_size=8,
            opacity=0.85,
            hover_text=labels,
            hover_template="%{customdata}<extra>%{fullData.name}</extra>",
        ))
        if params.get("trendline"):
            trend = _trendline(x, y, label, color, lang)
            if trend is not None:
                traces.append(trend)
            else:
                notes.append(translate("plot.scatter.trend_unavailable", lang).format(
                    series=label))

    chart = ChartData(
        title=translate("plot.scatter.title", lang).format(
            y=metric_label(y_metric, lang), x=metric_label(x_metric, lang)),
        x_axis=metric_axis(x_metric, lang),
        y_axis=metric_axis(y_metric, lang),
        traces=traces,
    )
    return PlotOutput(charts=[chart], notes=notes)


def _series(frame, color_by: str):
    column = GROUP_COLUMN if color_by == "group" else "sport_type"
    if column not in frame.columns:
        return [(translate("panel.all", "en"), frame)]
    return [(str(key), subset) for key, subset in frame.groupby(column, sort=False)]


def _group_index(resolved: ResolvedPanelData, label: str) -> int:
    for group in resolved.groups:
        if group.label == label:
            return group.index
    return 0


def _trendline(x, y, label: str, color: str, lang: str):
    """Least-squares fit drawn across the observed x range, or ``None``."""
    if x.size < _MIN_TREND_POINTS or np.ptp(x) <= 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    x_line = [float(x.min()), float(x.max())]
    return Trace(
        name=f"{label} · {translate('plot.scatter.trend', lang)}",
        x=x_line,
        y=[slope * v + intercept for v in x_line],
        kind=TraceKind.LINE,
        color=color,
        dash="--",
        width=6.4,
        show_legend=False,
        legend_group=label,
        hover_template="<extra></extra>",
    )


register(PlotDefinition(
    key="metric_scatter",
    label_key="plot.metric_scatter.label",
    description_key="plot.metric_scatter.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    category_key="plotcat.explore",
))
