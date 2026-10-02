"""Distribution of any activity metric — where does the volume actually sit?

Answers the questions a trend can't: is my 60 km week two long runs or six short
ones, how are my paces spread, did the shape of my training change between two
blocks. One series per data-source group, on shared bins so the groups overlay
honestly.
"""

from typing import Any, Dict, List

import numpy as np

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    ChartData,
    PlotOutput,
    Trace,
    TraceKind,
    empty_output,
)
from src.domain.dataset.metrics import metric_or_default
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.gap import theme
from src.domain.plots.base import (
    PlotDefinition,
    group_color,
    hover_template,
    metric_axis,
    metric_hover_texts,
    metric_label,
    register,
)
from src.domain.spec.params import Choice, ParamSpec, boolean, choice, integer
from src.translations import translate

# A slope histogram reads by the sign of the gradient, not by series order
# (charts.md § v1.1): descents moss, flat forest, climbs terra — the flat band of
# GRADIENT_BANDS. The three most frequent bins stay solid and carry their value;
# the rest fade.
_SLOPE_METRIC = "avg_gradient_pct"
_FLAT_PCT = 3.0
_TOP_BINS = 3
_FADED_OPACITY = 0.55


def _slope_color(center: float) -> str:
    if center < -_FLAT_PCT:
        return theme.SLOPE_DOWN
    if center > _FLAT_PCT:
        return theme.SLOPE_UP
    return theme.SLOPE_FLAT


PARAMS: List[ParamSpec] = [
    choice("metric", "param.metric", "distance_km", choices_from="activity_metrics"),
    integer("bins", "param.bins", 20, min=3, max=100),
    boolean("normalize", "param.normalize", False, help_key="param.normalize.help"),
    choice("chart", "param.chart", "bar", choices=[
        Choice("bar", "param.chart.bar"),
        Choice("line", "param.chart.line"),
    ]),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    metric = metric_or_default(params.get("metric"))
    bin_count = int(params.get("bins") or 20)
    normalize = bool(params.get("normalize"))

    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    all_values = metric.values(frame).replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        return empty_output(translate("plot.metric_unavailable", lang).format(
            metric=metric_label(metric, lang)))

    # Shared edges across groups, otherwise the overlay compares different bins.
    edges = np.histogram_bin_edges(all_values.to_numpy(), bins=bin_count)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    traces: List[Trace] = []
    for group in resolved.groups:
        group_frame = resolved.group_features(group)
        if group_frame.empty:
            continue
        values = metric.values(group_frame).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        counts, _ = np.histogram(values.to_numpy(), bins=edges)
        y = counts.astype(float)
        if normalize and y.sum() > 0:
            y = y / y.sum() * 100.0
        traces.append(Trace(
            name=group.label,
            x=centers,
            y=y.tolist(),
            kind=TraceKind.BAR if (params.get("chart") or "bar") == "bar" else TraceKind.LINE,
            color=group_color(group.index),
            opacity=0.75,
            hover_text=metric_hover_texts(metric, centers),
            hover_template=(
                "%{customdata}<br>%{y:,.1f} "
                + translate("plot.distribution.pct" if normalize else "plot.distribution.count", lang)
                + "<extra>%{fullData.name}</extra>"
            ),
        ))

    y_title = translate(
        "plot.distribution.y_pct" if normalize else "plot.distribution.y_count", lang
    )
    if len(traces) > 1 and all(t.kind is TraceKind.BAR for t in traces):
        # charts.md § v1.2: overlaid bars hide each other, so two groups or more
        # become outlines — one 2 px step per group, unfilled.
        for trace in traces:
            _as_outline(trace, edges.tolist())

    chart = ChartData(
        family="composition",
        title=translate("plot.distribution.title", lang).format(
            metric=metric_label(metric, lang)),
        x_axis=metric_axis(metric, lang),
        y_axis=Axis(title=y_title, kind=AxisKind.LINEAR,
                    tick_format=",.0f" if not normalize else ",.1f"),
        traces=traces,
    )
    if metric.key == _SLOPE_METRIC and len(traces) == 1 and traces[0].kind is TraceKind.BAR:
        # One group only: with several, colour is what tells the groups apart.
        _paint_slopes(traces[0], centers, normalize)
    return PlotOutput(charts=[chart])


# A group's histogram drawn as its outline.
_OUTLINE_WIDTH = 2.0


def _as_outline(trace: Trace, edges: List[float]) -> None:
    """Turn a bar histogram into its step outline over the shared bin edges.

    Steps start at each bin's left edge and hold its count to the next one, so
    the last count is repeated at the right edge to close the shape.
    """
    trace.kind = TraceKind.STEP
    trace.x = list(edges)
    trace.y = list(trace.y) + ([trace.y[-1]] if trace.y else [])
    if trace.hover_text:
        trace.hover_text = list(trace.hover_text) + [trace.hover_text[-1]]
    trace.width = _OUTLINE_WIDTH
    trace.opacity = 1.0


def _paint_slopes(trace: Trace, centers: List[float], normalize: bool) -> None:
    """Colour a slope histogram by gradient sign; highlight its three tallest bins."""
    top = set(sorted(range(len(trace.y)), key=lambda i: trace.y[i], reverse=True)[:_TOP_BINS])
    top = {i for i in top if trace.y[i] > 0}
    trace.opacity = 1.0
    trace.point_colors = [_slope_color(c) for c in centers]
    trace.point_opacity = [1.0 if i in top else _FADED_OPACITY for i in range(len(trace.y))]
    trace.point_text = [
        (f"{trace.y[i]:.1f}" if normalize else f"{trace.y[i]:.0f}") if i in top else ""
        for i in range(len(trace.y))
    ]


register(PlotDefinition(
    key="metric_distribution",
    label_key="plot.metric_distribution.label",
    description_key="plot.metric_distribution.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    category_key="plotcat.explore",
))
