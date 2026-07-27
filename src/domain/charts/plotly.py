"""The one renderer: chart IR → styled Plotly figure.

Every figure in the app comes through here, so the Trail / Earthy look, the
palette cycle, duration-axis handling and hover styling are defined exactly once.
Plot definitions never touch Plotly — they describe data (see
:mod:`src.domain.charts.ir`) and get all of this for free.

This replaces the per-topic ``plotting.py`` modules the app used to carry, where
each analysis re-implemented its own axis and legend styling.
"""

from typing import Any, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace, TraceKind
from src.domain.plotting_common import (
    CURVE_PALETTE,
    DASH_BY_LINESTYLE,
    base_figure,
    durations_to_datetimes,
    rgba,
)

# Opacity of the ±band ribbon drawn around a line (GAP ±1σ).
_BAND_ALPHA = 0.16

# Plotly line shape per trace kind; only STEP differs from a plain line.
_LINE_SHAPE = {TraceKind.STEP: "hv"}


def render_chart(chart: ChartData) -> go.Figure:
    """Draw one :class:`ChartData` as a themed, interactive Plotly figure."""
    fig = base_figure(
        title=chart.title,
        x_title=_axis_title(chart.x_axis),
        y_title=_axis_title(chart.y_axis),
        height=chart.height,
    )

    for index, trace in enumerate(chart.traces):
        color = trace.color or CURVE_PALETTE[index % len(CURVE_PALETTE)]
        _add_band(fig, trace, chart, color)
        _add_trace(fig, trace, chart, color)

    _apply_axis(fig.update_xaxes, chart.x_axis)
    _apply_axis(fig.update_yaxes, chart.y_axis)
    if chart.hover_mode:
        fig.update_layout(hovermode=chart.hover_mode)
    if any(t.kind is TraceKind.BAR for t in chart.traces):
        # Bars from different series sit side by side unless explicitly stacked.
        stacked = any(t.stack_group for t in chart.traces)
        fig.update_layout(barmode="stack" if stacked else "group")
    return fig


def _axis_title(axis: Axis) -> str:
    return axis.title or ""


def _apply_axis(update, axis: Axis) -> None:
    """Push one IR axis onto a Plotly axis (already themed by ``base_figure``)."""
    kwargs: dict = {}
    if axis.kind is AxisKind.DURATION:
        # Durations ride on a date axis so ticks read as clock times.
        kwargs["type"] = "date"
        kwargs["tickformat"] = axis.tick_format or "%M:%S"
    elif axis.kind is AxisKind.DATE:
        kwargs["type"] = "date"
        if axis.tick_format:
            kwargs["tickformat"] = axis.tick_format
    elif axis.kind is AxisKind.CATEGORY:
        kwargs["type"] = "category"
    elif axis.tick_format:
        kwargs["tickformat"] = axis.tick_format

    if axis.reversed:
        kwargs["autorange"] = "reversed"
    elif axis.range:
        kwargs["range"] = list(axis.range)
    if axis.suffix:
        kwargs["ticksuffix"] = axis.suffix
    if axis.dtick is not None:
        kwargs["dtick"] = axis.dtick
    if kwargs:
        update(**kwargs)


def _encode(values: Sequence[Any], axis: Axis) -> Any:
    """Map IR values onto what Plotly needs for this axis kind."""
    if axis.kind is AxisKind.DURATION:
        return durations_to_datetimes([_float_or_nan(v) for v in values])
    return list(values)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _add_trace(fig: go.Figure, trace: Trace, chart: ChartData, color: str) -> None:
    x = _encode(trace.x, chart.x_axis)
    y = _encode(trace.y, chart.y_axis)

    common = dict(
        x=x,
        y=y,
        name=trace.name,
        legendgroup=trace.legend_group or trace.name,
        showlegend=trace.show_legend,
        opacity=trace.opacity,
    )
    if trace.hover_text is not None:
        common["customdata"] = list(trace.hover_text)
    if trace.hover_template:
        common["hovertemplate"] = trace.hover_template

    if trace.kind is TraceKind.BAR:
        fig.add_trace(go.Bar(marker=dict(color=color), **common))
        return

    line = dict(color=color, width=trace.width)
    dash = DASH_BY_LINESTYLE.get(trace.dash, "solid")
    if dash != "solid":
        line["dash"] = dash
    shape = _LINE_SHAPE.get(trace.kind)
    if shape:
        line["shape"] = shape

    scatter = dict(line=line, **common)
    if trace.kind is TraceKind.SCATTER:
        scatter["mode"] = "markers"
        scatter["marker"] = dict(color=color, size=trace.marker_size)
    else:
        scatter["mode"] = "lines+markers" if trace.markers else "lines"
        if trace.markers:
            scatter["marker"] = dict(color=color, size=trace.marker_size)

    if trace.kind is TraceKind.AREA:
        scatter["fillcolor"] = rgba(color, 0.35 if trace.stack_group else 0.2)
        scatter["stackgroup"] = trace.stack_group or "area"
        # A hairline keeps stacked bands readable without dominating the fill.
        scatter["line"] = dict(color=color, width=0.5)

    fig.add_trace(go.Scatter(**scatter))


def _add_band(fig: go.Figure, trace: Trace, chart: ChartData, color: str) -> None:
    """Draw the translucent ±band ribbon behind a line, if the trace has one."""
    if trace.band_upper is None or trace.band_lower is None:
        return
    upper = [_float_or_nan(v) for v in trace.band_upper]
    lower = [_float_or_nan(v) for v in trace.band_lower]
    if not upper or len(upper) != len(lower):
        return

    x = list(trace.x)
    ring_x = _encode(x + x[::-1], chart.x_axis)
    ring_y = _encode(upper + lower[::-1], chart.y_axis)
    fig.add_trace(
        go.Scatter(
            x=ring_x,
            y=ring_y,
            fill="toself",
            fillcolor=rgba(color, _BAND_ALPHA),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=trace.legend_group or trace.name,
            name=trace.name,
        )
    )


def render_charts(charts: List[ChartData]) -> List[go.Figure]:
    """Convenience: render a plot's whole chart list in order."""
    return [render_chart(c) for c in charts]


def chart_to_dataframe_rows(chart: ChartData) -> List[dict]:
    """Long-format rows (``trace``, ``x``, ``y``) behind a chart, for CSV export.

    Every figure in the app is downloadable this way, so any plot the user builds
    can leave the app as data — the point of a data-science tool.
    """
    rows: List[dict] = []
    for trace in chart.traces:
        for x, y in zip(trace.x, trace.y):
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue
            rows.append({"trace": trace.name, "x": x, "y": y})
    return rows
