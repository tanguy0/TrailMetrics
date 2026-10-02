"""The one renderer: chart IR → styled Plotly figure.

Every figure in the app comes through here, so the TAGG chart chrome, the
palette cycle, duration-axis handling and hover styling are defined exactly once.
Plot definitions never touch Plotly — they describe data (see
:mod:`src.domain.charts.ir`) and get all of this for free.

This replaces the per-topic ``plotting.py`` modules the app used to carry, where
each analysis re-implemented its own axis and legend styling.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from src.domain.charts.families import Plan, finite, has_points, plan as family_plan
from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace, TraceKind
from src.domain.gap import theme
from src.domain.plotting_common import (
    CURVE_PALETTE,
    DASH_BY_LINESTYLE,
    MARGIN,
    axis_style,
    base_figure,
    durations_to_datetimes,
    fmt_hms,
    rgba,
)

# Opacity of the ±band ribbon drawn around a line (GAP ±1σ).
_BAND_ALPHA = 0.16

# A reference series (balanced runner, Kilian, a target) is the reference grey,
# thinner than the athlete's own lines and drawn behind them (charts.md § Séries).
_REF_WIDTH = 1.5


def _is_reference(color: str) -> bool:
    return color.upper() == theme.CHART_REF.upper()


def _default_color(index: int) -> str:
    return CURVE_PALETTE[index % len(CURVE_PALETTE)]


# --- v1.1 (charts.md § Plus de caractère) ----------------------------------
# Every constant below has a twin in web/components/ChartView.tsx.

# Rounded bar tops.
_BAR_RADIUS = 4
# The end-of-line dot (r 4) and its value label, just right of the last point.
_END_MARKER_SIZE = 8
_END_LABEL_SHIFT = 9
_END_LABEL_FONT_SIZE = 11
# Right margin once lines carry an end label, so the label is not clipped.
_END_LABEL_MARGIN_R = 52
# Label above a highlighted bar (the three tallest of a slope histogram).
_BAR_LABEL_FONT_SIZE = 11
# Today's dotted line, and a race's dot on the x-axis.
_MARKER_DASH = "2px,4px"
_RACE_DOT_PX = 4


# Headroom either side of the data, as a share of its span (Plotly's autorange
# pads about the same).
_RANGE_PAD = 0.05
# A backdrop (altitude, profile): line-strong at this opacity, no stroke.
_BACKGROUND_ALPHA = 0.35
# An oscillation's reference level.
_BASELINE_WIDTH = 1
# An axis holding only a backdrop frames its relief, not zero: a bit below the
# low point, more above the high one (the course profile's own rule).
_BACKGROUND_PAD_LOW = 0.15
_BACKGROUND_PAD_HIGH = 0.3
# The hidden axis a backdrop moves to when both visible axes are taken.
_BACKGROUND_AXIS = "y3"


def background_range(values: Sequence[float]) -> Optional[List[float]]:
    """The range an axis of backdrops only takes, or ``None`` with no data."""
    data = finite(values)
    if not data:
        return None
    lo, hi = min(data), max(data)
    span = hi - lo or 1.0
    return [lo - _BACKGROUND_PAD_LOW * span, hi + _BACKGROUND_PAD_HIGH * span]


def area_y_range(chart: ChartData) -> Optional[Tuple[float, float, float]]:
    """The left axis's range when the figure carries an area: ``(lo, hi, data_max)``.

    An area is the sign of a quantity that accumulates (charts.md § v1.2), so it
    starts at zero — on the data's side of it. The range is set explicitly anyway,
    padded like Plotly's own autorange, so the gradient can fade over exactly what
    is visible. An explicit ``y_axis.range`` is kept as given; ``None`` means a
    chart with no data to fit.
    """
    primary = [
        t for t in chart.traces
        if (t.axis != "y2" or chart.y2_axis is None) and not t.background
    ]
    values = [
        v for t in primary
        for v in finite(t.y) + finite(t.band_upper) + finite(t.band_lower)
    ]
    if not values:
        return None
    lo, hi = min(values), max(values)
    if chart.y_axis.range:
        return chart.y_axis.range[0], chart.y_axis.range[1], hi
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = _RANGE_PAD * (hi - lo or 1.0)
    return (lo if lo == 0 else lo - pad), (hi if hi == 0 else hi + pad), max(values)


def resolve_hover_mode(chart: ChartData) -> str:
    """Unified hover (charts.md § v1.1) unless the chart is a scatter.

    "closest" is treated as "auto" too: it was the IR's default before v1.1, and
    outputs cached back then still carry it.
    """
    if chart.hover_mode not in ("auto", "closest"):
        return chart.hover_mode
    plotted = [t for t in chart.traces if has_points(t)]
    if plotted and all(t.kind is TraceKind.SCATTER for t in plotted):
        return "closest"
    return "x unified"


def end_label_text(value: float, axis: Axis) -> str:
    """A line's last value as its end label reads it: ``4:21``, ``68``, ``1.42``."""
    if axis.kind is AxisKind.DURATION:
        return fmt_hms(value)
    magnitude = abs(value)
    decimals = 0 if magnitude >= 10 else 1 if magnitude >= 1 else 2
    return f"{value:,.{decimals}f}" + (axis.suffix or "")


def _last_point(trace: Trace):
    for x, y in zip(reversed(trace.x), reversed(trace.y)):
        if y is not None and not (isinstance(y, float) and np.isnan(y)):
            return x, y
    return None


# Plotly line shape per trace kind; only STEP differs from a plain line.
_LINE_SHAPE = {TraceKind.STEP: "hv"}

# Where the badge row sits, as a share of the plot's height (1 = the very top).
# Just inside the frame rather than above it: outside would fight the title and
# the legend for the same strip of margin.
_BADGE_ROW_Y = 0.98
_BADGE_FONT_SIZE = 9
# Tight: a 30-week window leaves each badge ~20 px of x to sit in.
_BADGE_PADDING = 1
# Pixels a badge needs before its full wording fits rather than its ``short``
# form. This renderer has no width to measure — a figure is responsive and sized
# by whatever embeds it — so it assumes a desktop-width figure, which is what a
# notebook or an export is. The browser twin measures for real.
_MIN_FULL_BADGE_PX = 62
_ASSUMED_FIGURE_PX = 900


def render_chart(chart: ChartData) -> go.Figure:
    """Draw one :class:`ChartData` as a themed, interactive Plotly figure."""
    fig = base_figure(
        title=chart.title,
        x_title=_axis_title(chart.x_axis),
        y_title=_axis_title(chart.y_axis),
        height=chart.height,
    )

    _add_bands(fig, chart)
    colored = [
        (index, trace, trace.color or _default_color(index))
        for index, trace in enumerate(chart.traces)
    ]
    # Backdrops first, then references, so both sit underneath the athlete's
    # lines; `legendrank` keeps the legend in the chart's own order regardless.
    colored.sort(key=lambda item: (not item[1].background, not _is_reference(item[2])))

    decided = family_plan(chart)
    area_range = area_y_range(chart) if decided.area is not None else None
    lonely = _lonely_background(chart)

    for index, trace, color in colored:
        if trace.background:
            _add_background(fig, trace, chart, show_legend=lonely)
            continue
        _add_band(fig, trace, chart, color)
        _add_trace(
            fig, trace, chart, color, rank=index + 1, plan=decided, index=index,
            area=area_range if index == decided.area else None,
        )
        if index in decided.end_labels:
            _add_end_label(fig, trace, chart, color)
    _add_baseline(fig, chart, decided)
    _add_badges(fig, chart)
    _add_markers(fig, chart)
    if decided.end_labels:
        fig.update_layout(margin={**MARGIN, "r": _END_LABEL_MARGIN_R})

    _apply_axis(fig.update_xaxes, chart.x_axis)
    _apply_axis(fig.update_yaxes, chart.y_axis)
    if area_range is not None:
        # Left axis only: `update_yaxes` would reach a right-hand axis too.
        fig.update_layout(yaxis_range=[area_range[0], area_range[1]])
    if chart.y2_axis is not None:
        # Overlaid on the left axis and drawn on the right. No grid of its own is
        # not cosmetic: two sets of gridlines at different intervals produce a mesh
        # that makes both scales harder to read than either alone. No line either —
        # only the x-axis draws one (charts.md).
        secondary = {
            **axis_style(grid=False),
            "showline": False,
            "title": {"text": _axis_title(chart.y2_axis)},
            "overlaying": "y",
            "side": "right",
        }
        # Axis kwargs win: they carry the coloured title when one is set.
        secondary.update(_axis_kwargs(chart.y2_axis))
        fig.update_layout(yaxis2=secondary)
    _frame_backgrounds(fig, chart)
    fig.update_layout(hovermode=resolve_hover_mode(chart))
    if any(t.kind is TraceKind.BAR for t in chart.traces):
        # Bars from different series sit side by side unless explicitly stacked.
        stacked = any(t.stack_group for t in chart.traces)
        fig.update_layout(barmode="stack" if stacked else "group")
        if chart.bargap is not None:
            fig.update_layout(bargap=chart.bargap)
    return fig


def _lonely_background(chart: ChartData) -> bool:
    """A backdrop joins the legend only when it is the figure's only series."""
    return not any(has_points(t) and not t.background for t in chart.traces)


def _add_background(fig: go.Figure, trace: Trace, chart: ChartData, *, show_legend: bool) -> None:
    """A flat line-strong fill to zero, no stroke — never "the area" (charts.md § v1.2)."""
    background = dict(
        x=_encode(trace.x, chart.x_axis),
        y=_encode(trace.y, _y_axis_for(trace, chart)),
        name=trace.name,
        mode="lines",
        line=dict(width=0, color=theme.LINE_STRONG),
        fill="tozeroy",
        fillcolor=rgba(theme.LINE_STRONG, _BACKGROUND_ALPHA),
        showlegend=show_legend and trace.show_legend,
        legendgroup=trace.legend_group or trace.name,
    )
    if trace.hover_text is not None:
        background["customdata"] = list(trace.hover_text)
    if trace.hover_template:
        background["hovertemplate"] = trace.hover_template
    if trace.axis == "y2" and chart.y2_axis is not None:
        background["yaxis"] = "y2"
    elif trace.axis == _BACKGROUND_AXIS:
        background["yaxis"] = _BACKGROUND_AXIS
    fig.add_trace(go.Scatter(**background))


def _frame_backgrounds(fig: go.Figure, chart: ChartData) -> None:
    """Range every axis that only holds backdrops to their relief.

    The hidden third axis is created here; a visible axis that also carries a
    real series, or that has an explicit range, is left alone.
    """
    for name, axis, layout_key in (("y", chart.y_axis, "yaxis"), ("y2", chart.y2_axis, "yaxis2"),
                                   (_BACKGROUND_AXIS, None, "yaxis3")):
        on_axis = [t for t in chart.traces if _axis_name(t, chart) == name]
        if not on_axis or not all(t.background for t in on_axis):
            continue
        if axis is not None and axis.range:
            continue
        framed = background_range([v for t in on_axis for v in t.y])
        if framed is None:
            continue
        if name == _BACKGROUND_AXIS:
            fig.update_layout(yaxis3=dict(overlaying="y", visible=False, range=framed))
        else:
            fig.update_layout(**{f"{layout_key}_range": framed})


def _axis_name(trace: Trace, chart: ChartData) -> str:
    if trace.axis == "y2" and chart.y2_axis is not None:
        return "y2"
    if trace.axis == _BACKGROUND_AXIS:
        return _BACKGROUND_AXIS
    return "y"


def _add_baseline(fig: go.Figure, chart: ChartData, decided: Plan) -> None:
    """An oscillation's reference level, a line-strong rule across the plot."""
    if decided.baseline is None:
        return
    y = _encode([decided.baseline], chart.y_axis)[0]
    fig.add_shape(
        type="line", xref="paper", yref="y", x0=0, x1=1, y0=y, y1=y,
        line=dict(color=theme.LINE_STRONG, width=_BASELINE_WIDTH), layer="below",
    )


def _based_bars(trace: Trace, axis: Axis) -> dict:
    """Bars from ``bar_base`` to each value: Plotly reads a bar's ``y`` as its
    length from ``base``, and on a duration axis that length is in milliseconds."""
    base = float(trace.bar_base)
    scale = 1000.0 if axis.kind is AxisKind.DURATION else 1.0
    lengths = [None if v is None else (float(v) - base) * scale for v in trace.y]
    return dict(y=lengths, base=_encode([base], axis)[0])


def _add_end_label(fig: go.Figure, trace: Trace, chart: ChartData, color: str) -> None:
    """A dot on the line's last point and its value beside it, in its colour."""
    last = _last_point(trace)
    if last is None:
        return
    axis = _y_axis_for(trace, chart)
    x = _encode([last[0]], chart.x_axis)[0]
    y = _encode([last[1]], axis)[0]
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers",
        marker=dict(color=color, size=_END_MARKER_SIZE),
        hoverinfo="skip", showlegend=False, cliponaxis=False,
        legendgroup=trace.legend_group or trace.name,
    ))
    fig.add_annotation(
        x=x, y=y, xref="x", yref="y",
        text=end_label_text(float(last[1]), axis),
        showarrow=False, xanchor="left", xshift=_END_LABEL_SHIFT,
        font=dict(family=theme.FONT_MONO, size=_END_LABEL_FONT_SIZE, color=color),
    )


def _add_markers(fig: go.Figure, chart: ChartData) -> None:
    """Today as a dotted sun line with its label; a race as a terra dot on the axis."""
    for marker in chart.markers:
        x = _encode([marker.x], chart.x_axis)[0]
        if marker.kind == "today":
            fig.add_shape(
                type="line", xref="x", yref="y domain", x0=x, x1=x, y0=0, y1=1,
                line=dict(color=theme.TODAY_MARKER, width=1, dash=_MARKER_DASH),
            )
            fig.add_annotation(
                x=x, xref="x", y=1, yref="y domain", yanchor="bottom",
                text=marker.label, showarrow=False,
                font=dict(family=theme.FONT_MONO, size=_END_LABEL_FONT_SIZE, color=theme.SUN_INK),
            )
        elif marker.kind == "boundary":
            fig.add_shape(
                type="line", xref="x", yref="y domain", x0=x, x1=x, y0=0, y1=1,
                line=dict(color=theme.LINE, width=1), layer="below",
            )
        elif marker.kind in ("race", "aid"):
            fig.add_shape(
                type="circle", xref="x", yref="y domain",
                xsizemode="pixel", ysizemode="pixel", xanchor=x, yanchor=0,
                x0=-_RACE_DOT_PX, x1=_RACE_DOT_PX, y0=0, y1=2 * _RACE_DOT_PX,
                fillcolor=theme.RACE_MARKER, line_width=0,
            )
            fig.add_annotation(
                x=x, xref="x", y=0, yref="y domain", yanchor="bottom", yshift=2 * _RACE_DOT_PX + 2,
                text=marker.label, showarrow=False,
                font=dict(family=theme.FONT_MONO, size=_END_LABEL_FONT_SIZE, color=theme.RACE_MARKER),
            )


def _add_bands(fig: go.Figure, chart: ChartData) -> None:
    """Shade every band across the full height of the plot, behind the traces."""
    for band in chart.bands:
        x0, x1 = _encode([band.x0, band.x1], chart.x_axis)
        fig.add_shape(
            type="rect",
            xref="x", yref="y domain",
            x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=rgba(band.color, band.opacity),
            line_width=0, layer="below",
        )


def _add_badges(fig: go.Figure, chart: ChartData) -> None:
    """Pin the badge row just inside the top of the plot area.

    ``y domain`` coordinates rather than data ones, so the row stays put whatever
    the y-scale is. Keeping it clear of the data is the *chart's* job — see
    :class:`~src.domain.charts.ir.Badge`.
    """
    room = _ASSUMED_FIGURE_PX / max(len(chart.badges), 1)
    for badge in chart.badges:
        x = _encode([badge.x], chart.x_axis)[0]
        text = badge.text
        if badge.short and room < _MIN_FULL_BADGE_PX:
            text = badge.short
        fig.add_annotation(
            x=x, xref="x",
            y=_BADGE_ROW_Y, yref="y domain", yanchor="top",
            text=text, showarrow=False,
            font=dict(family=theme.FONT_MONO, color=badge.color, size=_BADGE_FONT_SIZE),
            bgcolor=badge.fill,
            bordercolor=badge.color, borderwidth=1, borderpad=_BADGE_PADDING,
        )


def _axis_title(axis: Axis) -> str:
    return axis.title or ""


def _apply_axis(update, axis: Axis) -> None:
    """Push one IR axis onto a Plotly axis (already themed by ``base_figure``)."""
    kwargs = _axis_kwargs(axis)
    if kwargs:
        update(**kwargs)


def _axis_kwargs(axis: Axis) -> dict:
    """One IR axis as Plotly axis properties, independent of where they are applied.

    Shared by the left axis (via ``update_yaxes``) and the overlaid right axis (which
    has to be built inside ``layout.yaxis2``, since ``update_yaxes`` would hit both).
    """
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
    if axis.tick_values and axis.tick_labels:
        kwargs.update(tickmode="array", tickvals=list(axis.tick_values),
                      ticktext=list(axis.tick_labels))
    if axis.color:
        kwargs["title"] = dict(
            text=axis.title or "", font=dict(family=theme.FONT_MONO, size=11, color=axis.color)
        )
        kwargs["tickfont"] = dict(family=theme.FONT_MONO, size=11, color=axis.color)
    return kwargs


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


def _add_trace(
    fig: go.Figure, trace: Trace, chart: ChartData, color: str, rank: int,
    *, plan: Plan, index: int, area: Optional[Tuple[float, float, float]] = None,
) -> None:
    main = index == plan.main
    y_axis = _y_axis_for(trace, chart)
    x = _encode(trace.x, chart.x_axis)
    y = _encode(trace.y, y_axis)

    common = dict(
        x=x,
        y=y,
        name=trace.name,
        legendgroup=trace.legend_group or trace.name,
        showlegend=trace.show_legend and index not in plan.hidden_legend,
        legendrank=rank,
        opacity=plan.opacities.get(index, trace.opacity),
    )
    if trace.axis == "y2" and chart.y2_axis is not None:
        common["yaxis"] = "y2"
    if trace.hover_text is not None:
        common["customdata"] = list(trace.hover_text)
    if trace.hover_template:
        common["hovertemplate"] = trace.hover_template
    elif main and _y_axis_for(trace, chart).kind is AxisKind.LINEAR:
        # The main series leads the unified hover, its value bold in sun-ink.
        common["hovertemplate"] = (
            f"%{{fullData.name}} : <b><span style='color:{theme.SUN_INK}'>%{{y}}</span></b>"
            "<extra></extra>"
        )

    if trace.kind is TraceKind.BAR:
        marker: dict = dict(color=trace.point_colors or color, cornerradius=_BAR_RADIUS)
        if trace.point_opacity:
            marker["opacity"] = trace.point_opacity
        bar = dict(marker=marker, **common)
        if trace.point_widths:
            bar["width"] = list(trace.point_widths)
        if trace.bar_base is not None:
            bar.update(_based_bars(trace, _y_axis_for(trace, chart)))
        if trace.point_text:
            bar.update(
                text=trace.point_text, textposition="outside", cliponaxis=False,
                textfont=dict(family=theme.FONT_MONO, size=_BAR_LABEL_FONT_SIZE, color=theme.INK_MUTED),
            )
        fig.add_trace(go.Bar(**bar))
        return

    width = plan.widths.get(index, trace.width)
    if _is_reference(color):
        width = min(width, _REF_WIDTH)
    line = dict(color=color, width=width)
    dash = DASH_BY_LINESTYLE.get(trace.dash, "solid")
    if dash != "solid":
        line["dash"] = dash
    shape = _LINE_SHAPE.get(trace.kind)
    if shape:
        line["shape"] = shape

    scatter = dict(line=line, **common)
    size = plan.marker_sizes.get(index, trace.marker_size)
    if trace.kind is TraceKind.SCATTER:
        scatter["mode"] = "markers"
        scatter["marker"] = dict(color=trace.point_colors or color, size=size)
    else:
        scatter["mode"] = "lines+markers" if trace.markers else "lines"
        if trace.markers:
            scatter["marker"] = dict(color=trace.point_colors or color, size=size)

    if trace.kind is TraceKind.AREA:
        scatter["fillcolor"] = rgba(color, 0.35 if trace.stack_group else 0.2)
        scatter["stackgroup"] = trace.stack_group or "area"
        # A hairline keeps stacked bands readable without dominating the fill.
        scatter["line"] = dict(color=color, width=0.35)
    elif area is not None:
        # The figure's one area (tracking, or a declared one — the fatigue): the
        # colour at AREA_ALPHA_TOP at the data's top, fading to nothing at the
        # bottom of the *visible* axis — not at zero, which may be far below.
        scatter["fill"] = "tozeroy"
        scatter["fillgradient"] = dict(
            type="vertical",
            start=area[0], stop=area[2],
            colorscale=[[0, rgba(color, 0)], [1, rgba(color, theme.AREA_ALPHA_TOP)]],
        )

    fig.add_trace(go.Scatter(**scatter))


def _y_axis_for(trace: Trace, chart: ChartData) -> Axis:
    """The axis a trace is measured against — its values are encoded for that axis."""
    if trace.axis == "y2" and chart.y2_axis is not None:
        return chart.y2_axis
    if trace.axis == _BACKGROUND_AXIS:
        # The hidden backdrop axis is plain numbers (altitude), whatever the left is.
        return Axis(kind=AxisKind.LINEAR)
    return chart.y_axis


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
    ring_y = _encode(upper + lower[::-1], _y_axis_for(trace, chart))
    band = dict(
        x=ring_x,
        y=ring_y,
        fill="toself",
        fillcolor=rgba(color, trace.band_opacity if trace.band_opacity is not None else _BAND_ALPHA),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=trace.legend_group or trace.name,
        name=trace.name,
    )
    if trace.axis == "y2" and chart.y2_axis is not None:
        band["yaxis"] = "y2"
    fig.add_trace(go.Scatter(**band))


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
