"""Shared Plotly styling so every figure in the app matches the theme.

All charts are Plotly (interactive: clickable legends, hover read-outs, zoom).
This module is the single source of truth for the TAGG chart chrome —
design/tagg/charts.md, setting by setting — plus the line-color cycle and a few
formatting helpers, so the GAP, race-comparison and long-term-progress figures
stay visually coherent. ``web/components/ChartView.tsx`` mirrors it.
"""

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from src.domain.gap import theme

# On-theme line colors, cycled for traces without an explicit color: the
# athlete's five series, then the reference grey. Past five groups a chart should
# become small multiples rather than reach for a sixth hue (charts.md).
CURVE_PALETTE = theme.CURVE_CYCLE

# matplotlib linestyle → Plotly dash, so existing GapCurve.linestyle values port.
# The two dashed styles are the reference patterns of charts.md: "--" is 5-4
# (the balanced runner, targets), ":" is 2-4 (Kilian).
DASH_BY_LINESTYLE = {"-": "solid", "--": "5px,4px", "-.": "dashdot", ":": "2px,4px"}

# Margins of charts.md: the figure has no title of its own (the card around it
# carries one), so the top only has to clear the legend.
MARGIN = dict(l=44, r=16, t=16, b=32)


def axis_style(*, grid: bool) -> dict:
    """Axis chrome: mono 11 px ticks; only the y-axis draws a grid, only x a line."""
    return dict(
        showgrid=grid,
        gridcolor=theme.CHART_GRID,
        gridwidth=1,
        zeroline=False,
        showline=not grid,
        linecolor=theme.LINE,
        ticks="",
        tickfont=dict(family=theme.FONT_MONO, size=11, color=theme.CHART_AXIS),
        title_font=dict(family=theme.FONT_MONO, size=11, color=theme.CHART_AXIS),
        automargin=True,
    )


def base_figure(*, title: str, x_title: str, y_title: str, height: int = 480) -> go.Figure:
    """An empty figure pre-styled with the TAGG chart chrome.

    ``title`` is kept as the figure's *name* (``layout.meta``) for exports, but not
    drawn: on screen the title belongs to the card around the figure.
    """
    fig = go.Figure()
    fig.update_layout(
        meta=dict(title=title),
        paper_bgcolor=theme.BG_CHART,
        plot_bgcolor=theme.BG_CHART,
        font=dict(family=theme.FONT_SANS, color=theme.INK, size=12),
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.02,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            itemwidth=30,
            font=dict(family=theme.FONT_SANS, color=theme.INK_MUTED, size=12),
        ),
        margin=MARGIN,
        height=height,
        hoverlabel=dict(
            bgcolor=theme.BG_SURFACE,
            bordercolor=theme.LINE,
            font=dict(family=theme.FONT_SANS, color=theme.INK, size=12),
        ),
    )
    fig.update_xaxes(title_text=x_title, **axis_style(grid=False))
    fig.update_yaxes(title_text=y_title, **axis_style(grid=True))
    return fig


def rgba(color: str, alpha: float) -> str:
    """``#RRGGBB`` → ``rgba(r,g,b,alpha)`` for translucent fills; passthrough else."""
    c = color.lstrip("#")
    if len(c) == 6:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return color


def durations_to_datetimes(seconds: Sequence[float]) -> np.ndarray:
    """Encode durations (s) as datetimes from the epoch for tidy time-axis ticks.

    Plotted on a ``date`` axis with e.g. ``tickformat="%M:%S"``, this renders
    paces/times as clean clock labels instead of raw seconds. Non-finite inputs
    (``NaN``) become ``NaT`` so the line shows a gap there rather than a spike.
    """
    arr = np.asarray(seconds, dtype="float64")
    out = np.full(arr.shape, np.datetime64("NaT"), dtype="datetime64[ms]")
    finite = np.isfinite(arr)
    out[finite] = np.datetime64("1970-01-01T00:00:00") + (
        arr[finite] * 1000.0
    ).astype("int64").astype("timedelta64[ms]")
    return out


def fmt_hms(seconds: float) -> str:
    """Seconds → ``h:mm:ss`` (or ``m:ss`` under an hour); ``""`` for non-finite."""
    if not np.isfinite(seconds):
        return ""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def fmt_pace(seconds_per_km: float) -> str:
    """Seconds-per-km → ``m:ss /km``; ``""`` for non-finite."""
    if not np.isfinite(seconds_per_km):
        return ""
    minutes, secs = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{secs:02d}/km"
