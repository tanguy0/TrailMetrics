"""Shared Plotly styling so every figure in the app matches the theme.

All charts are Plotly (interactive: clickable legends, hover read-outs, zoom).
This module is the single source of truth for the Trail / Earthy look — the base
figure layout, the line-color cycle and a few formatting helpers — so the GAP,
race-comparison and long-term-progress figures stay visually coherent.
"""

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from src.domain.gap import theme

# Distinct on-theme line colors, cycled for traces without an explicit color.
CURVE_PALETTE = [
    "#2E6F40", "#C65D3B", "#E8A33D", "#3A6EA5", "#7A4E9E",
    "#5E9C4E", "#A6843E", "#14532B", "#B5651D", "#6B4226",
    "#2A7E8C", "#9E4E6E",
]

# matplotlib linestyle → Plotly dash, so existing GapCurve.linestyle values port.
DASH_BY_LINESTYLE = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}


def base_figure(*, title: str, x_title: str, y_title: str, height: int = 480) -> go.Figure:
    """An empty figure pre-styled with the Trail / Earthy theme."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme.TEXT, size=18)),
        paper_bgcolor=theme.FIGURE_FACE,
        plot_bgcolor=theme.AXES_FACE,
        font=dict(color=theme.TEXT, size=13),
        legend=dict(
            bgcolor=theme.AXES_FACE,
            bordercolor=theme.SPINE,
            borderwidth=1,
            font=dict(color=theme.TEXT),
        ),
        margin=dict(l=70, r=25, t=60, b=55),
        height=height,
        hoverlabel=dict(bgcolor=theme.AXES_FACE, font=dict(color=theme.TEXT)),
    )
    fig.update_xaxes(
        title_text=x_title, gridcolor=theme.GRID, linecolor=theme.SPINE,
        zeroline=False, color=theme.TEXT,
    )
    fig.update_yaxes(
        title_text=y_title, gridcolor=theme.GRID, linecolor=theme.SPINE,
        zeroline=False, color=theme.TEXT,
    )
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
