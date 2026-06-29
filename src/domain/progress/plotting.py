"""Interactive Plotly figures for the long-term-progress page.

Clickable legends (show/hide a distance, season or gradient band), hover
read-outs and zoom for free. Layout, palette and formatting come from
:mod:`src.domain.plotting_common` so every figure matches the rest of the app.
"""

from typing import Dict, List, Sequence, Tuple

import plotly.graph_objects as go

from src.domain.plotting_common import (
    CURVE_PALETTE,
    base_figure,
    durations_to_datetimes,
    fmt_hms,
    fmt_pace,
)
from src.domain.progress.aggregates import (
    CumulativeYear,
    GradientMap,
    GradientYear,
    PowerHrSeries,
)
from src.domain.progress.models import GRADIENT_BANDS, PR_DISTANCES
from src.translations import DEFAULT_LANG, translate

# Gradient bands, green (descent) → red (ascent), keyed by band key.
BAND_COLORS = {
    "steep_descent": "#1B7A3D",
    "gentle_descent": "#7FB069",
    "flat": "#E8A33D",
    "gentle_ascent": "#C65D3B",
    "steep_ascent": "#8E2C18",
}

_PR_METERS = {label: meters for label, meters in PR_DISTANCES}


# --- 1. Personal-record evolution ------------------------------------------

def plot_pr_progression(
    progressions: Dict[str, List[Tuple]],
    *,
    as_pace: bool = True,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """Stepped record-evolution line, one clickable trace per distance.

    The y-axis is reversed so a new (faster) record sits *higher* — each record
    jumps up. ``as_pace`` shows pace (min/km, comparable across distances);
    otherwise the raw record time. The step shape (``hv``) holds the previous
    record flat until the day a new one is set, then jumps to it.
    """
    y_key = "plot.ltp.records.y_pace" if as_pace else "plot.ltp.records.y_time"
    fig = base_figure(
        title=translate("plot.ltp.records.title", lang),
        x_title=translate("plot.ltp.records.x", lang),
        y_title=translate(y_key, lang),
    )
    record_lbl = translate("plot.ltp.records.hover_record", lang)
    pace_lbl = translate("plot.ltp.records.hover_pace", lang)

    for i, (label, _) in enumerate(PR_DISTANCES):
        points = progressions.get(label) or []
        if not points:
            continue
        dates = [d for d, _ in points]
        times_s = [t for _, t in points]
        meters = _PR_METERS[label]
        paces = [t / (meters / 1000.0) for t in times_s]
        customdata = [[fmt_hms(t), fmt_pace(p)] for t, p in zip(times_s, paces)]
        y = durations_to_datetimes(paces if as_pace else times_s)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y,
                customdata=customdata,
                name=label,
                mode="lines+markers",
                line=dict(
                    shape="hv",
                    width=2.4,
                    color=CURVE_PALETTE[i % len(CURVE_PALETTE)],
                ),
                marker=dict(size=8),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"{record_lbl}: %{{customdata[0]}}<br>"
                    f"{pace_lbl}: %{{customdata[1]}}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    tickformat = "%M:%S" if as_pace else "%H:%M:%S"
    fig.update_yaxes(type="date", tickformat=tickformat, autorange="reversed")
    return fig


# --- 2 & 3. Annual cumulative (distance / elevation) -----------------------

def plot_annual_cumulative(
    series: Sequence[CumulativeYear],
    *,
    y_title: str,
    title: str,
    unit: str,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """Cumulative-over-the-year curves, one clickable trace per season."""
    fig = base_figure(
        title=title, x_title=translate("plot.ltp.x.month", lang), y_title=y_title
    )
    for i, cy in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=cy.x,
                y=cy.y,
                name=str(cy.year),
                mode="lines",
                line=dict(
                    shape="hv",
                    width=2.4,
                    color=CURVE_PALETTE[i % len(CURVE_PALETTE)],
                ),
                hovertemplate=(
                    "%{x|%d %b}<br>%{y:,.0f} " + unit
                    + "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_xaxes(tickformat="%b", dtick="M1")
    return fig


# --- 4. Average gradient per season ----------------------------------------

def plot_avg_gradient(
    series: Sequence[GradientYear],
    *,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """Average-gradient (%) curves per bin, one clickable trace per season."""
    fig = base_figure(
        title=translate("plot.ltp.gradient.title", lang),
        x_title=translate("plot.ltp.x.month", lang),
        y_title=translate("plot.ltp.gradient.y", lang),
    )
    for i, gy in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=gy.x,
                y=gy.y,
                name=str(gy.year),
                mode="lines+markers",
                line=dict(
                    width=2.4, color=CURVE_PALETTE[i % len(CURVE_PALETTE)]
                ),
                marker=dict(size=6),
                hovertemplate=(
                    "%{x|%d %b}<br>%{y:.1f} %<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_xaxes(tickformat="%b", dtick="M1")
    return fig


# --- 6. Power-to-HR efficiency ---------------------------------------------

def plot_power_hr(
    series: Sequence[PowerHrSeries],
    *,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """Weekly power-to-HR along one continuous timeline, colored per season.

    One trace per year (own color, clickable in the legend) on a shared real-date
    axis, so the curve runs end-to-end and simply changes color at each Jan 1.
    """
    fig = base_figure(
        title=translate("plot.ltp.power_hr.title", lang),
        x_title=translate("plot.ltp.power_hr.x", lang),
        y_title=translate("plot.ltp.power_hr.y", lang),
    )
    for i, phs in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=phs.x,
                y=phs.y,
                name=str(phs.year),
                mode="lines+markers",
                line=dict(
                    width=2.4, color=CURVE_PALETTE[i % len(CURVE_PALETTE)]
                ),
                marker=dict(size=6),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>%{y:.2f} W/bpm<extra>%{fullData.name}</extra>"
                ),
            )
        )
    return fig


# --- 5. Gradient map -------------------------------------------------------

def plot_gradient_map(
    gmap: GradientMap,
    *,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """100%-stacked area of time-in-band per bin, green (descent) → red (ascent)."""
    fig = base_figure(
        title=translate("plot.ltp.gradient_map.title", lang),
        x_title=translate("plot.ltp.gradient_map.x", lang),
        y_title=translate("plot.ltp.gradient_map.y", lang),
    )
    for key, _, _ in GRADIENT_BANDS:
        fig.add_trace(
            go.Scatter(
                x=gmap.x,
                y=gmap.band_pct.get(key, []),
                name=translate(f"ltp.band.{key}", lang),
                mode="lines",
                line=dict(width=0.5, color=BAND_COLORS[key]),
                stackgroup="one",
                fillcolor=BAND_COLORS[key],
                hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} %<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_yaxes(range=[0, 100], ticksuffix=" %")
    fig.update_layout(hovermode="x unified")
    return fig
