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
    UNASSIGNED_INDEX,
    GradientMap,
    SeasonCurve,
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

# Neutral color for the "outside every season" curve (continuous mode).
UNASSIGNED_COLOR = "#9AA0A6"

_PR_METERS = {label: meters for label, meters in PR_DISTANCES}


def _curve_color(index: int) -> str:
    if index == UNASSIGNED_INDEX:
        return UNASSIGNED_COLOR
    return CURVE_PALETTE[index % len(CURVE_PALETTE)]


# --- 1. Personal-record evolution ------------------------------------------

def plot_pr_progression(
    progressions: Dict[str, List[Tuple]],
    *,
    as_pace: bool = True,
    lang: str = DEFAULT_LANG,
    end_date=None,
) -> go.Figure:
    """Stepped record-evolution line, one clickable trace per distance.

    The y-axis is reversed so a new (faster) record sits *higher* — each record
    jumps up. ``as_pace`` shows pace (min/km, comparable across distances);
    otherwise the raw record time. The step shape (``hv``) holds the previous
    record flat until the day a new one is set, then jumps to it. When
    ``end_date`` is given, each line is extended flat from its last record to
    that date (a markerless point) so the current record runs to the plot edge.
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
        # A real marker on each record; the optional trailing point that carries
        # the line to the plot edge gets a size-0 (invisible) marker.
        marker_sizes = [8] * len(dates)

        if end_date is not None and dates and end_date > dates[-1]:
            dates = dates + [end_date]
            times_s = times_s + [times_s[-1]]
            paces = paces + [paces[-1]]
            customdata = customdata + [customdata[-1]]
            marker_sizes = marker_sizes + [0]

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
                marker=dict(size=marker_sizes),
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


# --- 2/3/4/6. Season curves (mileage / elevation / gradient / power-to-HR) --

def plot_season_curves(
    curves: Sequence[SeasonCurve],
    *,
    title: str,
    y_title: str,
    y_fmt: str,
    hover_unit: str,
    overlay: bool,
    step: bool = False,
    markers: bool = True,
    x_max_months=None,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """One clickable trace per season, on the overlay or continuous axis.

    ``overlay`` picks the x-axis: elapsed months since each season's start (all
    seasons from 0) vs. the real calendar timeline. ``step`` draws a staircase
    (cumulative); ``markers`` adds per-point dots. Colors are stable per season;
    the out-of-season curve is grey.
    """
    x_title = translate(
        "plot.ltp.x.months_since_start" if overlay else "plot.ltp.x.time", lang
    )
    fig = base_figure(title=title, x_title=x_title, y_title=y_title)

    mode = "lines+markers" if markers else "lines"
    months_lbl = translate("plot.ltp.months", lang)
    for curve in curves:
        line = dict(width=2.4, color=_curve_color(curve.index))
        if step:
            line["shape"] = "hv"
        if overlay:
            hovertemplate = (
                f"%{{x:.1f}} {months_lbl}<br>%{{y:{y_fmt}}} {hover_unit}"
                "<extra>%{fullData.name}</extra>"
            )
        else:
            hovertemplate = (
                f"%{{x|%d %b %Y}}<br>%{{y:{y_fmt}}} {hover_unit}"
                "<extra>%{fullData.name}</extra>"
            )
        fig.add_trace(
            go.Scatter(
                x=curve.x,
                y=curve.y,
                name=curve.name,
                mode=mode,
                line=line,
                marker=dict(size=5),
                hovertemplate=hovertemplate,
            )
        )

    if overlay:
        rng = [0, x_max_months] if x_max_months else None
        fig.update_xaxes(dtick=1, range=rng)
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
