"""Cross-race comparison plots: one line per race, switchable x-axis.

Each metric (GAP pace, power, heart rate, power-to-HR) gets its own interactive
figure with one clickable trace per race. The x-axis can be elapsed time or
distance covered.
"""

from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go

from src.domain.plotting_common import (
    CURVE_PALETTE,
    base_figure,
    durations_to_datetimes,
    fmt_pace,
)
from src.domain.races.metrics import RaceSeries
from src.translations import DEFAULT_LANG, translate

# metric key -> (attribute on RaceSeries, y-axis label translation key, title key)
_METRICS = {
    "gap_pace": ("gap_pace_s_per_km", "plot.races.gap_pace.y", "plot.races.gap_pace.title"),
    "power": ("power_w", "plot.races.power.y", "plot.races.power.title"),
    "heartrate": ("heartrate", "plot.races.hr.y", "plot.races.hr.title"),
    "power_to_hr": ("power_to_hr", "plot.races.p2hr.y", "plot.races.p2hr.title"),
}

# x-axis key -> (attribute on RaceSeries, axis label translation key, unit scale)
_X_AXES = {
    "time": ("time_s", "plot.races.x.time", 1.0 / 60.0),
    "distance": ("distance_m", "plot.races.x.distance", 1.0 / 1000.0),
}


def plot_metric_comparison(
    series_list: Sequence[RaceSeries],
    *,
    metric_key: str,
    x_axis: str,
    gap_as_speed: bool = False,
    lang: str = DEFAULT_LANG,
) -> go.Figure:
    """Plot one metric across all races. Raises KeyError on unknown keys.

    ``gap_as_speed`` only affects the GAP plot: when True it shows GAP as speed
    (km/h, higher = faster) instead of pace (min/km). Pace is drawn on a reversed
    duration axis so faster sits at the top and ticks read as m:ss.
    """
    attr, y_label_key, title_key = _METRICS[metric_key]
    x_attr, x_label_key, x_scale = _X_AXES[x_axis]
    y_label = translate(y_label_key, lang)
    x_label = translate(x_label_key, lang)
    title = f"{translate(title_key, lang)} {translate('plot.races.title_suffix', lang)}"

    show_gap_speed = metric_key == "gap_pace" and gap_as_speed
    y_is_pace = metric_key == "gap_pace" and not gap_as_speed
    if show_gap_speed:
        y_label = translate("plot.races.gap_speed.y", lang)

    fig = base_figure(title=title, x_title=x_label, y_title=y_label, height=420)

    for i, series in enumerate(series_list):
        raw = getattr(series, attr)
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=float)
        x = np.asarray(getattr(series, x_attr), dtype=float) * x_scale
        color = CURVE_PALETTE[i % len(CURVE_PALETTE)]
        common = dict(
            x=x, name=series.label, mode="lines", line=dict(color=color, width=2.2),
        )

        if show_gap_speed:
            y = 3600.0 / raw  # s/km → km/h
            fig.add_trace(go.Scatter(
                y=y, hovertemplate="%{y:.1f} km/h<extra>%{fullData.name}</extra>", **common
            ))
        elif y_is_pace:
            fig.add_trace(go.Scatter(
                y=durations_to_datetimes(raw),
                customdata=[fmt_pace(p) for p in raw],
                hovertemplate="%{customdata}<extra>%{fullData.name}</extra>",
                **common,
            ))
        else:
            fig.add_trace(go.Scatter(
                y=raw, hovertemplate="%{y:.0f}<extra>%{fullData.name}</extra>", **common
            ))

    if y_is_pace:
        # Lower pace (faster) at the top reads more naturally for performance.
        fig.update_yaxes(type="date", tickformat="%M:%S", autorange="reversed")
    return fig


def available_metric_keys(series_list: List[RaceSeries]) -> List[str]:
    """Metric keys that at least one race can actually plot (skips all-None)."""
    keys = []
    for key, (attr, _, _) in _METRICS.items():
        if any(getattr(s, attr) is not None for s in series_list):
            keys.append(key)
    return keys
