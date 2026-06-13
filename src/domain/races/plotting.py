"""Cross-race comparison plots: one line per race, switchable x-axis.

Each metric (GAP pace, power, heart rate, power-to-HR) gets its own figure with
one line per race. The x-axis can be either elapsed time or distance covered.
"""

from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.domain.gap import theme
from src.domain.races.metrics import RaceSeries

# metric key -> (attribute on RaceSeries, y-axis label, title)
_METRICS = {
    "gap_pace": ("gap_pace_s_per_km", "GAP pace (min/km, lower = faster)", "Gradient-Adjusted Pace"),
    "power": ("power_w", "Power (W)", "Power"),
    "heartrate": ("heartrate", "Heart rate (bpm)", "Heart Rate"),
    "power_to_hr": ("power_to_hr", "Power / HR (W/bpm)", "Power-to-Heart-Rate"),
}

_X_AXES = {
    "time": ("time_s", "Time (min)", 1.0 / 60.0),
    "distance": ("distance_m", "Distance (km)", 1.0 / 1000.0),
}


def _format_pace_ticks(ax: plt.Axes) -> None:
    """Render y-axis seconds/km as m:ss pace labels."""
    def fmt(value, _pos):
        if value < 0 or not np.isfinite(value):
            return ""
        minutes, seconds = divmod(int(round(value)), 60)
        return f"{minutes}:{seconds:02d}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    # Lower pace (faster) at the top reads more naturally for performance.
    ax.invert_yaxis()


def plot_metric_comparison(
    series_list: Sequence[RaceSeries],
    *,
    metric_key: str,
    x_axis: str,
    gap_as_speed: bool = False,
) -> plt.Figure:
    """Plot one metric across all races. Raises KeyError on unknown keys.

    ``gap_as_speed`` only affects the GAP plot: when True it shows GAP as speed
    (km/h, higher = faster) instead of pace (min/km).
    """
    attr, y_label, title = _METRICS[metric_key]
    x_attr, x_label, x_scale = _X_AXES[x_axis]

    show_gap_speed = metric_key == "gap_pace" and gap_as_speed
    if show_gap_speed:
        y_label = "GAP speed (km/h, higher = faster)"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(theme.FIGURE_FACE)
    ax.set_facecolor(theme.AXES_FACE)

    plotted = 0
    for i, series in enumerate(series_list):
        y = getattr(series, attr)
        if y is None:
            continue
        if show_gap_speed:
            y = 3600.0 / np.asarray(y, dtype=float)  # s/km → km/h
        x = getattr(series, x_attr) * x_scale
        color = theme.CURVE_CYCLE[i % len(theme.CURVE_CYCLE)]
        ax.plot(
            x, y, color=color, linewidth=2.2, label=series.label,
            solid_capstyle="round", alpha=0.9, zorder=3,
        )
        plotted += 1

    ax.set_xlabel(x_label, color=theme.TEXT, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, color=theme.TEXT, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{title} across the race", color=theme.TEXT, fontweight="bold",
        fontsize=15, pad=12,
    )
    ax.grid(True, alpha=0.45, color=theme.GRID, linewidth=0.9)
    ax.tick_params(colors=theme.TEXT)
    for side, spine in ax.spines.items():
        spine.set_color(theme.SPINE)
        spine.set_visible(side in ("left", "bottom"))

    if metric_key == "gap_pace" and not show_gap_speed:
        _format_pace_ticks(ax)

    if plotted:
        legend = ax.legend(
            facecolor=theme.AXES_FACE, edgecolor=theme.SPINE, framealpha=0.95, fontsize=10
        )
        for text in legend.get_texts():
            text.set_color(theme.TEXT)
    return fig


def available_metric_keys(series_list: List[RaceSeries]) -> List[str]:
    """Metric keys that at least one race can actually plot (skips all-None)."""
    keys = []
    for key, (attr, _, _) in _METRICS.items():
        if any(getattr(s, attr) is not None for s in series_list):
            keys.append(key)
    return keys
