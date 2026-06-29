from typing import Dict

import numpy as np
import plotly.graph_objects as go

from src.domain.models.gap import GapCurve
from src.domain.plotting_common import (
    CURVE_PALETTE,
    DASH_BY_LINESTYLE,
    base_figure,
    rgba,
)
from src.translations import DEFAULT_LANG, translate


def plot_gap_curves(
    gap_curves: Dict[str, GapCurve], show_std: bool = True, lang: str = DEFAULT_LANG
) -> go.Figure:
    """GAP curves (speed adjuster vs elevation gain), one clickable trace each.

    When ``show_std`` is set, each curve carries a translucent ±1σ band. Curve
    color / line style come from the :class:`GapCurve` (matplotlib-style line
    codes are mapped to Plotly dashes), falling back to the shared palette.
    """
    fig = base_figure(
        title=translate("plot.gap.title_std" if show_std else "plot.gap.title", lang),
        x_title=translate("plot.gap.xlabel", lang),
        y_title=translate("plot.gap.ylabel", lang),
    )

    for i, (name, curve) in enumerate(gap_curves.items()):
        color = curve.color or CURVE_PALETTE[i % len(CURVE_PALETTE)]
        dash = DASH_BY_LINESTYLE.get(curve.linestyle, "solid")
        centers = np.asarray(curve.bin_centers, dtype=float)
        means = np.asarray(curve.means, dtype=float)

        if show_std:
            stds = np.asarray(curve.stds, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([centers, centers[::-1]]),
                    y=np.concatenate([means + stds, (means - stds)[::-1]]),
                    fill="toself",
                    fillcolor=rgba(color, 0.16),
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=name,
                    name=name,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=means,
                name=name,
                legendgroup=name,
                mode="lines",
                line=dict(color=color, width=2.8, dash=dash),
                hovertemplate="%{x:.0f} m/km<br>%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )

    return fig
