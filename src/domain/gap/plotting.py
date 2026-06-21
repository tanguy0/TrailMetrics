from typing import Dict

import matplotlib.pyplot as plt

from src.domain.gap import theme
from src.domain.models.gap import GapCurve
from src.translations import DEFAULT_LANG, translate


def plot_gap_curves(
    gap_curves: Dict[str, GapCurve], show_std: bool = True, lang: str = DEFAULT_LANG
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(theme.FIGURE_FACE)
    ax.set_facecolor(theme.AXES_FACE)

    for i, (name, curve) in enumerate(gap_curves.items()):
        color = curve.color or theme.CURVE_CYCLE[i % len(theme.CURVE_CYCLE)]
        ax.plot(
            curve.bin_centers,
            curve.means,
            color=color,
            linewidth=2.8,
            linestyle=curve.linestyle,
            label=name,
            solid_capstyle="round",
            zorder=3,
        )
        if show_std:
            ax.fill_between(
                curve.bin_centers,
                curve.means - curve.stds,
                curve.means + curve.stds,
                alpha=0.16,
                color=color,
                zorder=2,
            )

    ax.set_xlabel(
        translate("plot.gap.xlabel", lang), color=theme.TEXT, fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        translate("plot.gap.ylabel", lang), color=theme.TEXT, fontsize=12, fontweight="bold"
    )
    ax.set_title(
        translate("plot.gap.title_std" if show_std else "plot.gap.title", lang),
        color=theme.TEXT,
        fontweight="bold",
        fontsize=15,
        pad=12,
    )
    ax.grid(True, alpha=0.45, color=theme.GRID, linewidth=0.9)
    ax.tick_params(colors=theme.TEXT)
    for side, spine in ax.spines.items():
        spine.set_color(theme.SPINE)
        spine.set_visible(side in ("left", "bottom"))
    legend = ax.legend(
        facecolor=theme.AXES_FACE, edgecolor=theme.SPINE, framealpha=0.95, fontsize=10
    )
    for text in legend.get_texts():
        text.set_color(theme.TEXT)
    return fig
