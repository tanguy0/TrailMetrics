from typing import Dict

import matplotlib.pyplot as plt

from src.domain.models.gap import GapCurve


def plot_gap_curves(gap_curves: Dict[str, GapCurve]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6))

    for name, curve in gap_curves.items():
        color = curve.color or None
        plt.plot(curve.bin_centers, curve.means, color=color, linewidth=2, label=name)
        plt.fill_between(
            curve.bin_centers,
            curve.means - curve.stds,
            curve.means + curve.stds,
            alpha=0.2,
            color=color,
        )

    plt.xlabel("Elevation Gain (m/km)")
    plt.ylabel("Speed Adjuster (GAP/speed)")
    plt.title("GAP Curve(s) and standard deviation(s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig
