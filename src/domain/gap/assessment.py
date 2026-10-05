"""The GAP profile: how a runner's slopes compare with the balanced runner's.

Four terrains, by gradient. On each, the runner's speed adjuster (GAP / speed —
how much a gradient costs them) is compared with the reference's at the same
gradients: ``mean(runner / reference) − 1``, as a percentage. Above zero the
terrain costs them more than it costs the reference runner. Flat ground (−3 % to
+3 %) is left out on purpose: every curve is ≈ 1 there, so it says nothing.

A terrain the runner's curve does not reach — no fitted bucket in that gradient
range — has no level: ``insufficient`` rather than a guess extrapolated from the
neighbouring terrain.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.domain.assessment import Assessment
from src.domain.charts.ir import Axis, AxisKind, ChartData, Marker, Trace, TraceKind
from src.domain.gap import theme
from src.domain.models.gap import GapCurve
from src.translations import translate

# Gradients in m/km (the curves' x unit): 120 m/km = 12 %.
STEEP = 120.0
GENTLE = 30.0

# ``(key, lowest gradient, highest gradient)``, open-ended at the extremes.
TERRAINS: Tuple[Tuple[str, float, float], ...] = (
    ("steep_downhill", -np.inf, -STEEP),
    ("downhill", -STEEP, -GENTLE),
    ("uphill", GENTLE, STEEP),
    ("steep_uphill", STEEP, np.inf),
)


def assess(curve: Optional[GapCurve], reference: GapCurve) -> List[Assessment]:
    """One assessment per terrain, in :data:`TERRAINS` order."""
    return [
        Assessment.of(key, _extra_cost(curve, reference, low, high))
        for key, low, high in TERRAINS
    ]


def _extra_cost(curve: Optional[GapCurve], reference: GapCurve, low: float, high: float) -> Optional[float]:
    if curve is None:
        return None
    x = np.asarray(curve.bin_centers, dtype=float)
    y = np.asarray(curve.means, dtype=float)
    inside = (x >= low) & (x <= high) & np.isfinite(y) & (y > 0)
    if not inside.any():
        return None
    order = np.argsort(reference.bin_centers)
    ref = np.interp(
        x[inside],
        np.asarray(reference.bin_centers, dtype=float)[order],
        np.asarray(reference.means, dtype=float)[order],
    )
    return float(np.mean(y[inside] / ref) - 1.0) * 100.0


# The chart's x window, in m/km: wide enough for every terrain, and widened to
# wherever the runner's own curve reaches.
_CHART_X = (-250.0, 250.0)


def profile_chart(curve: GapCurve, reference: GapCurve, lang: str) -> ChartData:
    """The runner's curve against the reference — the one the levels were read on.

    x in % of gradient (the curves' m/km ÷ 10); a boundary rule at each terrain
    limit, so the four tiles above map onto the figure. No title: it lives in the
    card (charts.md).
    """
    x = np.asarray(curve.bin_centers, dtype=float)
    low, high = min(_CHART_X[0], float(x.min())), max(_CHART_X[1], float(x.max()))
    rx = np.asarray(reference.bin_centers, dtype=float)
    keep = (rx >= low) & (rx <= high)

    def trace(cx, cy, name, color, dash, width):
        return Trace(
            name=name,
            x=(np.asarray(cx) / 10).round(1).tolist(),
            y=np.asarray(cy, dtype=float).round(3).tolist(),
            kind=TraceKind.LINE,
            color=color,
            dash=dash,
            width=width,
            hover_template="%{x:+.0f} %<br>×%{y:.2f}<extra>%{fullData.name}</extra>",
        )

    return ChartData(
        x_axis=Axis(title=translate("ui.gap_tool.chart.x", lang), kind=AxisKind.LINEAR,
                    tick_format="+.0f", suffix=" %"),
        y_axis=Axis(title=translate("ui.gap_tool.chart.y", lang), kind=AxisKind.LINEAR,
                    tick_format=".1f"),
        traces=[
            trace(rx[keep], np.asarray(reference.means)[keep],
                  translate("gap.refs.balanced", lang), theme.BALANCED_RUNNER, "--", 1.5),
            trace(x, curve.means, translate("ui.gap_tool.chart.you", lang),
                  theme.EFFICIENCY, "-", 2.4),
        ],
        markers=[Marker(kind="boundary", x=v / 10) for v in (-STEEP, -GENTLE, GENTLE, STEEP)],
    )
