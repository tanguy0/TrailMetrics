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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.domain.assessment import rate
from src.domain.models.gap import GapCurve

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


@dataclass(frozen=True)
class TerrainAssessment:
    key: str
    # ``None`` when the runner's curve has no point on this terrain.
    extra_cost_pct: Optional[float]
    level: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "extra_cost_pct": None if self.extra_cost_pct is None else round(self.extra_cost_pct, 1),
            "level": self.level,
        }


def assess(curve: Optional[GapCurve], reference: GapCurve) -> List[TerrainAssessment]:
    """One assessment per terrain, in :data:`TERRAINS` order."""
    return [
        TerrainAssessment(key, extra, rate(extra))
        for key, extra in ((key, _extra_cost(curve, reference, low, high)) for key, low, high in TERRAINS)
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
