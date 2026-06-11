"""Reference GAP curves used for comparison against the personalized models."""

import numpy as np

from src.domain.gap import theme
from src.domain.models.gap import GapCurve


def balanced_runner() -> GapCurve:
    """Reference 'balanced runner' GAP curve.

    Source data: https://medium.com/strava-engineering/an-improved-gap-model-8b07ae8886c3
    """
    return GapCurve(
        bin_centers=np.array([-350, -300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350]),
        means=np.array([1.7, 1.5, 1.3, 1.1, 0.9, 0.85, 0.9, 1, 1.2, 1.45, 1.8, 2.3, 2.75, 3.15, 3.55]),
        stds=np.zeros(15),
        counts=np.ones(15, dtype=int),
        color=theme.BALANCED_RUNNER,
        linestyle="--",
    )


def kilian_jornet() -> GapCurve:
    """Kilian Jornet's personalized GAP curve.

    Source: https://pickletech.eu/blog-gap/
    """
    return GapCurve(
        bin_centers=np.array([-350, -300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350]),
        means=np.array([1.6, 1.35, 1.15, 1, 0.87, 0.85, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9, 2.15, 2.4]),
        stds=np.zeros(15),
        counts=np.ones(15, dtype=int),
        color=theme.KILIAN,
        linestyle="--",
    )
