"""Smoothing services for GAP curves.

Note on the std field after smoothing: smoothers apply the same kernel to
`means` and `stds`. After smoothing, `stds` represents the displayed band
around the smoothed curve, not the raw per-bin spread.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.domain.models.gap import GapCurve


class CurveSmoother(ABC):
    """A smoother that takes a GapCurve and returns a smoothed copy."""

    @abstractmethod
    def smooth(self, curve: GapCurve) -> GapCurve:
        ...


class RollingMeanCurveSmoother(CurveSmoother):
    """Symmetric rolling-mean smoother with edge padding.

    Cheap and shape-agnostic. Distorts edges (pads via edge repetition) and
    ignores per-bin sample counts. Prefer LoessCurveSmoother for GAP curves
    where bins are non-uniform and edges are sparse.
    """

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window if window % 2 == 1 else window + 1

    def smooth(self, curve: GapCurve) -> GapCurve:
        if self.window == 1 or len(curve.bin_centers) <= 1:
            return curve
        return GapCurve(
            bin_centers=curve.bin_centers,
            means=self._rolling_mean(curve.means),
            stds=self._rolling_mean(curve.stds),
            counts=curve.counts,
            color=curve.color,
            extra=dict(curve.extra),
        )

    def _rolling_mean(self, values: np.ndarray) -> np.ndarray:
        pad = (self.window - 1) // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.array(
            [padded[i:i + self.window].mean() for i in range(len(values))]
        )


class LoessCurveSmoother(CurveSmoother):
    """LOESS (locally weighted polynomial) smoother in gradient space.

    For each bin, fits a local polynomial (default degree 2) to neighbors,
    weighted by:
      1. proximity in gradient space (tricube kernel),
      2. per-bin sample count (so well-supported bins dominate).

    This handles three problems that a naive rolling mean cannot:
      - non-uniform bin spacing (efficiency model uses variable-width buckets),
      - sparse edge bins (down-weighted by their low count),
      - boundary effects (no padding/extrapolation; the local regression
        naturally uses whichever side has data).

    Parameters
    ----------
    bandwidth_fraction : float in (0, 1]
        Fraction of total bins used as the local neighborhood for each fit.
        0.5 means "the closest 50% of bins contribute to each smoothed point."
    polyorder : int
        Degree of the local polynomial. 2 preserves U-shape; 1 is more linear.
    """

    def __init__(self, bandwidth_fraction: float = 0.4, polyorder: int = 2):
        if not 0 < bandwidth_fraction <= 1:
            raise ValueError("bandwidth_fraction must be in (0, 1]")
        if polyorder < 1:
            raise ValueError("polyorder must be >= 1")
        self.bandwidth_fraction = bandwidth_fraction
        self.polyorder = polyorder

    def smooth(self, curve: GapCurve) -> GapCurve:
        x = curve.bin_centers.astype(float)
        n = len(x)
        if n <= self.polyorder + 1:
            return curve

        weights = curve.counts.astype(float)
        if not np.any(weights):
            weights = np.ones(n)

        means = self._loess(x, curve.means.astype(float), weights)
        stds = self._loess(x, curve.stds.astype(float), weights)

        return GapCurve(
            bin_centers=curve.bin_centers,
            means=means,
            stds=stds,
            counts=curve.counts,
            color=curve.color,
            extra=dict(curve.extra),
        )

    def _loess(self, x: np.ndarray, y: np.ndarray, count_weights: np.ndarray) -> np.ndarray:
        n = len(x)
        k = max(self.polyorder + 1, int(np.ceil(self.bandwidth_fraction * n)))
        k = min(k, n)
        smoothed = np.empty(n)

        for i in range(n):
            distances = np.abs(x - x[i])
            kth_distance = np.partition(distances, k - 1)[k - 1]
            # Tricube kernel: (1 - (d/d_max)^3)^3 for d <= d_max, else 0.
            scale = kth_distance if kth_distance > 0 else 1.0
            u = np.clip(distances / scale, 0, 1)
            kernel = (1 - u ** 3) ** 3
            w = kernel * count_weights
            if w.sum() == 0:
                smoothed[i] = y[i]
                continue
            # Local polynomial fit via weighted least squares (Vandermonde basis centered at x[i]).
            dx = x - x[i]
            basis = np.vander(dx, self.polyorder + 1, increasing=True)
            wsqrt = np.sqrt(w)
            A = basis * wsqrt[:, None]
            b = y * wsqrt
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
            smoothed[i] = coeffs[0]  # value at dx=0 is the intercept

        return smoothed
