from typing import Optional, Tuple

import numpy as np

from src.domain.gap.base import GapModel
from src.domain.models.gap import DownsampledDataset, GapCurve


class EfficiencyGapModel(GapModel):
    """
    Strava-style GAP model: bin data by gradient, compute mean normalized
    `heartrate/speed` efficiency per bucket. GAP = speed * efficiency_factor.
    """

    DEFAULT_FLAT_RANGE = (-10.0, 10.0)
    MIN_FLAT_SAMPLES = 100

    def __init__(self, min_samples_per_bucket: int = 250):
        self.min_samples_per_bucket = min_samples_per_bucket
        self.bucket_centers: Optional[np.ndarray] = None
        self.bucket_means: Optional[np.ndarray] = None
        self.bucket_stds: Optional[np.ndarray] = None
        self.bucket_counts: Optional[np.ndarray] = None

    def fit_on_subset(
        self,
        dataset: DownsampledDataset,
        heartrate_range: Tuple[float, float],
    ) -> "EfficiencyGapModel":
        """Fit a fresh efficiency model on the slice of the dataset within the given HR range.

        Each HR slice is smaller than the full dataset, so callers typically lower
        `min_samples_per_bucket` (or pass it in via the constructor) to keep usable resolution.
        """
        hr_mask = (
            (dataset.heartrate >= heartrate_range[0]) & (dataset.heartrate <= heartrate_range[1])
        )
        subset = DownsampledDataset(
            speed=dataset.speed[hr_mask],
            elevation_gain=dataset.elevation_gain[hr_mask],
            heartrate=dataset.heartrate[hr_mask],
            sport_types=dataset.sport_types[hr_mask],
        )
        return self.fit(subset)

    def fit(self, dataset: DownsampledDataset) -> "EfficiencyGapModel":
        efficiencies = self._normalized_efficiency(
            dataset.speed, dataset.elevation_gain, dataset.heartrate
        )

        sort_idx = np.argsort(dataset.elevation_gain)
        sorted_grad = dataset.elevation_gain[sort_idx]
        sorted_eff = efficiencies[sort_idx]

        centers, means, stds, counts = [], [], [], []
        current_bucket, current_grads = [], []

        for grad, eff in zip(sorted_grad, sorted_eff):
            current_bucket.append(eff)
            current_grads.append(grad)
            if len(current_bucket) >= self.min_samples_per_bucket:
                centers.append(np.mean(current_grads))
                means.append(np.mean(current_bucket))
                stds.append(np.std(current_bucket))
                counts.append(len(current_bucket))
                current_bucket, current_grads = [], []

        if current_bucket:
            centers.append(np.mean(current_grads))
            means.append(np.mean(current_bucket))
            stds.append(np.std(current_bucket))
            counts.append(len(current_bucket))

        self.bucket_centers = np.array(centers)
        self.bucket_means = np.array(means)
        self.bucket_stds = np.array(stds)
        self.bucket_counts = np.array(counts)
        return self

    def predict_gap(
        self,
        speed: np.ndarray,
        elevation_gain: np.ndarray,
        heartrate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._require_fitted()
        speed = np.atleast_1d(np.asarray(speed))
        elev = np.atleast_1d(np.asarray(elevation_gain))
        idx = np.argmin(np.abs(self.bucket_centers[None, :] - elev[:, None]), axis=1)
        factors = self.bucket_means[idx]
        return speed * factors

    def gap_curve(self) -> GapCurve:
        self._require_fitted()
        return GapCurve(
            bin_centers=self.bucket_centers,
            means=self.bucket_means,
            stds=self.bucket_stds,
            counts=self.bucket_counts,
        )

    def _normalized_efficiency(
        self,
        speeds: np.ndarray,
        elevation_gains: np.ndarray,
        heartrates: np.ndarray,
    ) -> np.ndarray:
        efficiencies = heartrates / speeds
        flat_mask = (
            (elevation_gains > self.DEFAULT_FLAT_RANGE[0])
            & (elevation_gains < self.DEFAULT_FLAT_RANGE[1])
        )

        if np.sum(flat_mask) > self.MIN_FLAT_SAMPLES:
            median_flat = np.median(efficiencies[flat_mask])
            return efficiencies / median_flat

        print(f"WARNING: Only {np.sum(flat_mask)} flat sections used to compute efficiency")
        return efficiencies

    def _require_fitted(self):
        if self.bucket_centers is None:
            raise RuntimeError("Model is not fitted. Call .fit(dataset) first.")
