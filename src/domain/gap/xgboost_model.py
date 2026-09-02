from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.domain.gap.base import GapModel
from src.domain.models.gap import GapCurve


class XgboostGapModel(GapModel):
    """
    XGBoost regressor that predicts flat-equivalent speed (GAP) given
    [instant_speed, elevation_gain, heartrate].

    Rows are **weighted**: one row per split, carrying how many similar-HR flat
    splits backed its target (see
    :meth:`~src.domain.gap.preprocessing.DefaultStreamPreprocessor.prepare_calibration_dataset`
    for why that is the same fit as one row per pair, and not an approximation).
    A caller passing unweighted rows still works — the weights default to 1.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        xgb_kwargs: Optional[dict] = None,
        retain_training_data: bool = False,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.xgb_kwargs = xgb_kwargs or {"objective": "reg:squarederror", "random_state": 42}
        self.model = XGBRegressor(**self.xgb_kwargs)
        # The held-out split is kept because `gap_curve` uses it as the sample grid
        # it bins; the training half is not, and a fitted model is memoized for the
        # life of the process, so holding it would pin the whole calibration set per
        # (athlete, group). `retain_training_data` keeps it for the notebooks, which
        # plot train-vs-test calibration.
        self.retain_training_data = retain_training_data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.w_train: Optional[np.ndarray] = None
        self.w_test: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XgboostGapModel":
        weights = (
            np.ones(len(X), dtype=float) if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train, sample_weight=w_train)
        self.X_test, self.w_test = X_test, w_test
        if self.retain_training_data:
            self.X_train, self.y_train = X_train, y_train
            self.y_test, self.w_train = y_test, w_train
        return self

    def predict_gap(
        self,
        speed: np.ndarray,
        elevation_gain: np.ndarray,
        heartrate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if heartrate is None:
            raise ValueError("XgboostGapModel requires heartrate as a feature.")
        X = np.stack([np.asarray(speed), np.asarray(elevation_gain), np.asarray(heartrate)], axis=1)
        return self.model.predict(X)

    def gap_curve(
        self,
        X: Optional[np.ndarray] = None,
        bin_width: float = 20.0,
        heartrate_range: Optional[Tuple[float, float]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> GapCurve:
        """Mean speed adjuster per gradient bin, over the held-out splits.

        Every statistic here is **weighted** by the row weights, so the curve is
        the one the equivalent one-row-per-pair dataset would have produced: a
        split matched by a thousand flat splits should count for a thousand times
        as much in its bin as one matched by a single flat split.
        """
        if X is None:
            X = self.X_test
            if weights is None:
                weights = self.w_test
        if X is None:
            raise RuntimeError("Model has no test set yet. Call .fit(X, y) or pass X explicitly.")
        w = (
            np.ones(len(X), dtype=float) if weights is None
            else np.asarray(weights, dtype=float)
        )

        if heartrate_range is not None:
            hr_mask = (X[:, 2] >= heartrate_range[0]) & (X[:, 2] <= heartrate_range[1])
            X, w = X[hr_mask], w[hr_mask]
            if len(X) == 0:
                raise ValueError(f"No data points in heart rate range {heartrate_range}")

        gaps = self.model.predict(X)
        speed_adjusters = gaps / X[:, 0]

        min_elev = np.floor(np.min(X[:, 1]) / bin_width) * bin_width
        max_elev = np.ceil(np.max(X[:, 1]) / bin_width) * bin_width
        bin_edges = np.arange(min_elev, max_elev + bin_width, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        means, stds, counts = [], [], []
        for i in range(len(bin_edges) - 1):
            mask = (X[:, 1] >= bin_edges[i]) & (X[:, 1] < bin_edges[i + 1])
            bin_weights = w[mask]
            total = bin_weights.sum()
            if total > 0:
                values = speed_adjusters[mask]
                mean = float(np.dot(bin_weights, values) / total)
                variance = float(np.dot(bin_weights, (values - mean) ** 2) / total)
                means.append(mean)
                stds.append(np.sqrt(max(variance, 0.0)))
                counts.append(int(round(total)))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                counts.append(0)

        return GapCurve(
            bin_centers=bin_centers,
            means=np.array(means),
            stds=np.array(stds),
            counts=np.array(counts),
        )
