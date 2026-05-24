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
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        xgb_kwargs: Optional[dict] = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.xgb_kwargs = xgb_kwargs or {"objective": "reg:squarederror", "random_state": 42}
        self.model = XGBRegressor(**self.xgb_kwargs)
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XgboostGapModel":
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
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
    ) -> GapCurve:
        X = X if X is not None else self.X_test
        if X is None:
            raise RuntimeError("Model has no test set yet. Call .fit(X, y) or pass X explicitly.")

        if heartrate_range is not None:
            hr_mask = (X[:, 2] >= heartrate_range[0]) & (X[:, 2] <= heartrate_range[1])
            X = X[hr_mask]
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
            if np.any(mask):
                means.append(np.mean(speed_adjusters[mask]))
                stds.append(np.std(speed_adjusters[mask]))
                counts.append(int(np.sum(mask)))
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
