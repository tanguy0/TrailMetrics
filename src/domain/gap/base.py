from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.domain.models.gap import GapCurve


class GapModel(ABC):
    """A model that predicts gradient-adjusted pace for trail data."""

    @abstractmethod
    def fit(self, *args, **kwargs) -> "GapModel":
        ...

    @abstractmethod
    def predict_gap(
        self,
        speed: np.ndarray,
        elevation_gain: np.ndarray,
        heartrate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def gap_curve(self, *args, **kwargs) -> GapCurve:
        ...
