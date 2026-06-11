from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProcessedStream:
    """Per-second derived telemetry for one activity."""
    time: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    elevation_gain: np.ndarray
    heartrate: np.ndarray


@dataclass
class DownsampledDataset:
    """Aggregated (split-level) telemetry across many activities."""
    speed: np.ndarray
    elevation_gain: np.ndarray
    heartrate: np.ndarray
    sport_types: np.ndarray


@dataclass
class GapCurve:
    """A GAP curve: speed adjuster (GAP/speed) as a function of elevation gain."""
    bin_centers: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    counts: np.ndarray
    color: Optional[str] = None
    linestyle: str = "-"
    extra: dict = field(default_factory=dict)
