from dataclasses import dataclass

import numpy as np


@dataclass
class ActivityStream:
    """Raw per-second telemetry for a single activity."""
    activity_id: int
    sport_type: str
    time: np.ndarray
    distance: np.ndarray
    altitude: np.ndarray
    heartrate: np.ndarray
