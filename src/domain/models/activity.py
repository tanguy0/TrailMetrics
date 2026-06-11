from dataclasses import dataclass
from datetime import datetime
from typing import Optional

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
    # When the activity took place. Optional so older callers/mocks stay valid;
    # used by the app to filter pre-fetched history by date without re-fetching.
    start_date: Optional[datetime] = None
