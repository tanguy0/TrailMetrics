import time as time_module
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
from stravalib import Client

from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_stream_source import ActivityStreamSource


class StravaClient(ActivityStreamSource):
    """Strava implementation of the ActivityStreamSource port.

    Takes a pre-authenticated stravalib.Client. The OAuth dance stays
    in the caller (notebook / streamlit app / future API layer).
    """

    DEFAULT_STREAM_TYPES = ["time", "distance", "altitude", "heartrate"]

    def __init__(self, client: Client, throttle_seconds: float = 0.1):
        self.client = client
        self.throttle_seconds = throttle_seconds

    def list_activities(
        self,
        sport_types: List[str],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[dict]:
        results = []
        for act in self.client.get_activities(after=from_date, before=to_date):
            if act.has_heartrate and act.sport_type in sport_types:
                results.append(
                    {
                        "id": act.id,
                        "sport_type": act.sport_type,
                        "start_date": act.start_date,
                    }
                )
        return results

    def fetch_streams(
        self,
        sport_types: List[str],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_activities: Optional[int] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ActivityStream]:
        activities = self.list_activities(sport_types, from_date=from_date, to_date=to_date)
        if max_activities is not None:
            activities = activities[:max_activities]

        total = len(activities)
        streams: List[ActivityStream] = []
        for i, act in enumerate(activities):
            try:
                stream = self._fetch_raw_stream(act["id"], resolution="high")
                streams.append(
                    self._to_activity_stream(
                        act["id"],
                        act["sport_type"],
                        stream,
                        start_date=act.get("start_date"),
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"Error getting streams for activity {act['id']}: {e}")
                continue
            finally:
                if progress_callback is not None:
                    progress_callback(i + 1, total)
            time_module.sleep(self.throttle_seconds)
        return streams

    def fetch_single_stream(self, activity_id: int, resolution: str = "high") -> ActivityStream:
        stream = self._fetch_raw_stream(activity_id, resolution=resolution)
        # Sport type not strictly needed here, default to TrailRun for the apply-to-one-activity flow.
        return self._to_activity_stream(activity_id, "TrailRun", stream)

    def _fetch_raw_stream(self, activity_id: int, resolution: str) -> dict:
        return self.client.get_activity_streams(
            activity_id,
            types=self.DEFAULT_STREAM_TYPES,
            resolution=resolution,
            series_type="time",
        )

    @staticmethod
    def _to_activity_stream(
        activity_id: int,
        sport_type: str,
        raw: dict,
        start_date: Optional[datetime] = None,
    ) -> ActivityStream:
        return ActivityStream(
            activity_id=activity_id,
            sport_type=sport_type,
            time=np.array(raw["time"].data),
            distance=np.array(raw["distance"].data),
            altitude=np.array(raw["altitude"].data),
            heartrate=np.array(raw["heartrate"].data),
            start_date=start_date,
        )
