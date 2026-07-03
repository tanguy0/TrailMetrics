import time as time_module
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
from stravalib import Client

from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_stream_source import ActivityStreamSource


def _to_float(value) -> Optional[float]:
    """Coerce a Strava summary field (plain number or stravalib Quantity) to float.

    stravalib may expose magnitudes as pint/quantity-like objects with a
    ``.magnitude``; fall back to ``float(value)`` otherwise. ``None`` stays None.
    """
    if value is None:
        return None
    value = getattr(value, "magnitude", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_seconds(value) -> Optional[float]:
    """Coerce a duration field (``timedelta`` or number of seconds) to seconds."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    return _to_float(value)


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
        require_heartrate: bool = True,
    ) -> List[dict]:
        results = []
        for act in self.client.get_activities(after=from_date, before=to_date):
            if act.sport_type not in sport_types:
                continue
            if require_heartrate and not act.has_heartrate:
                continue
            results.append(
                {
                    "id": act.id,
                    "sport_type": act.sport_type,
                    "start_date": act.start_date,
                    # Activity-level totals, kept so we can still surface an
                    # activity whose per-second streams aren't available.
                    "distance_m": _to_float(getattr(act, "distance", None)),
                    "moving_time_s": _to_seconds(getattr(act, "moving_time", None)),
                    "elevation_gain_m": _to_float(
                        getattr(act, "total_elevation_gain", None)
                    ),
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
        require_heartrate: bool = True,
        keep_streamless: bool = False,
    ) -> List[ActivityStream]:
        activities = self.list_activities(
            sport_types,
            from_date=from_date,
            to_date=to_date,
            require_heartrate=require_heartrate,
        )
        if max_activities is not None:
            activities = activities[:max_activities]

        total = len(activities)
        streams: List[ActivityStream] = []
        for i, act in enumerate(activities):
            stream: Optional[ActivityStream] = None
            try:
                raw = self._fetch_raw_stream(act["id"], resolution="high")
                stream = self._to_activity_stream(
                    act["id"],
                    act["sport_type"],
                    raw,
                    start_date=act.get("start_date"),
                )
            except Exception as e:
                if verbose:
                    print(f"Error getting streams for activity {act['id']}: {e}")
                if keep_streamless:
                    stream = self._summary_only_stream(act)
            if stream is not None:
                streams.append(stream)
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
        # time + distance are required (they define the activity); altitude and
        # heartrate are optional — when absent (no GPS / no HR sensor) they're
        # filled with NaN so the activity still counts wherever they're not
        # needed (distance, records), and downstream NaN-guards skip the rest.
        time = np.array(raw["time"].data, dtype=float)
        distance = np.array(raw["distance"].data, dtype=float)
        n = time.size
        altitude = (
            np.array(raw["altitude"].data, dtype=float)
            if "altitude" in raw
            else np.full(n, np.nan)
        )
        heartrate = (
            np.array(raw["heartrate"].data, dtype=float)
            if "heartrate" in raw
            else np.full(n, np.nan)
        )
        return ActivityStream(
            activity_id=activity_id,
            sport_type=sport_type,
            time=time,
            distance=distance,
            altitude=altitude,
            heartrate=heartrate,
            start_date=start_date,
        )

    @staticmethod
    def _summary_only_stream(act: dict) -> ActivityStream:
        """A streamless :class:`ActivityStream` carrying only activity-level totals."""
        empty = np.array([], dtype=float)
        return ActivityStream(
            activity_id=act["id"],
            sport_type=act["sport_type"],
            time=empty,
            distance=empty,
            altitude=empty,
            heartrate=empty,
            start_date=act.get("start_date"),
            has_streams=False,
            summary_distance_m=act.get("distance_m"),
            summary_moving_time_s=act.get("moving_time_s"),
            summary_elevation_gain_m=act.get("elevation_gain_m"),
        )
