from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional

from src.domain.models.activity import ActivityStream


class ActivityStreamSource(ABC):
    """Port for any source of raw activity streams (Strava, file, mock, ...)."""

    @abstractmethod
    def list_activities(
        self,
        sport_types: List[str],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        require_heartrate: bool = True,
    ) -> List[dict]:
        """Return a list of activities filtered by sport type and (optionally) date range.

        When ``require_heartrate`` is ``False``, activities without heart-rate
        data are kept too (e.g. for distance/elevation-only analyses).
        """

    @abstractmethod
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
        """Fetch raw streams for activities matching the given sport types and date range.

        ``progress_callback(done, total)`` is invoked after each activity's
        stream is fetched, so callers (e.g. a UI) can render a progress bar.
        ``require_heartrate`` keeps/drops activities without HR. When
        ``keep_streamless`` is ``True``, activities whose streams can't be
        fetched are still returned as summary-only :class:`ActivityStream`s
        (``has_streams=False``) carrying their activity-level totals.
        """

    @abstractmethod
    def fetch_single_stream(self, activity_id: int) -> ActivityStream:
        """Fetch one stream by activity id (used for the 'apply to a real run' flow)."""
