from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from src.domain.models.activity import ActivityStream
from src.domain.ports.activity_stream_source import ActivityStreamSource
from src.usecases.base import UseCase


@dataclass
class FetchAthleteHistoryInput:
    """Fetch all history back to ``from_date``; analyses filter it later, in memory.

    ``from_date`` is the oldest date to fetch initially (the "max date in the
    past"). ``max_activities`` is left ``None`` by default: no cap.
    """
    sport_types: List[str] = field(default_factory=lambda: ["TrailRun", "Run"])
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    max_activities: Optional[int] = None
    verbose: bool = False


@dataclass
class FetchAthleteHistoryOutput:
    streams: List[ActivityStream]
    oldest_date: Optional[datetime]
    newest_date: Optional[datetime]

    @property
    def activity_count(self) -> int:
        return len(self.streams)


class FetchAthleteHistory(UseCase):
    """Fetch the athlete's raw activity streams once, to be cached and reused.

    The returned streams are kept in memory by the caller (the Streamlit app),
    so analyses can re-filter and re-fit without ever hitting Strava again.
    """

    def __init__(self, stream_source: ActivityStreamSource):
        self.stream_source = stream_source

    def execute(
        self,
        params: Optional[FetchAthleteHistoryInput] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> FetchAthleteHistoryOutput:
        params = params or FetchAthleteHistoryInput()

        streams = self.stream_source.fetch_streams(
            sport_types=params.sport_types,
            from_date=params.from_date,
            to_date=params.to_date,
            max_activities=params.max_activities,
            verbose=params.verbose,
            progress_callback=progress_callback,
        )

        dates = [s.start_date for s in streams if s.start_date is not None]
        oldest = min(dates) if dates else None
        newest = max(dates) if dates else None

        return FetchAthleteHistoryOutput(
            streams=streams, oldest_date=oldest, newest_date=newest
        )
