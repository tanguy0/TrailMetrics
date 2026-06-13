"""Compare several races side by side from pre-fetched activity streams.

Runs entirely on streams already loaded in memory (the app's "fetch once,
analyse many" flow) — it never hits Strava. Produces per-race summary metrics
and aligned time series for the comparison table and evolution plots.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.models.activity import ActivityStream
from src.domain.races.metrics import RaceMetrics, RaceSeries, compute_race
from src.domain.races.smoothing import SmoothingParams
from src.usecases.base import UseCase

# Mirrors the selector cap on the Race Comparator page.
MAX_RACES = 4


@dataclass
class CompareRacesInput:
    streams: List[ActivityStream]
    # Optional display labels, aligned with ``streams``. Falls back to the
    # activity id when missing.
    labels: Optional[List[str]] = None
    # Runner weight (kg); enables the power metrics. Power stays unavailable
    # when omitted.
    mass_kg: Optional[float] = None
    # Per-signal smoothing config; falls back to defaults when omitted.
    smoothing: Optional[SmoothingParams] = None


@dataclass
class CompareRacesOutput:
    metrics: List[RaceMetrics] = field(default_factory=list)
    series: List[RaceSeries] = field(default_factory=list)


class CompareRaces(UseCase):
    """Compute comparison metrics and series for up to ``MAX_RACES`` races."""

    def execute(self, params: CompareRacesInput) -> CompareRacesOutput:
        if not params.streams:
            raise ValueError("No races to compare — select at least one workout.")
        if len(params.streams) > MAX_RACES:
            raise ValueError(
                f"Can compare at most {MAX_RACES} races at once "
                f"(got {len(params.streams)})."
            )

        labels = params.labels or [None] * len(params.streams)

        metrics: List[RaceMetrics] = []
        series: List[RaceSeries] = []
        for stream, label in zip(params.streams, labels):
            label = label or f"Activity {stream.activity_id}"
            race_metrics, race_series = compute_race(
                stream,
                label,
                mass_kg=params.mass_kg,
                smoothing=params.smoothing,
            )
            metrics.append(race_metrics)
            series.append(race_series)

        return CompareRacesOutput(metrics=metrics, series=series)
