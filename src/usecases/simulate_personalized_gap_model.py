from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.domain.gap import theme
from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.preprocessing import DefaultStreamPreprocessor, StreamPreprocessor
from src.domain.gap.reference_curves import balanced_runner, kilian_jornet
from src.domain.gap.smoothing import CurveSmoother, LoessCurveSmoother
from src.domain.gap.xgboost_model import XgboostGapModel
from src.domain.models.activity import ActivityStream
from src.domain.models.gap import DownsampledDataset, GapCurve
from src.domain.ports.activity_stream_source import ActivityStreamSource
from src.usecases.base import UseCase


@dataclass
class SimulatePersonalizedGapModelInput:
    sport_types: List[str] = field(default_factory=lambda: ["TrailRun"])
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    max_activities: Optional[int] = 1000
    split_min_time: float = 10.0
    flat_elevation_gain_range: Tuple[float, float] = (-10.0, 10.0)
    hr_tolerance: float = 3.0
    efficiency_min_samples_per_bucket: int = 250
    xgboost_bin_width: float = 20.0
    # Which personalized models to fit. At least one should be True for the
    # output to carry any personalized curve.
    fit_efficiency: bool = True
    fit_xgboost: bool = True
    # Reference curves. ``include_reference_curves`` is a master switch kept for
    # backward compatibility; the two specific flags let callers pick either one.
    include_reference_curves: bool = True
    include_balanced_runner: bool = True
    include_kilian: bool = True
    verbose: bool = True
    # When provided, these pre-fetched streams are used instead of hitting the
    # stream source — the date range and sport types above become in-memory
    # filters so the user can re-run analyses without re-fetching.
    streams: Optional[List[ActivityStream]] = None


@dataclass
class SimulatePersonalizedGapModelOutput:
    dataset: DownsampledDataset
    # Either model may be None when the caller chose not to fit it.
    efficiency_model: Optional[EfficiencyGapModel]
    xgboost_model: Optional[XgboostGapModel]
    gap_curves: Dict[str, GapCurve]


class SimulatePersonalizedGapModel(UseCase):
    """Fit both personalized GAP models on the athlete's history and return curves."""

    def __init__(
        self,
        stream_source: Optional[ActivityStreamSource] = None,
        preprocessor: Optional[StreamPreprocessor] = None,
        smoother: Optional[CurveSmoother] = None,
    ):
        # stream_source is optional: when callers pass pre-fetched streams in the
        # input, no source is needed (the app's "fetch once, analyse many" flow).
        self.stream_source = stream_source
        self.preprocessor = preprocessor or DefaultStreamPreprocessor()
        self.smoother = smoother or LoessCurveSmoother(bandwidth_fraction=0.4, polyorder=2)

    def execute(
        self, params: Optional[SimulatePersonalizedGapModelInput] = None
    ) -> SimulatePersonalizedGapModelOutput:
        params = params or SimulatePersonalizedGapModelInput()

        if params.streams is not None:
            streams = self._filter_streams(params.streams, params)
        elif self.stream_source is not None:
            streams = self.stream_source.fetch_streams(
                sport_types=params.sport_types,
                from_date=params.from_date,
                to_date=params.to_date,
                max_activities=params.max_activities,
                verbose=params.verbose,
            )
        else:
            raise ValueError(
                "No streams provided and no stream_source configured to fetch them."
            )

        dataset = self.preprocessor.process_many(
            streams, split_min_time=params.split_min_time, verbose=params.verbose
        )

        gap_curves: Dict[str, GapCurve] = {}

        efficiency_model: Optional[EfficiencyGapModel] = None
        if params.fit_efficiency:
            efficiency_model = EfficiencyGapModel(
                min_samples_per_bucket=params.efficiency_min_samples_per_bucket
            ).fit(dataset)
            efficiency_curve = self.smoother.smooth(efficiency_model.gap_curve())
            efficiency_curve.color = theme.EFFICIENCY
            gap_curves["Efficiency Model"] = efficiency_curve

        xgboost_model: Optional[XgboostGapModel] = None
        if params.fit_xgboost:
            X, y = self.preprocessor.prepare_calibration_dataset(
                dataset,
                flat_elevation_gain_range=params.flat_elevation_gain_range,
                hr_tolerance=params.hr_tolerance,
            )
            xgboost_model = XgboostGapModel().fit(X, y)
            xgboost_curve = self.smoother.smooth(
                xgboost_model.gap_curve(bin_width=params.xgboost_bin_width)
            )
            xgboost_curve.color = theme.AUTO_LEARNING
            gap_curves["XGBoost Model"] = xgboost_curve

        if params.include_reference_curves:
            if params.include_balanced_runner:
                gap_curves["Balanced Runner"] = balanced_runner()
            if params.include_kilian:
                gap_curves["Kilian Jornet"] = kilian_jornet()

        return SimulatePersonalizedGapModelOutput(
            dataset=dataset,
            efficiency_model=efficiency_model,
            xgboost_model=xgboost_model,
            gap_curves=gap_curves,
        )

    @staticmethod
    def _filter_streams(
        streams: List[ActivityStream],
        params: SimulatePersonalizedGapModelInput,
    ) -> List[ActivityStream]:
        """Filter pre-fetched streams by sport type and date range, in memory."""
        out: List[ActivityStream] = []
        for s in streams:
            if params.sport_types and s.sport_type not in params.sport_types:
                continue
            if s.start_date is not None:
                # Strava dates are tz-aware; panel dates are naive. Compare naive.
                started = s.start_date.replace(tzinfo=None)
                if params.from_date is not None and started < params.from_date:
                    continue
                if params.to_date is not None and started > params.to_date:
                    continue
            out.append(s)
        return out
