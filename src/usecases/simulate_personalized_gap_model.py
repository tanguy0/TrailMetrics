from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.preprocessing import DefaultStreamPreprocessor, StreamPreprocessor
from src.domain.gap.reference_curves import balanced_runner, kilian_jornet
from src.domain.gap.smoothing import CurveSmoother, LoessCurveSmoother
from src.domain.gap.xgboost_model import XgboostGapModel
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
    include_reference_curves: bool = True
    verbose: bool = True


@dataclass
class SimulatePersonalizedGapModelOutput:
    dataset: DownsampledDataset
    efficiency_model: EfficiencyGapModel
    xgboost_model: XgboostGapModel
    gap_curves: Dict[str, GapCurve]


class SimulatePersonalizedGapModel(UseCase):
    """Fit both personalized GAP models on the athlete's history and return curves."""

    def __init__(
        self,
        stream_source: ActivityStreamSource,
        preprocessor: Optional[StreamPreprocessor] = None,
        smoother: Optional[CurveSmoother] = None,
    ):
        self.stream_source = stream_source
        self.preprocessor = preprocessor or DefaultStreamPreprocessor()
        self.smoother = smoother or LoessCurveSmoother(bandwidth_fraction=0.4, polyorder=2)

    def execute(
        self, params: Optional[SimulatePersonalizedGapModelInput] = None
    ) -> SimulatePersonalizedGapModelOutput:
        params = params or SimulatePersonalizedGapModelInput()

        streams = self.stream_source.fetch_streams(
            sport_types=params.sport_types,
            from_date=params.from_date,
            to_date=params.to_date,
            max_activities=params.max_activities,
            verbose=params.verbose,
        )

        dataset = self.preprocessor.process_many(
            streams, split_min_time=params.split_min_time, verbose=params.verbose
        )

        efficiency_model = EfficiencyGapModel(
            min_samples_per_bucket=params.efficiency_min_samples_per_bucket
        ).fit(dataset)
        efficiency_curve = self.smoother.smooth(efficiency_model.gap_curve())
        efficiency_curve.color = "purple"

        X, y = self.preprocessor.prepare_calibration_dataset(
            dataset,
            flat_elevation_gain_range=params.flat_elevation_gain_range,
            hr_tolerance=params.hr_tolerance,
        )
        xgboost_model = XgboostGapModel().fit(X, y)
        xgboost_curve = self.smoother.smooth(
            xgboost_model.gap_curve(bin_width=params.xgboost_bin_width)
        )
        xgboost_curve.color = "green"

        gap_curves: Dict[str, GapCurve] = {
            "Efficiency Model": efficiency_curve,
            "XGBoost Model": xgboost_curve,
        }
        if params.include_reference_curves:
            gap_curves["Balanced Runner"] = balanced_runner()
            gap_curves["Kilian Jornet"] = kilian_jornet()

        return SimulatePersonalizedGapModelOutput(
            dataset=dataset,
            efficiency_model=efficiency_model,
            xgboost_model=xgboost_model,
            gap_curves=gap_curves,
        )
