"""Plan a race: GPX + target time + aid stations → a pace profile.

The plan is always paced on the athlete's own GAP curve — the whole point is
planning on *your* data, not an average runner's — and nobody is asked which curve
to use. The published references (balanced runner, Kilian) are comparisons, never
a choice: the balanced runner only stands in when there is no personal curve (a
visitor, or a fit that failed, said in a note).

Fitting is the expensive part and is kept out of :class:`PlanRace` entirely: the
caller passes a ``personal_curve`` function, so this use case stays storage-free
and the API decides how to cache (see ``api/routers/race_plan.py``).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.domain.dataset.sport import RUNNING_SPORT_TYPES
from src.domain.durability.config import DEFAULT_CONFIG, DurabilityConfig
from src.domain.durability.model import RaceWeather
from src.domain.durability.personalization import (
    REASON_FIT_FAILED,
    REASON_SIGNED_OUT,
    AthleteDurabilityModel,
    population_model,
)
from src.domain.durability.solver import RouteDurability
from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.preprocessing import DefaultStreamPreprocessor
from src.domain.gap.reference_curves import balanced_runner
from src.domain.gap.smoothing import LoessCurveSmoother
from src.domain.models.gap import GapCurve
from src.domain.ports.activity_data import ActivityDataSource
from src.domain.race_plan.gpx import parse_gpx
from src.domain.race_plan.output import build_outputs, summary
from src.domain.race_plan.planner import adjuster, build_course, plan_race, usable_curve
from src.translations import translate
from src.usecases.base import UseCase

logger = logging.getLogger(__name__)

PERSONAL_EFFICIENCY = "personal_efficiency"
BALANCED_RUNNER = "balanced_runner"

# The personal curve plans are paced on: the efficiency model only. The
# auto-learning one can still be fitted (``fit_personal_curve``) but is never used.
PERSONAL_CURVES = (PERSONAL_EFFICIENCY,)
CURVE_LABEL_KEYS = {
    PERSONAL_EFFICIENCY: "race_plan.curve.personal_efficiency",
    BALANCED_RUNNER: "gap.refs.balanced",
}

# ``model key → (curve, reason_key)``; a ``None`` curve carries why.
PersonalCurve = Callable[[str], Tuple[Optional[GapCurve], Optional[str]]]
# The signed-in athlete's durability model, fitted (and cached) by the caller.
DurabilityProvider = Callable[[], AthleteDurabilityModel]


@dataclass
class PlanRaceInput:
    gpx: bytes
    target_time_s: float
    aid_stations_km: Sequence[float] = ()
    aid_station_names: Sequence[str] = ()
    start_clock_s: Optional[float] = None
    lang: str = "en"
    # Durability: the cost drift of a long effort, on by default.
    durability: bool = True
    weather: RaceWeather = field(default_factory=RaceWeather)


@dataclass
class PlanRaceOutput:
    curve: str
    curve_label: str
    personalized: bool
    summary: Dict[str, float]
    outputs: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve": self.curve,
            "curve_label": self.curve_label,
            "personalized": self.personalized,
            "summary": self.summary,
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "notes": self.notes,
        }


class PlanRace(UseCase):
    def __init__(
        self,
        personal_curve: Optional[PersonalCurve] = None,
        durability_model: Optional[DurabilityProvider] = None,
        durability_config: DurabilityConfig = DEFAULT_CONFIG,
    ):
        self.personal_curve = personal_curve
        self.durability_model = durability_model
        self.durability_config = durability_config

    def execute(self, params: PlanRaceInput) -> PlanRaceOutput:
        lang = params.lang
        notes: List[str] = []

        curve, key = self._curve(lang, notes)

        course = build_course(parse_gpx(params.gpx))
        reference = balanced_runner() if key in PERSONAL_CURVES else None
        model = self._durability(params) if params.durability else None
        plan = plan_race(
            course,
            params.target_time_s,
            adjuster(curve, reference),
            params.aid_stations_km,
            params.aid_station_names,
            durability=RouteDurability(
                coefficients=model.coefficients,
                config=self.durability_config,
                athlete_reference=model.reference,
                weather=params.weather,
                pre_race_exposure=model.pre_race_exposure,
            ) if model is not None else None,
        )
        return PlanRaceOutput(
            curve=key,
            curve_label=translate(CURVE_LABEL_KEYS[key], lang),
            personalized=key in PERSONAL_CURVES,
            summary=summary(plan, model),
            outputs=build_outputs(plan, lang, params.start_clock_s, model),
            notes=notes,
        )

    def _durability(self, params: PlanRaceInput) -> AthleteDurabilityModel:
        """The athlete's model when signed in, else the population model.

        A failing fit must not cost the athlete their plan: it degrades to the
        population model, which is what a visitor gets anyway.
        """
        population = self.durability_config.population
        if self.durability_model is None:
            return population_model(population, [REASON_SIGNED_OUT])
        try:
            return self.durability_model()
        except Exception as error:
            logger.warning("durability fit failed, using the population model: %s", error)
            return population_model(population, [REASON_FIT_FAILED])

    def _curve(self, lang: str, notes: List[str]) -> Tuple[GapCurve, str]:
        """The athlete's curve; the balanced runner without one (a note says why)."""
        if self.personal_curve is not None:
            curve, reason = self.personal_curve(PERSONAL_EFFICIENCY)
            if usable_curve(curve):
                return curve, PERSONAL_EFFICIENCY
            notes.append(translate("race_plan.note.personal_fallback", lang).format(
                reason=translate(reason or "race_plan.reason.not_enough_data", lang),
            ))
        return balanced_runner(), BALANCED_RUNNER


# --- Personal curves --------------------------------------------------------

# The most recent runs to fit on. Enough splits for a stable curve many times
# over, and a bound on how many streams a first plan has to download.
MAX_ACTIVITIES = 150
SPLIT_MIN_TIME = 10.0
EFFICIENCY_MIN_SAMPLES = 250

_PREPROCESSOR = DefaultStreamPreprocessor()
_SMOOTHER = LoessCurveSmoother(bandwidth_fraction=0.4, polyorder=2)


def running_activity_ids(data: ActivityDataSource) -> Tuple[int, ...]:
    """The runs a personal curve is fitted on — the key its cache is keyed by."""
    runs = [
        s for s in data.summaries()
        if s.has_streams and s.sport_type in RUNNING_SPORT_TYPES
    ]
    runs.sort(key=lambda s: s.start_date)
    return tuple(s.activity_id for s in runs[-MAX_ACTIVITIES:])


def fit_personal_curve(
    data: ActivityDataSource, activity_ids: Sequence[int], model: str,
    memo: Optional[Dict[Any, Any]] = None,
) -> Tuple[Optional[GapCurve], Optional[str]]:
    """Fit one personal GAP curve on ``activity_ids``: ``(curve, reason_key)``.

    ``memo`` shares the pooled splits between the two models, so switching curve
    in the selector downloads and preprocesses the history once, not twice.
    """
    if not activity_ids:
        return None, "race_plan.reason.no_runs"
    memo = memo if memo is not None else {}
    dataset_key = ("race_plan_dataset", tuple(activity_ids))
    if dataset_key not in memo:
        streams = [s for s in (data.stream(i) for i in activity_ids) if s is not None]
        try:
            memo[dataset_key] = _PREPROCESSOR.process_many(
                streams, split_min_time=SPLIT_MIN_TIME, verbose=False
            ) if streams else None
        except (ValueError, IndexError):
            memo[dataset_key] = None
    dataset = memo[dataset_key]
    if dataset is None or dataset.speed.size < 2 * EFFICIENCY_MIN_SAMPLES:
        return None, "race_plan.reason.not_enough_data"

    try:
        if model == PERSONAL_EFFICIENCY:
            fitted = EfficiencyGapModel(min_samples_per_bucket=EFFICIENCY_MIN_SAMPLES).fit(dataset)
            return _SMOOTHER.smooth(fitted.gap_curve()), None

        from src.domain.gap.xgboost_model import XgboostGapModel

        features, targets, weights = _PREPROCESSOR.prepare_calibration_dataset(dataset)
        if features.size == 0:
            return None, "gap.reason.no_calibration"
        fitted = XgboostGapModel().fit(features, targets, sample_weight=weights)
        return _SMOOTHER.smooth(fitted.gap_curve(bin_width=20.0)), None
    except Exception:
        return None, "race_plan.reason.fit_failed"
