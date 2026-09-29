"""Every tunable number of the durability model, in one place.

Nothing in the business logic carries a magic constant: exponents, coefficients,
the thermal reference, caps, solver tolerances and personalization thresholds all
live here, with their unit and how they are meant to be calibrated.

**Status of the defaults.** No calibrated population coefficients exist in this
repository yet, so :data:`PLACEHOLDER_POPULATION` is a set of *conservative product
defaults* — the same kind of choice as the published "balanced runner" GAP curve or
the Banister 42/7-day time constants elsewhere in the app. They are chosen so a
well-paced road marathon shows a few percent of extra cost by the finish and a
mountain ultra substantially more, which matches the direction and rough size of
the durability literature, but they are **not validated individual physiology**.
``status="placeholder"`` travels with them into every output so the UI can say so,
and :mod:`src.domain.durability.calibration` is the offline path that replaces them
with fitted, versioned values.

Setting :attr:`DurabilityConfig.enabled` to ``False`` makes the multiplier neutral
(``phi = 1`` everywhere) without touching any caller.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

# The exposures the model sums, in log-cost space. Order is display order.
DURATION = "duration"
SEVERE_INTENSITY = "severe_intensity"
DOWNHILL = "downhill"
THERMAL = "thermal"
PRE_RACE_LOAD = "pre_race_load"

COMPONENTS = (DURATION, SEVERE_INTENSITY, DOWNHILL, THERMAL)

PLACEHOLDER = "placeholder"
FITTED = "fitted"


@dataclass(frozen=True)
class DurabilityCoefficients:
    """The additive log-cost coefficients — the athlete's ``theta``.

    ``logPhi = Σ coefficient · exposure``; every coefficient is ``>= 0`` so that
    ``phi`` can only grow with exposure.

    ======================  ===============  =======================================
    coefficient             unit             exposure it multiplies
    ======================  ===============  =======================================
    ``duration``            1 / h            ``Σ dt[h] · u^p``
    ``severe_intensity``    1 / h            ``Σ dt[h] · max(0, u - 1)^q``
    ``downhill``            1 / km           ``Σ |Δh⁻|[km] · u^r``
    ``thermal``             1 / h            ``Σ dt[h] · heat_stress``
    ``pre_race_load``       dimensionless    ``max(0, ATL / CTL - 1)`` (Banister)
    ======================  ===============  =======================================

    Calibration: ``duration``, ``severe_intensity`` and ``downhill`` are fitted by
    robust within-activity regression of exposure-driven cost drift (see
    :mod:`src.domain.durability.calibration`); ``thermal`` needs historical
    temperature, which no stream carries yet, so it stays a prior;
    ``pre_race_load`` needs race outcomes against prior load.
    """

    duration: float = 0.015
    severe_intensity: float = 0.5
    downhill: float = 0.015
    thermal: float = 0.02
    pre_race_load: float = 0.02
    # Where these numbers come from, carried into every output.
    version: str = "placeholder-2026.09"
    status: str = PLACEHOLDER

    def __post_init__(self):
        for name in COMPONENTS + (PRE_RACE_LOAD,):
            value = getattr(self, name)
            if not (value >= 0):  # also rejects NaN
                raise ValueError(f"durability coefficient {name} must be >= 0, got {value}")

    def get(self, name: str) -> float:
        return float(getattr(self, name))

    def with_values(self, values: Mapping[str, float], **meta: str) -> "DurabilityCoefficients":
        return replace(self, **{k: float(v) for k, v in values.items()}, **meta)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "DurabilityCoefficients":
        known = {k: raw[k] for k in DurabilityCoefficients.__dataclass_fields__ if k in raw}
        return DurabilityCoefficients(**known)


PLACEHOLDER_POPULATION = DurabilityCoefficients()


@dataclass(frozen=True)
class ExposureConfig:
    """Shapes of the exposures — exponents, thermal proxy and numerical caps."""

    # u^p weights ordinary duration by relative intensity: a slow hour tires less
    # than a fast one. Dimensionless.
    duration_exponent: float = 2.0
    # max(0, u - 1)^q — only time above the sustainable reference counts.
    severe_exponent: float = 1.0
    # u^r weights descended metres: fast descending loads the quads more.
    downhill_exponent: float = 1.0
    # Relative intensity is clipped here for numerical safety (a GPS glitch or an
    # implausible target must not explode the severe term). Dimensionless.
    max_intensity: float = 1.5
    # Heat-stress proxy: humidity-adjusted apparent temperature (°C), counted only
    # above this reference, normalized by `thermal_scale_c`, capped at `thermal_cap`.
    thermal_reference_c: float = 15.0
    thermal_scale_c: float = 10.0
    thermal_cap: float = 3.0
    # Used when a race gives temperatures but no humidity. %.
    default_relative_humidity_pct: float = 50.0
    # phi is clamped to [1, max_multiplier] — a numerical guard, reported when hit.
    max_multiplier: float = 2.0


@dataclass(frozen=True)
class CapabilityConfig:
    """How the athlete's reference (critical) speed is derived.

    A best effort of duration ``t`` is converted to a critical-speed estimate with
    the Riegel power law: sustainable speed ∝ ``t^(1 - riegel_exponent)``, anchored
    so the speed sustainable for ``cs_reference_duration_s`` *is* critical speed.
    The same relation turns a race target into the intensity it implies.
    """

    riegel_exponent: float = 1.06
    # CS is, by convention here, the speed sustainable for ~30 min. s.
    cs_reference_duration_s: float = 1800.0
    # Best efforts shorter than this are anaerobic-dominated; longer ones are
    # rarely maximal in training. s.
    min_effort_s: float = 150.0
    max_effort_s: float = 4 * 3600.0
    # Anything faster is a GPS glitch, not a runner. m/s.
    max_plausible_speed: float = 6.5
    # Outlier consensus: a distance's season best is rejected when the critical
    # speed it implies exceeds the median of the *other* distances' season bests by
    # more than this (log ratio; 0.15 ≈ +16 %). Riegel already normalizes duration,
    # so genuine efforts over different distances agree far better than this; a
    # tunnel or GPS jump does not.
    outlier_log_tolerance: float = 0.15
    # The consensus needs this many distances with candidates (the one being
    # checked plus at least two others); with fewer, efforts are used unchecked.
    min_distances_for_consensus: int = 3
    # Sustainable fraction is clamped to this range for extreme durations.
    min_fraction: float = 0.5
    max_fraction: float = 1.2


@dataclass(frozen=True)
class SolverConfig:
    """Fixed-point iteration between elapsed time and ``phi``."""

    # Converged when both the elapsed-time curve (relative to the finish time) and
    # log(phi) move less than this between iterations.
    tolerance: float = 1e-4
    max_iterations: int = 25


@dataclass(frozen=True)
class PersonalizationConfig:
    """What history counts as evidence, and how strongly it may move the prior."""

    # Only this much history is ever read. days.
    lookback_days: int = 365
    # Only runs at least this long carry durability information. s.
    min_activity_moving_s: float = 45 * 60.0
    # Bound on how many streams a fit downloads (most recent first).
    max_activities: int = 150
    # Heart rate settles after a warm-up; segments start after it. s. Same value
    # as the GAP preprocessor's warm-up cut.
    warmup_s: float = 15 * 60.0
    segment_s: float = 300.0
    min_segments_per_activity: int = 6
    # A segment with less moving time than this share of its wall time contains a
    # pause and is dropped.
    min_moving_fraction: float = 0.9
    # HR-to-effort: effort ∝ (HR - hr_rest) — %HRR tracks %VO2 reserve. bpm.
    hr_rest_bpm: float = 50.0
    # Segments whose HR is barely above rest are walking or sensor dropouts. bpm.
    min_hr_reserve_bpm: float = 25.0
    min_hr_coverage: float = 0.9
    # Population cardiac drift: log(HR - HR_rest) rises this much per hour at
    # constant metabolic output (neutral conditions). Subtracted before HR is read
    # as effort. 1/h.
    hr_drift_per_hour: float = 0.03
    # Below this GAP speed a segment is walking, not running. m/s.
    min_gap_speed: float = 1.6
    # Speed variability (coefficient of variation over 60 s chunks) above which a
    # segment is not steady enough for HR to follow it.
    max_segment_speed_cv: float = 0.3
    # Across an activity's segments: above this, the session is intermittent
    # (intervals, fartlek) and — with no workout-structure metadata — dropped.
    max_activity_speed_cv: float = 0.25
    # Mean segment gradient beyond which GAP itself is extrapolated. m/km.
    max_abs_grade: float = 250.0
    min_altitude_coverage: float = 0.9
    # Prior SD of the individual offset, per coefficient (same units as the
    # coefficient). Small = strong pull toward the population.
    prior_sd: Mapping[str, float] = field(default_factory=lambda: {
        DURATION: 0.01, SEVERE_INTENSITY: 0.25, DOWNHILL: 0.01,
    })
    # Consecutive 5-min segments of one run are not independent; each segment's
    # information is divided by this. Dimensionless.
    segment_correlation_inflation: float = 6.0
    noise_floor: float = 0.01
    huber_k: float = 1.5
    robust_iterations: int = 5
    # Confidence thresholds.
    min_activities: int = 3
    personalized_min_activities: int = 8
    personalized_min_weight: float = 0.6


@dataclass(frozen=True)
class PreRaceLoadConfig:
    """Optional pre-race load from recent Banister training load. Off by default.

    The race plan is for a future day, so today's load is at best a proxy for
    race-day freshness; and a *constant* multiplier cancels out of a target-time
    plan. It is implemented, explicitly named, and disabled unless turned on.
    """

    enabled: bool = False
    min_rated_activities: int = 20


@dataclass(frozen=True)
class DurabilityConfig:
    enabled: bool = True
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    capability: CapabilityConfig = field(default_factory=CapabilityConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    pre_race: PreRaceLoadConfig = field(default_factory=PreRaceLoadConfig)
    population: DurabilityCoefficients = PLACEHOLDER_POPULATION
    # Optional strata over athlete fields the app already has (e.g. an age band),
    # each overriding the population coefficients. Empty until calibration fills it.
    population_strata: Mapping[str, DurabilityCoefficients] = field(default_factory=dict)

    def population_for(self, stratum: Optional[str] = None) -> DurabilityCoefficients:
        if stratum and stratum in self.population_strata:
            return self.population_strata[stratum]
        return self.population


DEFAULT_CONFIG = DurabilityConfig()
