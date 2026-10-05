"""One hierarchical model: population prior plus a shrunk individual offset.

    theta_athlete = theta_population + offset

The offset is the posterior mean of a Bayesian ridge regression on the athlete's
segments, with a zero-mean Gaussian prior (SD ``prior_sd`` per coefficient):

    minimize  Σ_k w_k (r_k − X_k·offset)² / (σ² · inflation)  +  Σ_j offset_j² / sd_j²

where ``r = y − X·theta_population`` after removing each activity's own mean from
``y`` and ``X`` (the within-activity estimator — only drift *inside* a run is
evidence). ``w`` are Huber weights (robust to outlier segments), ``σ`` is a MAD
noise estimate floored at ``noise_floor``, and ``inflation`` accounts for
consecutive segments of one run not being independent.

That single formula is both modes. With no data the offset is exactly zero
(``population_only``); with sparse data the prior dominates and the result stays
close to the population (``partially_personalized``); only strong, consistent
evidence moves it far (``personalized``). Each coefficient's *personal weight* is
the variance reduction ``1 − posterior_var / prior_var`` ∈ [0, 1].

The coefficients are then clipped at 0 (phi must not decrease with exposure) and
any clipping is reported. ``thermal`` is never personalized: no historical stream
carries temperature.

Deliberate pacing (a planned negative split, easing off late) moves speed and HR
together, so the HR-reserve / GAP-speed ratio largely cancels it; what cannot be
separated — an athlete whose *cardiac* drift differs from the population's — is
exactly what the strong prior is for.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.domain.durability.capability import NONE, ReferenceSpeed
from src.domain.durability.config import (
    DurabilityCoefficients,
    DurabilityConfig,
    PersonalizationConfig,
)
from src.domain.durability.segments import IDENTIFIABLE, DurabilitySegment, design_matrix

POPULATION_ONLY = "population_only"
PARTIALLY_PERSONALIZED = "partially_personalized"
PERSONALIZED = "personalized"

# Why the model is (partly) the population's. Translatable under
# ``durability.reason.<key>``.
REASON_NO_HISTORY = "no_history"
REASON_NO_REFERENCE = "no_reference_speed"
REASON_TOO_FEW_RUNS = "too_few_runs"
REASON_WEAK_EVIDENCE = "weak_evidence"
REASON_CLIPPED = "clipped_at_zero"
REASON_SIGNED_OUT = "signed_out"
REASON_FIT_FAILED = "fit_failed"


@dataclass
class OffsetFit:
    offset: np.ndarray
    covariance: np.ndarray
    sigma: float
    robust_weights: np.ndarray
    residuals: np.ndarray


@dataclass
class AthleteDurabilityModel:
    """The durability model to apply for one athlete, and why it is what it is."""

    coefficients: DurabilityCoefficients
    population: DurabilityCoefficients
    confidence: str
    reasons: List[str] = field(default_factory=list)
    reference: ReferenceSpeed = field(default_factory=lambda: ReferenceSpeed(None))
    personal_weight: Dict[str, float] = field(default_factory=dict)
    posterior_sd: Dict[str, float] = field(default_factory=dict)
    n_activities: int = 0
    n_segments: int = 0
    hours: float = 0.0
    excluded: Dict[str, int] = field(default_factory=dict)
    pre_race_exposure: Optional[float] = None
    segments: List[DurabilitySegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "coefficients": self.coefficients.to_dict(),
            "population": self.population.to_dict(),
            "personal_weight": dict(self.personal_weight),
            "posterior_sd": dict(self.posterior_sd),
            "reference_speed_m_per_s": self.reference.speed_m_per_s,
            "reference_source": self.reference.source,
            "n_activities": self.n_activities,
            "n_segments": self.n_segments,
            "hours": self.hours,
            "excluded": dict(self.excluded),
        }

    def to_store(self) -> Dict[str, Any]:
        """Everything :meth:`from_store` needs — segments included, the charts read them."""
        return {
            **self.to_dict(),
            "reference_detail": dict(self.reference.detail),
            "pre_race_exposure": self.pre_race_exposure,
            "segments": [asdict(s) for s in self.segments],
        }

    @staticmethod
    def from_store(raw: Dict[str, Any]) -> "AthleteDurabilityModel":
        """The model :meth:`to_store` saved (stored NaNs come back as ``None``)."""
        speed = raw.get("reference_speed_m_per_s")
        return AthleteDurabilityModel(
            coefficients=DurabilityCoefficients.from_dict(raw["coefficients"]),
            population=DurabilityCoefficients.from_dict(raw["population"]),
            confidence=raw["confidence"],
            reasons=list(raw.get("reasons") or []),
            reference=ReferenceSpeed(
                float(speed) if speed is not None else None,
                raw.get("reference_source") or NONE,
                dict(raw.get("reference_detail") or {}),
            ),
            personal_weight=_floats(raw.get("personal_weight")),
            posterior_sd=_floats(raw.get("posterior_sd")),
            n_activities=int(raw.get("n_activities") or 0),
            n_segments=int(raw.get("n_segments") or 0),
            hours=float(raw.get("hours") or 0.0),
            excluded={k: int(v) for k, v in (raw.get("excluded") or {}).items()},
            pre_race_exposure=raw.get("pre_race_exposure"),
            segments=[_segment(s) for s in raw.get("segments") or []],
        )


def _floats(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    return {k: float(v) if v is not None else float("nan") for k, v in (raw or {}).items()}


def _segment(raw: Dict[str, Any]) -> DurabilitySegment:
    def number(name: str) -> float:
        value = raw.get(name)
        return float(value) if value is not None else float("nan")

    start = raw.get("start_date")
    return DurabilitySegment(
        activity_id=int(raw["activity_id"]),
        start_date=datetime.fromisoformat(start) if start else None,
        elapsed_s=number("elapsed_s"),
        gap_speed_m_per_s=number("gap_speed_m_per_s"),
        heartrate_bpm=number("heartrate_bpm"),
        intensity=number("intensity"),
        exposures=_floats(raw.get("exposures")),
        observed_log_cost=number("observed_log_cost"),
        athlete_key=raw.get("athlete_key") or "",
    )


def population_model(population: DurabilityCoefficients, reasons: List[str],
                     reference: Optional[ReferenceSpeed] = None) -> AthleteDurabilityModel:
    return AthleteDurabilityModel(
        coefficients=population,
        population=population,
        confidence=POPULATION_ONLY,
        reasons=list(reasons),
        reference=reference or ReferenceSpeed(None),
        personal_weight={name: 0.0 for name in IDENTIFIABLE},
    )


def fit_offset(y: np.ndarray, X: np.ndarray, groups: np.ndarray, prior_mean: np.ndarray,
               prior_sd: np.ndarray, config: PersonalizationConfig,
               sample_weight: Optional[np.ndarray] = None) -> OffsetFit:
    """Posterior mean and covariance of the offset from ``prior_mean`` (see module docstring).

    ``sample_weight`` (default 1) scales each segment's information — the offline
    calibration uses it so every athlete counts equally, however much they run.
    """
    y_dm = y - _group_mean(y, groups)
    X_dm = X - _group_mean(X, groups)
    r0 = y_dm - X_dm @ prior_mean
    precision_prior = np.diag(1.0 / np.square(prior_sd))

    base = np.ones_like(r0) if sample_weight is None else np.asarray(sample_weight, float)
    weights = np.ones_like(r0)
    residuals = r0.copy()
    offset = np.zeros(X.shape[1])
    A = precision_prior
    sigma = config.noise_floor
    for _ in range(max(1, config.robust_iterations)):
        sigma = max(config.noise_floor, 1.4826 * float(np.median(np.abs(residuals))))
        scale = base * weights / (sigma ** 2 * config.segment_correlation_inflation)
        A = X_dm.T @ (scale[:, None] * X_dm) + precision_prior
        b = X_dm.T @ (scale * r0)
        offset = np.linalg.solve(A, b)
        residuals = r0 - X_dm @ offset
        z = np.abs(residuals) / sigma
        weights = np.where(z <= config.huber_k, 1.0, config.huber_k / np.maximum(z, 1e-12))
    return OffsetFit(offset=offset, covariance=np.linalg.inv(A), sigma=sigma,
                     robust_weights=weights, residuals=residuals)


def _group_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    counts = np.bincount(groups).astype(float)
    if values.ndim == 1:
        return (np.bincount(groups, weights=values) / counts)[groups]
    means = np.column_stack([
        np.bincount(groups, weights=values[:, j]) / counts for j in range(values.shape[1])
    ])
    return means[groups]


def fit_athlete(
    segments: List[DurabilitySegment],
    reference: ReferenceSpeed,
    config: DurabilityConfig,
    population: Optional[DurabilityCoefficients] = None,
    excluded: Optional[Dict[str, int]] = None,
) -> AthleteDurabilityModel:
    """The athlete's durability model from their segments (empty → population)."""
    population = population or config.population
    settings = config.personalization
    excluded = dict(excluded or {})
    if not reference.known:
        return _with_counts(population_model(population, [REASON_NO_REFERENCE], reference),
                            segments, excluded)
    activities = {s.activity_id for s in segments}
    if not segments:
        return _with_counts(population_model(population, [REASON_NO_HISTORY], reference),
                            segments, excluded)
    if len(activities) < settings.min_activities:
        return _with_counts(population_model(population, [REASON_TOO_FEW_RUNS], reference),
                            segments, excluded)

    y, X, groups = design_matrix(segments, IDENTIFIABLE)
    prior_mean = np.array([population.get(name) for name in IDENTIFIABLE])
    prior_sd = np.array([settings.prior_sd[name] for name in IDENTIFIABLE])
    fit = fit_offset(y, X, groups, prior_mean, prior_sd, settings)

    raw = prior_mean + fit.offset
    values = np.maximum(raw, 0.0)
    posterior_var = np.diag(fit.covariance)
    weight = {name: float(np.clip(1.0 - posterior_var[j] / prior_sd[j] ** 2, 0.0, 1.0))
              for j, name in enumerate(IDENTIFIABLE)}

    reasons: List[str] = []
    if np.any(raw < 0):
        reasons.append(REASON_CLIPPED)
    strong = (weight[IDENTIFIABLE[0]] >= settings.personalized_min_weight
              and len(activities) >= settings.personalized_min_activities)
    if not strong:
        reasons.append(REASON_WEAK_EVIDENCE)

    coefficients = population.with_values(
        dict(zip(IDENTIFIABLE, values)),
        version=f"{population.version}+individual",
        status=population.status,
    )
    model = AthleteDurabilityModel(
        coefficients=coefficients,
        population=population,
        confidence=PERSONALIZED if strong else PARTIALLY_PERSONALIZED,
        reasons=reasons,
        reference=reference,
        personal_weight=weight,
        posterior_sd={name: float(np.sqrt(posterior_var[j]))
                      for j, name in enumerate(IDENTIFIABLE)},
    )
    return _with_counts(model, segments, excluded)


def _with_counts(model: AthleteDurabilityModel, segments: List[DurabilitySegment],
                 excluded: Dict[str, int]) -> AthleteDurabilityModel:
    model.segments = list(segments)
    model.n_segments = len(segments)
    model.n_activities = len({s.activity_id for s in segments})
    # Hours of running behind the evidence: each activity's last segment midpoint.
    model.hours = 0.0
    if segments:
        by_activity: Dict[int, float] = {}
        for s in segments:
            by_activity[s.activity_id] = max(by_activity.get(s.activity_id, 0.0), s.elapsed_s)
        model.hours = sum(by_activity.values()) / 3600.0
    model.excluded = excluded
    return model
