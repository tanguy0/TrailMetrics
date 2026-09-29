"""Offline calibration of the durability coefficients from many athletes' segments.

    python -m api.calibrate_durability --output durability_population.json

This module is the pure part — split, fit, evaluate, (de)serialize — and works on
:class:`~src.domain.durability.segments.DurabilitySegment` rows only, so it can be
driven from the production database (``api/calibrate_durability.py``), a notebook,
or a test.

**Protocol.**

1. *Split by athlete*, never by segment: a stable hash of ``athlete_key`` sends
   each athlete wholly to train or validation, so validation measures how the
   model transfers to runners it has never seen.
2. *Population fit* on the training athletes: the same robust (Huber),
   within-activity ridge regression as the individual fit, but anchored to the
   current coefficients with a *weak* prior (``population_prior_sd``), and with
   segment weights that give every athlete equal total weight. Negative values are
   clipped at 0.
3. *Evaluation* on the validation athletes, per activity, against three
   predictors of within-activity cost drift: none (``phi = 1``), the population,
   and the personalized model **fitted on the athlete's other activities only** —
   leave-one-activity-out, so no activity ever informs its own prediction.
   Reported: RMSE and MAE of drift in log-cost (≈ fraction of cost), weighted by
   segments.
4. *Versioned output*: a JSON with the coefficients, ``status="fitted"``, a
   version string, the sample sizes and the metrics. Only identifiable
   coefficients (duration, severe intensity, downhill) are fitted; ``thermal`` and
   ``pre_race_load`` keep their prior values until the data to fit them exist
   (historical temperature; race outcomes against prior load).
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.domain.durability.capability import ReferenceSpeed
from src.domain.durability.config import FITTED, DurabilityCoefficients, DurabilityConfig
from src.domain.durability.personalization import fit_athlete, fit_offset
from src.domain.durability.segments import IDENTIFIABLE, DurabilitySegment, design_matrix


@dataclass(frozen=True)
class CalibrationSettings:
    validation_fraction: float = 0.2
    # Weak prior around the current coefficients — the population fit should be
    # driven by data, the prior only keeps it defined when data are scarce.
    population_prior_sd: Mapping[str, float] = field(default_factory=lambda: {
        "duration": 0.05, "severe_intensity": 1.0, "downhill": 0.05,
    })
    min_train_athletes: int = 5


@dataclass
class CalibrationResult:
    coefficients: DurabilityCoefficients
    metrics: Dict[str, Dict[str, float]]
    train_athletes: int
    validation_athletes: int
    train_segments: int


def split_athletes(keys: Iterable[str], validation_fraction: float
                   ) -> Tuple[List[str], List[str]]:
    """Deterministic athlete-level split by a hash of the key."""
    train, validation = [], []
    for key in sorted(set(keys)):
        digest = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (validation if digest < validation_fraction else train).append(key)
    return train, validation


def fit_population(segments: Sequence[DurabilitySegment], base: DurabilityCoefficients,
                   config: DurabilityConfig, settings: CalibrationSettings
                   ) -> DurabilityCoefficients:
    y, X, groups = design_matrix(list(segments), IDENTIFIABLE)
    athletes = np.array([s.athlete_key for s in segments])
    _, athlete_index, counts = np.unique(athletes, return_inverse=True, return_counts=True)
    weight = (len(segments) / len(counts)) / counts[athlete_index]
    prior_mean = np.array([base.get(n) for n in IDENTIFIABLE])
    prior_sd = np.array([settings.population_prior_sd[n] for n in IDENTIFIABLE])
    fit = fit_offset(y, X, groups, prior_mean, prior_sd, config.personalization,
                     sample_weight=weight)
    values = np.maximum(prior_mean + fit.offset, 0.0)
    return base.with_values(dict(zip(IDENTIFIABLE, values)))


def evaluate(segments: Sequence[DurabilitySegment], population: DurabilityCoefficients,
             config: DurabilityConfig) -> Dict[str, Dict[str, float]]:
    """Held-out drift error of the none / population / personalized predictors."""
    config = replace(config, population=population)
    errors: Dict[str, List[np.ndarray]] = {"none": [], "population": [], "personalized": []}
    by_athlete: Dict[str, List[DurabilitySegment]] = {}
    for s in segments:
        by_athlete.setdefault(s.athlete_key, []).append(s)

    for athlete_segments in by_athlete.values():
        activities = sorted({s.activity_id for s in athlete_segments})
        for target in activities:
            held_out = [s for s in athlete_segments if s.activity_id == target]
            others = [s for s in athlete_segments if s.activity_id != target]
            # Reference speed is irrelevant to the fit itself (exposures are
            # already computed); a known placeholder lets fit_athlete run.
            personal = fit_athlete(others, ReferenceSpeed(1.0, "calibration"), config)
            y, X, _ = design_matrix(held_out, IDENTIFIABLE)
            observed = y - y.mean()
            for name, coefficients in (("population", population),
                                       ("personalized", personal.coefficients)):
                theta = np.array([coefficients.get(n) for n in IDENTIFIABLE])
                predicted = X @ theta
                errors[name].append(observed - (predicted - predicted.mean()))
            errors["none"].append(observed)

    metrics: Dict[str, Dict[str, float]] = {}
    for name, chunks in errors.items():
        if not chunks:
            continue
        e = np.concatenate(chunks)
        metrics[name] = {
            "rmse_log_cost": float(np.sqrt(np.mean(e ** 2))),
            "mae_log_cost": float(np.mean(np.abs(e))),
            "segments": int(e.size),
        }
    return metrics


def calibrate(segments: Sequence[DurabilitySegment], config: DurabilityConfig,
              settings: CalibrationSettings = CalibrationSettings(),
              version: str = "") -> CalibrationResult:
    train_keys, validation_keys = split_athletes(
        (s.athlete_key for s in segments), settings.validation_fraction
    )
    if len(train_keys) < settings.min_train_athletes:
        raise ValueError(
            f"need at least {settings.min_train_athletes} training athletes, "
            f"got {len(train_keys)}"
        )
    train = [s for s in segments if s.athlete_key in set(train_keys)]
    validation = [s for s in segments if s.athlete_key in set(validation_keys)]
    stamp = version or datetime.now(timezone.utc).strftime("fitted-%Y.%m.%d")
    fitted = fit_population(train, config.population, config, settings)
    fitted = replace(fitted, version=stamp, status=FITTED)
    return CalibrationResult(
        coefficients=fitted,
        metrics=evaluate(validation, fitted, config) if validation else {},
        train_athletes=len(train_keys),
        validation_athletes=len(validation_keys),
        train_segments=len(train),
    )


def to_json(result: CalibrationResult) -> str:
    return json.dumps({
        "coefficients": result.coefficients.to_dict(),
        "metrics": result.metrics,
        "train_athletes": result.train_athletes,
        "validation_athletes": result.validation_athletes,
        "train_segments": result.train_segments,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


def load_coefficients(raw: str) -> DurabilityCoefficients:
    """The coefficients of a calibration JSON written by :func:`to_json`."""
    return DurabilityCoefficients.from_dict(json.loads(raw)["coefficients"])
