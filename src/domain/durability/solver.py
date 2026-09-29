"""Feeding ``phi`` back into the race planner, and iterating to a fixed point.

**Why pace scales with phi, exactly.** The planner's cost model is the project's
``P = m·v·Cr·a(g)`` — linear in speed, with the GAP adjuster ``a(g)`` pricing the
terrain. Durability multiplies the required cost:

    P_required(v, point, t) = phi(t) · m · v · Cr · a(g)

Holding the athlete's available power ``P_avail`` (the constant-effort
assumption the planner already makes) and solving for speed gives

    v = P_avail / (phi · m · Cr · a(g))   ⇒   pace = P · phi(t) · a(g)

where ``P`` is the fresh GAP pace. That is the existing solver re-run on the
adjusted cost, not a flat-course ``speed / phi`` shortcut: ``a(g)`` still sets the
climb/descent ratio at every point. The finish time stays linear in ``P``,

    T(P) = P · Σ ds_i · a(g_i) · phi_i

so the target-time planner still *solves* ``P`` rather than searching for it.
Keeping the target is the planner's choice; this module only supplies ``phi``.

**Why iterate.** ``phi`` depends on elapsed time (and on intensity), which depends
on pace, which depends on ``phi``. Starting from the fresh plan, each round
evaluates ``phi`` on the current elapsed-time curve and re-solves. The map is a
strong contraction (a few percent of cost moves elapsed time by a few percent), so
it converges in a handful of rounds. If it has not converged within
``max_iterations``, the last finite iterate is kept and flagged; if an iterate is
ever non-finite, the fresh plan is returned with ``phi = 1`` and flagged
``fallback="fresh"``.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from src.domain.durability.capability import ReferenceSpeed, race_reference
from src.domain.durability.config import DurabilityCoefficients, DurabilityConfig
from src.domain.durability.model import (
    DurabilityProfile,
    RaceWeather,
    accumulate_exposures,
    durability_profile,
    heat_stress,
)

# (elapsed per point, pace per interval, phi per interval) -> profile
Evaluator = Callable[[np.ndarray, np.ndarray, np.ndarray], DurabilityProfile]


@dataclass
class RouteDurability:
    """Everything the planner needs to apply durability to one race."""

    coefficients: DurabilityCoefficients
    config: DurabilityConfig
    # The athlete's own reference speed, if any (see capability.race_reference).
    athlete_reference: Optional[ReferenceSpeed] = None
    weather: RaceWeather = field(default_factory=RaceWeather)
    pre_race_exposure: Optional[float] = None


@dataclass
class DurabilitySolution:
    gap_pace_s_per_km: float           # fresh GAP pace P
    pace: np.ndarray                   # s/km per interval
    elapsed: np.ndarray                # s per point
    profile: DurabilityProfile         # phi per point
    interval_multiplier: np.ndarray    # phi per interval, as used in the pacing
    intensity: np.ndarray              # u per interval
    reference: ReferenceSpeed
    iterations: int
    converged: bool
    fallback: Optional[str] = None


def solve_pacing(ds_km: np.ndarray, factors: np.ndarray, target_time_s: float,
                 multiplier: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """The planner's closed form, generalised with a cost multiplier per interval."""
    gap_pace = target_time_s / float(np.sum(ds_km * factors * multiplier))
    pace = gap_pace * factors * multiplier
    elapsed = np.concatenate([[0.0], np.cumsum(ds_km * pace)])
    return gap_pace, pace, elapsed


def route_evaluator(
    route: RouteDurability,
    factors: np.ndarray,
    descent_m: np.ndarray,
    reference: ReferenceSpeed,
) -> Evaluator:
    """``phi`` along the course for a given elapsed-time / pace curve.

    Intensity is the *required* power relative to critical power, in the project's
    cost model: ``u = phi · a(g) · v / CS``. On the first round (``phi = 1``) that is
    exactly the fresh baseline's required power.
    """
    exposure = route.config.exposure
    cs = float(reference.speed_m_per_s)

    def evaluate(elapsed: np.ndarray, pace: np.ndarray, multiplier: np.ndarray
                 ) -> DurabilityProfile:
        speed = np.divide(1000.0, pace, out=np.zeros_like(pace), where=pace > 0)
        intensity = multiplier * factors * speed / cs
        finish = float(elapsed[-1]) if elapsed[-1] > 0 else 1.0
        mids = (elapsed[:-1] + elapsed[1:]) / 2 / finish
        heat = heat_stress(
            route.weather.temperature_at(mids),
            route.weather.humidity_at(mids, exposure.default_relative_humidity_pct),
            exposure,
        )
        exposures = accumulate_exposures(elapsed, intensity, descent_m, heat, exposure)
        profile = durability_profile(
            exposures, route.coefficients, exposure, route.pre_race_exposure
        )
        profile.extra["intensity"] = intensity
        return profile

    return evaluate


def solve_route(route: RouteDurability, ds_km: np.ndarray, factors: np.ndarray,
                descent_m: np.ndarray, target_time_s: float) -> DurabilitySolution:
    ones = np.ones_like(factors)
    fresh = solve_pacing(ds_km, factors, target_time_s, ones)
    reference = race_reference(
        route.athlete_reference, fresh[0], target_time_s, route.config.capability
    )
    evaluate = route_evaluator(route, factors, descent_m, reference)

    if not route.config.enabled:
        return _fresh_solution(fresh, evaluate, reference, fallback="disabled")

    solver = route.config.solver
    gap_pace, pace, elapsed = fresh
    multiplier = ones
    profile = evaluate(elapsed, pace, multiplier)
    converged = False
    iterations = 0
    for iterations in range(1, solver.max_iterations + 1):
        new_multiplier = profile.interval_multiplier()
        if not np.all(np.isfinite(new_multiplier)):
            return _fresh_solution(fresh, evaluate, reference, fallback="fresh")
        new_gap, new_pace, new_elapsed = solve_pacing(
            ds_km, factors, target_time_s, new_multiplier
        )
        if not (np.isfinite(new_gap) and np.all(np.isfinite(new_elapsed))):
            return _fresh_solution(fresh, evaluate, reference, fallback="fresh")
        d_elapsed = float(np.max(np.abs(new_elapsed - elapsed))) / target_time_s
        d_log_phi = float(np.max(np.abs(np.log(new_multiplier) - np.log(multiplier))))
        gap_pace, pace, elapsed, multiplier = new_gap, new_pace, new_elapsed, new_multiplier
        profile = evaluate(elapsed, pace, multiplier)
        if d_elapsed < solver.tolerance and d_log_phi < solver.tolerance:
            converged = True
            break

    return DurabilitySolution(
        gap_pace_s_per_km=gap_pace,
        pace=pace,
        elapsed=elapsed,
        profile=profile,
        interval_multiplier=multiplier,
        intensity=profile.extra["intensity"],
        reference=reference,
        iterations=iterations,
        converged=converged,
        fallback=None if converged else "not_converged",
    )


def _fresh_solution(fresh, evaluate: Evaluator, reference: ReferenceSpeed,
                    fallback: str) -> DurabilitySolution:
    """The plan without durability, with a neutral profile for the diagnostics."""
    gap_pace, pace, elapsed = fresh
    ones = np.ones_like(pace)
    measured = evaluate(elapsed, pace, ones)
    neutral = DurabilityProfile(
        multiplier=np.ones_like(elapsed),
        log_multiplier=np.zeros_like(elapsed),
        components={k: np.zeros_like(elapsed) for k in measured.components},
        exposures=measured.exposures,
        coefficients=measured.coefficients,
    )
    return DurabilitySolution(
        gap_pace_s_per_km=gap_pace,
        pace=pace,
        elapsed=elapsed,
        profile=neutral,
        interval_multiplier=ones,
        intensity=measured.extra["intensity"],
        reference=reference,
        iterations=0,
        converged=True,
        fallback=fallback,
    )
