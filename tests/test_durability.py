"""Durability model: exposures, phi, personalization, planner integration.

    /path/to/venv/bin/python -m unittest discover -s tests -v

Plain ``unittest`` so no test dependency is added to the project.
"""

import json
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

from src.domain.dataset.features import best_column, build_activity_features, gap_best_column
from src.domain.dataset.in_memory import InMemoryActivityData
from src.domain.durability import solver as solver_module
from src.domain.durability.capability import (
    BEST_EFFORTS,
    TARGET_TIME,
    ReferenceSpeed,
    race_reference,
    reference_from_best_efforts,
    sustainable_fraction,
)
from src.domain.durability.config import (
    DEFAULT_CONFIG,
    DOWNHILL,
    DURATION,
    SEVERE_INTENSITY,
    THERMAL,
    DurabilityCoefficients,
    SolverConfig,
)
from src.domain.durability.history import fit_athlete_durability
from src.domain.durability.model import (
    DurabilityProfile,
    RaceWeather,
    accumulate_exposures,
    durability_profile,
    heat_stress,
)
from src.domain.durability.personalization import (
    PARTIALLY_PERSONALIZED,
    PERSONALIZED,
    POPULATION_ONLY,
    REASON_NO_HISTORY,
    AthleteDurabilityModel,
    fit_athlete,
)
from src.domain.durability.segments import extract_segments
from src.domain.durability.solver import RouteDurability, solve_route
from src.domain.plots.durability_curve import projected_extra_cost
from src.infrastructure.postgres.athlete_model_repository import plain
from src.domain.gap.reference_curves import balanced_runner
from src.domain.models.activity import ActivityStream
from src.domain.race_plan.gpx import CoursePoints
from src.domain.race_plan.planner import adjuster, build_course, plan_race

EXPOSURE = DEFAULT_CONFIG.exposure
POPULATION = DEFAULT_CONFIG.population
CS = 4.0  # m/s — reference speed used by the synthetic athletes


def _phi(hours=2.0, u=0.8, descent_m=0.0, temperature=None, humidity=50.0, n=200,
         coefficients=POPULATION):
    elapsed = np.linspace(0, hours * 3600, n + 1)
    heat = heat_stress(np.full(n, np.nan if temperature is None else temperature),
                       np.full(n, humidity), EXPOSURE)
    exposures = accumulate_exposures(elapsed, np.full(n, u), np.full(n, descent_m / n),
                                     heat, EXPOSURE)
    return durability_profile(exposures, coefficients, EXPOSURE)


class ModelTests(unittest.TestCase):
    def test_fresh_athlete_starts_at_one(self):  # 1
        profile = _phi()
        self.assertEqual(profile.multiplier[0], 1.0)
        self.assertEqual(profile.at(0)["durability_cost_multiplier"], 1.0)

    def test_phi_finite_and_at_least_one(self):  # 2
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = 100
            elapsed = np.cumsum(rng.normal(30, 60, n + 1))  # includes negative dt
            u = rng.normal(0.8, 1.0, n)
            u[::7] = np.nan
            u[::11] = 50.0
            descent = rng.normal(0, 20, n)
            heat = rng.normal(0, 3, n)
            profile = durability_profile(
                accumulate_exposures(elapsed, u, descent, heat, EXPOSURE), POPULATION, EXPOSURE)
            self.assertTrue(np.all(np.isfinite(profile.multiplier)))
            self.assertTrue(np.all(profile.multiplier >= 1.0))
            self.assertTrue(np.all(profile.multiplier <= EXPOSURE.max_multiplier + 1e-12))

    def test_longer_duration_increases_phi(self):  # 3
        self.assertGreater(_phi(hours=4).multiplier[-1], _phi(hours=2).multiplier[-1])
        self.assertTrue(np.all(np.diff(_phi(hours=6).multiplier) >= 0))

    def test_higher_intensity_more_exposure(self):  # 4
        low, high = _phi(u=0.7), _phi(u=0.9)
        self.assertGreater(high.exposures[DURATION][-1], low.exposures[DURATION][-1])
        self.assertEqual(low.exposures[SEVERE_INTENSITY][-1], 0.0)
        self.assertGreater(_phi(u=1.1).exposures[SEVERE_INTENSITY][-1], 0.0)

    def test_downhill_increases_phi(self):  # 5
        flat, hilly = _phi(descent_m=0), _phi(descent_m=2000)
        self.assertGreater(hilly.multiplier[-1], flat.multiplier[-1])
        self.assertGreater(hilly.components[DOWNHILL][-1], 0.0)

    def test_heat_and_humidity_increase_thermal(self):  # 6
        warm = _phi(temperature=25, humidity=40).components[THERMAL][-1]
        hot = _phi(temperature=30, humidity=40).components[THERMAL][-1]
        humid = _phi(temperature=25, humidity=90).components[THERMAL][-1]
        self.assertGreater(warm, 0.0)
        self.assertGreater(hot, warm)
        self.assertGreater(humid, warm)

    def test_cool_conditions_no_thermal_penalty(self):  # 7
        for t, rh in ((5, 90), (12, 60), (15, 30)):
            self.assertEqual(_phi(temperature=t, humidity=rh).components[THERMAL][-1], 0.0)
        self.assertEqual(_phi(temperature=None).components[THERMAL][-1], 0.0)

    def test_weather_interpolates_temperature(self):
        weather = RaceWeather(temperature_start_c=10, temperature_end_c=30,
                              relative_humidity_pct=60)
        np.testing.assert_allclose(weather.temperature_at(np.array([0, 0.5, 1])), [10, 20, 30])
        np.testing.assert_allclose(weather.humidity_at(np.array([0, 1]), 50), [60, 60])
        later = RaceWeather(relative_humidity_start_pct=40, relative_humidity_end_pct=80)
        np.testing.assert_allclose(later.humidity_at(np.array([0.5]), 50), [60])

    def test_clamping_is_reported(self):
        huge = POPULATION.with_values({DURATION: 5.0})
        profile = _phi(hours=10, coefficients=huge)
        self.assertTrue(profile.clamped)
        self.assertAlmostEqual(profile.multiplier[-1], EXPOSURE.max_multiplier)

    def test_negative_coefficient_rejected(self):
        with self.assertRaises(ValueError):
            DurabilityCoefficients(duration=-0.1)


class CapabilityTests(unittest.TestCase):
    def test_target_floor(self):
        cfg = DEFAULT_CONFIG.capability
        # No athlete reference: u equals the sustainable fraction of the target duration.
        ref = race_reference(None, fresh_gap_pace_s_per_km=300.0, target_time_s=4 * 3600, config=cfg)
        self.assertEqual(ref.source, TARGET_TIME)
        self.assertAlmostEqual((1000 / 300) / ref.speed_m_per_s, sustainable_fraction(4 * 3600, cfg))
        # A strong athlete keeps their own reference.
        strong = ReferenceSpeed(6.0, BEST_EFFORTS)
        self.assertIs(race_reference(strong, 300.0, 4 * 3600, cfg), strong)
        # A stale, slow reference is floored by the target.
        slow = ReferenceSpeed(2.0, BEST_EFFORTS)
        self.assertEqual(race_reference(slow, 300.0, 4 * 3600, cfg).source, TARGET_TIME)


def _efforts(rows, gap=False):
    """Feature rows from ``{activity_id: {label: seconds}}``."""
    column = gap_best_column if gap else best_column
    return pd.DataFrame([{"activity_id": a, **{column(k): v for k, v in bests.items()}}
                         for a, bests in rows.items()])


class BestEffortTests(unittest.TestCase):
    """A 4:00/km-ish runner: 1 km 3:30, 3 km 11:30, 5 km 19:45, 10 km 41:00."""

    cfg = DEFAULT_CONFIG.capability
    honest = {"1 km": 210, "3 km": 690, "5 km": 1185, "10 km": 2460}

    def test_tunnel_outlier_rejected_and_next_best_used(self):
        rows = {1: dict(self.honest), 2: {"1 km": 160, "3 km": 720}, 3: {"1 km": 215}}
        # Activity 2's 1 km (2:40, a GPS jump in a tunnel) is far beyond the rest.
        ref = reference_from_best_efforts(_efforts(rows), self.cfg)
        outliers = ref.detail["outliers"]
        self.assertEqual([(o["distance"], o["activity_id"]) for o in outliers], [("1 km", 2)])
        clean = reference_from_best_efforts(_efforts({1: dict(self.honest), 3: {"1 km": 215}}),
                                            self.cfg)
        self.assertAlmostEqual(ref.speed_m_per_s, clean.speed_m_per_s)

    def test_consistent_efforts_all_kept(self):
        ref = reference_from_best_efforts(_efforts({1: dict(self.honest)}), self.cfg)
        self.assertEqual(ref.detail["outliers"], [])
        self.assertEqual(ref.source, BEST_EFFORTS)

    def test_too_few_distances_unchecked(self):
        ref = reference_from_best_efforts(_efforts({1: {"1 km": 160, "3 km": 690}}), self.cfg)
        self.assertEqual(ref.detail["outliers"], [])

    def test_gap_bests_preferred(self):
        raw = _efforts({1: {"10 km": 3600, "Semi": 8400}})     # slow, hilly
        gap = _efforts({1: {"10 km": 2700, "Semi": 6000}}, gap=True)
        both = pd.concat([raw, gap.drop(columns="activity_id")], axis=1)
        ref = reference_from_best_efforts(both, self.cfg)
        self.assertTrue(ref.detail["gap_adjusted"])
        self.assertGreater(ref.speed_m_per_s, reference_from_best_efforts(raw, self.cfg).speed_m_per_s)


def _hilly_run(climb: bool):
    n = 3000
    t = np.arange(n, dtype=float)
    grade = 0.10 if climb else 0.0
    speed = 2.5 if climb else 4.0
    distance = t * speed
    return ActivityStream(
        activity_id=7, sport_type="TrailRun", time=t, distance=distance,
        altitude=100 + distance * grade, heartrate=np.full(n, 150.0),
        start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


class GapBestEffortFeatureTests(unittest.TestCase):
    def test_flat_gap_best_equals_raw(self):
        row = build_activity_features(_hilly_run(climb=False))
        self.assertAlmostEqual(row[gap_best_column("5 km")], row[best_column("5 km")], delta=1.0)

    def test_climb_gap_best_faster_than_raw(self):
        row = build_activity_features(_hilly_run(climb=True))
        self.assertLess(row[gap_best_column("5 km")], 0.8 * row[best_column("5 km")])


# --- Synthetic history ----------------------------------------------------------

def _run(activity_id, day, hours, theta, rng, speed=3.2, hr_noise=1.5):
    """A steady flat run whose HR reserve rises with a true duration coefficient ``theta``."""
    n = int(hours * 3600)
    t = np.arange(n + 1, dtype=float)
    distance = t * speed
    u = speed / CS
    t_h = t / 3600
    true_log_phi = theta * t_h * u ** 2
    cfg = DEFAULT_CONFIG.personalization
    reserve = 90.0 * np.exp(true_log_phi + cfg.hr_drift_per_hour * t_h)
    hr = cfg.hr_rest_bpm + reserve + rng.normal(0, hr_noise, n + 1)
    return ActivityStream(
        activity_id=activity_id, sport_type="Run", time=t, distance=distance,
        altitude=np.full(n + 1, 100.0), heartrate=hr,
        start_date=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
    )


def _segments(streams):
    out = []
    for stream in streams:
        found, _ = extract_segments(stream, ReferenceSpeed(CS, BEST_EFFORTS),
                                    EXPOSURE, DEFAULT_CONFIG.personalization)
        out.extend(found)
    return out


class PersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.today = date(2026, 9, 29)

    def test_missing_history_gives_population(self):  # 8
        model = fit_athlete_durability(InMemoryActivityData([]), self.today)
        self.assertEqual(model.confidence, POPULATION_ONLY)
        self.assertEqual(model.coefficients, POPULATION)
        empty = fit_athlete([], ReferenceSpeed(CS, BEST_EFFORTS), DEFAULT_CONFIG)
        self.assertEqual(empty.confidence, POPULATION_ONLY)
        self.assertIn(REASON_NO_HISTORY, empty.reasons)

    def test_sparse_history_stays_close_to_population(self):  # 9
        streams = [_run(i, self.today - timedelta(days=i), 1.0, 0.06, self.rng, hr_noise=4.0)
                   for i in range(1, 4)]
        model = fit_athlete(_segments(streams), ReferenceSpeed(CS, BEST_EFFORTS), DEFAULT_CONFIG)
        self.assertEqual(model.confidence, PARTIALLY_PERSONALIZED)
        gap = abs(model.coefficients.duration - POPULATION.duration)
        self.assertLess(gap, abs(0.06 - POPULATION.duration) / 2)

    def test_strong_evidence_moves_coefficients(self):  # 10
        streams = [_run(i, self.today - timedelta(days=3 * i), 2.5, 0.06, self.rng)
                   for i in range(1, 21)]
        model = fit_athlete(_segments(streams), ReferenceSpeed(CS, BEST_EFFORTS), DEFAULT_CONFIG)
        self.assertEqual(model.confidence, PERSONALIZED)
        self.assertGreater(model.coefficients.duration, POPULATION.duration + 0.02)
        self.assertGreater(model.personal_weight[DURATION], 0.6)
        # Thermal has no historical signal and is never personalized.
        self.assertEqual(model.coefficients.thermal, POPULATION.thermal)

    def test_a_stored_model_comes_back_whole(self):
        # What the Tools keep in ``athlete_models``: JSON with NaNs as nulls.
        streams = [_run(i, self.today - timedelta(days=3 * i), 2.5, 0.06, self.rng)
                   for i in range(1, 9)]
        model = fit_athlete(_segments(streams), ReferenceSpeed(CS, BEST_EFFORTS), DEFAULT_CONFIG)
        model.segments[0] = replace(model.segments[0], heartrate_bpm=float("nan"))
        stored = json.loads(json.dumps(plain(model.to_store()), allow_nan=False))
        back = AthleteDurabilityModel.from_store(stored)
        self.assertEqual(back.to_dict(), model.to_dict())
        self.assertEqual(len(back.segments), len(model.segments))
        self.assertEqual(back.segments[3], model.segments[3])
        self.assertTrue(np.isnan(back.segments[0].heartrate_bpm))
        self.assertEqual(
            projected_extra_cost(back, DEFAULT_CONFIG, np.array([1.0, 4.0])).tolist(),
            projected_extra_cost(model, DEFAULT_CONFIG, np.array([1.0, 4.0])).tolist(),
        )

    def test_old_runs_ignored(self):
        old = [_run(i, self.today - timedelta(days=400 + i), 2.5, 0.06, self.rng)
               for i in range(1, 10)]
        model = fit_athlete_durability(InMemoryActivityData(old), self.today)
        self.assertEqual(model.n_activities, 0)
        self.assertEqual(model.confidence, POPULATION_ONLY)

    def test_intermittent_session_excluded(self):
        stream = _run(1, self.today, 1.5, 0.0, self.rng)
        speed = np.where((stream.time // 300) % 2 == 0, 4.5, 2.0)
        stream.distance = np.concatenate([[0], np.cumsum(speed[1:])])
        _, reason = extract_segments(stream, ReferenceSpeed(CS, BEST_EFFORTS), EXPOSURE,
                                     DEFAULT_CONFIG.personalization)
        self.assertIsNotNone(reason)


# --- Planner integration ---------------------------------------------------------

def _course(km=42.0, climb=False):
    n = 400
    lat = np.full(n, 45.0)
    lon = np.linspace(6.0, 6.0 + km / 78.7, n)  # ~78.7 km per degree at 45°N
    x = np.linspace(0, 1, n)
    elevation = 500 + (400 * np.sin(2 * np.pi * 3 * x) if climb else 0 * x)
    return build_course(CoursePoints(lat=lat, lon=lon, elevation=elevation))


def _route(**kw):
    return RouteDurability(coefficients=POPULATION, config=DEFAULT_CONFIG, **kw)


class PlannerTests(unittest.TestCase):
    def test_phi_changes_cost_without_bypassing_terrain(self):  # 11
        course = _course(climb=True)
        adjust = adjuster(balanced_runner())
        fresh = plan_race(course, 4 * 3600, adjust)
        tired = plan_race(course, 4 * 3600, adjust, durability=_route())
        phi = tired.durability.interval_multiplier
        factors = adjust(course.grade)
        # Same target, a faster start and a slower finish.
        self.assertAlmostEqual(tired.elapsed[-1], 4 * 3600, places=6)
        self.assertLess(tired.gap_pace_s_per_km, fresh.gap_pace_s_per_km)
        self.assertGreater(phi[-1], phi[0])
        # Terrain still sets every point: pace / (a(g) · phi) is one constant.
        np.testing.assert_allclose(tired.pace / (factors * phi), tired.gap_pace_s_per_km)
        np.testing.assert_allclose(fresh.pace / factors, fresh.gap_pace_s_per_km)

    def test_iteration_converges(self):  # 12
        course = _course(km=100, climb=True)
        plan = plan_race(course, 14 * 3600, adjuster(balanced_runner()), durability=_route(
            weather=RaceWeather(temperature_start_c=10, temperature_end_c=28)))
        solution = plan.durability
        self.assertTrue(solution.converged)
        self.assertIsNone(solution.fallback)
        self.assertLessEqual(solution.iterations, DEFAULT_CONFIG.solver.max_iterations)

    def test_iteration_cap_is_safe(self):  # 12
        config = replace(DEFAULT_CONFIG, solver=SolverConfig(tolerance=1e-12, max_iterations=1))
        route = RouteDurability(coefficients=POPULATION, config=config)
        plan = plan_race(_course(), 4 * 3600, adjuster(balanced_runner()), durability=route)
        self.assertFalse(plan.durability.converged)
        self.assertEqual(plan.durability.fallback, "not_converged")
        self.assertTrue(np.all(np.isfinite(plan.pace)))
        self.assertAlmostEqual(plan.elapsed[-1], 4 * 3600, places=6)

    def test_non_finite_falls_back_to_fresh(self):  # 12
        course = _course()
        adjust = adjuster(balanced_runner())
        real = solver_module.route_evaluator

        def broken(*args, **kwargs):
            evaluate = real(*args, **kwargs)

            def bad(elapsed, pace, multiplier):
                profile = evaluate(elapsed, pace, multiplier)
                profile.multiplier = np.full_like(profile.multiplier, np.nan)
                return profile
            return bad

        with mock.patch.object(solver_module, "route_evaluator", broken):
            plan = plan_race(course, 4 * 3600, adjust, durability=_route())
        fresh = plan_race(course, 4 * 3600, adjust)
        self.assertEqual(plan.durability.fallback, "fresh")
        np.testing.assert_allclose(plan.pace, fresh.pace)

    def test_disabled_is_neutral(self):
        config = replace(DEFAULT_CONFIG, enabled=False)
        route = RouteDurability(coefficients=POPULATION, config=config)
        course = _course()
        adjust = adjuster(balanced_runner())
        plan = plan_race(course, 4 * 3600, adjust, durability=route)
        np.testing.assert_allclose(plan.pace, plan_race(course, 4 * 3600, adjust).pace)
        self.assertTrue(np.all(plan.durability.profile.multiplier == 1.0))

    def test_no_iteration_needed_when_coefficients_zero(self):
        zero = DurabilityCoefficients(duration=0, severe_intensity=0, downhill=0, thermal=0)
        route = RouteDurability(coefficients=zero, config=DEFAULT_CONFIG)
        course = _course()
        adjust = adjuster(balanced_runner())
        solution = solve_route(route, np.diff(course.distance) / 1000, adjust(course.grade),
                               np.zeros(len(course.grade)), 4 * 3600)
        self.assertTrue(solution.converged)
        self.assertEqual(solution.iterations, 1)
        self.assertIsInstance(solution.profile, DurabilityProfile)


if __name__ == "__main__":
    unittest.main()
