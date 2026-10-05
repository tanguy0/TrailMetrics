"""The five-level scale, and the GAP profile built on it."""

import unittest

import numpy as np

from src.domain.assessment import rate
from src.domain.durability import assessment as durability
from src.domain.durability.config import DEFAULT_CONFIG, DOWNHILL, DURATION, SEVERE_INTENSITY
from src.domain.durability.personalization import (
    PARTIALLY_PERSONALIZED,
    AthleteDurabilityModel,
    population_model,
)
from src.domain.gap.assessment import assess, profile_chart
from src.domain.gap.reference_curves import balanced_runner
from src.domain.models.gap import GapCurve


def _curve(x, y):
    n = len(x)
    return GapCurve(np.asarray(x, float), np.asarray(y, float), np.zeros(n), np.ones(n, int))


class RateTest(unittest.TestCase):
    def test_bands(self):
        cases = {20: "poor", 15.01: "poor", 15: "limited", 6: "limited", 5: "average",
                 0: "average", -5: "average", -6: "good", -15: "good", -15.01: "excellent",
                 None: "insufficient"}
        for value, level in cases.items():
            self.assertEqual(rate(value), level, value)


class GapAssessmentTest(unittest.TestCase):
    def test_the_reference_is_average_everywhere(self):
        levels = [a.level for a in assess(balanced_runner(), balanced_runner())]
        self.assertEqual(levels, ["average"] * 4)

    def test_each_terrain_reads_its_own_gradients(self):
        ref = balanced_runner()
        x = np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200], float)
        base = np.interp(x, ref.bin_centers, ref.means)
        # 20 % dearer on steep descents, 10 % cheaper on climbs, rest as the reference.
        factor = np.where(x <= -120, 1.2, np.where((x >= 30) & (x <= 120), 0.9, 1.0))
        by_key = {a.key: a for a in assess(_curve(x, base * factor), ref)}
        self.assertEqual(by_key["steep_downhill"].level, "poor")
        self.assertAlmostEqual(by_key["steep_downhill"].extra_cost_pct, 20.0)
        self.assertEqual(by_key["downhill"].level, "average")
        self.assertEqual(by_key["uphill"].level, "good")
        self.assertEqual(by_key["steep_uphill"].level, "average")

    def test_a_terrain_without_points_is_insufficient(self):
        ref = balanced_runner()
        x = np.array([-100, -50, 0, 50, 100], float)
        by_key = {a.key: a for a in assess(_curve(x, np.interp(x, ref.bin_centers, ref.means)), ref)}
        self.assertEqual(by_key["steep_downhill"].level, "insufficient")
        self.assertIsNone(by_key["steep_uphill"].extra_cost_pct)
        self.assertEqual([a.level for a in assess(None, ref)], ["insufficient"] * 4)


class DurabilityAssessmentTest(unittest.TestCase):
    def _model(self, ratios, weights):
        population = DEFAULT_CONFIG.population
        return AthleteDurabilityModel(
            coefficients=population.with_values(
                {name: population.get(name) * ratio for name, ratio in ratios.items()}
            ),
            population=population,
            confidence=PARTIALLY_PERSONALIZED,
            personal_weight=weights,
        )

    def test_each_quality_is_its_coefficient_against_the_population(self):
        model = self._model(
            {DURATION: 1.3, SEVERE_INTENSITY: 1.0, DOWNHILL: 0.8},
            {DURATION: 0.9, SEVERE_INTENSITY: 0.1, DOWNHILL: 0.5},
        )
        by_key = {a.key: a for a in durability.assess(model)}
        self.assertEqual(by_key["long_efforts"].level, "poor")
        self.assertAlmostEqual(by_key["long_efforts"].extra_cost_pct, 30.0)
        # Barely informed by the athlete's runs: no level rather than the prior's.
        self.assertEqual(by_key["hard_efforts"].level, "insufficient")
        self.assertEqual(by_key["descents"].level, "excellent")

    def test_no_personal_fit_means_no_levels_and_no_chart(self):
        model = population_model(DEFAULT_CONFIG.population, [])
        self.assertEqual([a.level for a in durability.assess(model)], ["insufficient"] * 3)
        self.assertIsNone(durability.profile_chart(model, DEFAULT_CONFIG, "fr"))

    def test_the_chart_is_the_runner_against_the_average(self):
        model = self._model({DURATION: 1.3}, {DURATION: 0.9})
        chart = durability.profile_chart(model, DEFAULT_CONFIG, "fr")
        average, you = chart.traces
        self.assertEqual((average.name, you.name), ("Coureur moyen", "Vous"))
        self.assertGreater(you.y[-1], average.y[-1])


class GapProfileChartTest(unittest.TestCase):
    def test_draws_the_assessed_curve_against_the_reference_in_percent(self):
        ref = balanced_runner()
        x = np.array([-300, -100, 0, 100, 300], float)
        mine = _curve(x, np.interp(x, ref.bin_centers, ref.means) * 0.9)
        chart = profile_chart(mine, ref, "fr")
        reference, you = chart.traces
        self.assertEqual(you.name, "Vous")
        self.assertEqual(you.x, [-30.0, -10.0, 0.0, 10.0, 30.0])
        # The reference is cut to the window, which widens to the runner's reach.
        self.assertEqual((min(reference.x), max(reference.x)), (-30.0, 30.0))
        self.assertEqual([m.x for m in chart.markers], [-12.0, -3.0, 3.0, 12.0])


if __name__ == "__main__":
    unittest.main()
