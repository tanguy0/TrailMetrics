"""The five-level scale, and the GAP profile built on it."""

import unittest

import numpy as np

from src.domain.assessment import rate
from src.domain.gap.assessment import assess
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


if __name__ == "__main__":
    unittest.main()
