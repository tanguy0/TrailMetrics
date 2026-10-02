"""Each figure lands in its charts.md § v1.2 family, with that family's treatment."""

import unittest

from src.domain.charts.families import plan
from tests.chart_fixtures import FIXTURES


def _plans():
    return {name: (plan(chart), chart, expected) for name, chart, expected in FIXTURES}


class ChartFamiliesTest(unittest.TestCase):
    def test_every_fixture_lands_in_its_family(self) -> None:
        for name, (decided, _, expected) in _plans().items():
            with self.subTest(chart=name):
                self.assertEqual(decided.family, expected)

    def test_only_tracking_and_a_declared_area_fill(self) -> None:
        plans = _plans()
        filled = sorted(name for name, (decided, _, _) in plans.items() if decided.area is not None)
        self.assertEqual(filled, ["fitness_fatigue_form", "volume_line_declared", "volume_line_undeclared"])
        # The fatigue (second trace), not the fitness.
        self.assertEqual(plans["fitness_fatigue_form"][0].area, 1)

    def test_comparison_emphasises_the_current_period(self) -> None:
        decided = _plans()["cumulative_three_years"][0]
        # 2024 and 2025 reach furthest on the calendar axis; ties go to the first.
        self.assertEqual(decided.end_labels, [decided.main])
        self.assertEqual(decided.widths[decided.main], 2.4)
        self.assertTrue(all(w == 1.5 for i, w in decided.widths.items() if i != decided.main))

    def test_declared_current_wins(self) -> None:
        decided = _plans()["cumulative_elapsed_declared"][0]
        self.assertEqual((decided.main, decided.end_labels), (1, [1]))

    def test_oscillation_has_a_baseline_and_no_area(self) -> None:
        decided = _plans()["power_to_hr"][0]
        self.assertIsNone(decided.area)
        self.assertAlmostEqual(decided.baseline, 1.5125)
        self.assertEqual(decided.end_labels, [0])

    def test_a_second_axis_drops_end_labels(self) -> None:
        self.assertEqual(_plans()["stream_with_altitude_background"][0].end_labels, [])

    def test_function_has_no_end_labels(self) -> None:
        self.assertEqual(_plans()["gap_curves"][0].end_labels, [])


if __name__ == "__main__":
    unittest.main()
