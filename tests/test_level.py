"""Level assessment (design/specs/level.md): VDOT, the three tests, the zones.

    /path/to/venv/bin/python -m unittest discover -s tests -t . -v
"""

import unittest

from src.domain.level import vdot as model
from src.domain.level.estimate import (
    LevelInputError,
    estimate,
    from_critical_speed,
    from_half_cooper,
    from_records,
)
from src.domain.level.zones import pace_zones


class VdotTest(unittest.TestCase):
    def test_known_values(self):
        # level.md § Pivot commun.
        self.assertAlmostEqual(model.vdot(5000, 20), 49.8, delta=0.1)
        self.assertAlmostEqual(model.vdot(10000, 40), 52.0, delta=0.1)
        self.assertAlmostEqual(model.vdot(42195, 180), 53.5, delta=0.1)

    def test_short_effort_follows_the_formula(self):
        # level.md quotes ≈ 50.4 for 1500 m in 5:00; the formula it gives says
        # 54.5 — the published tables flatten the short end, the formula does
        # not. The formula is what the app computes.
        self.assertAlmostEqual(model.vdot(1500, 5), 54.5, delta=0.1)

    def test_vma_is_the_speed_whose_vo2_is_the_vdot(self):
        for value in (35.0, 50.0, 70.0):
            speed = model.vma_kmh(value) * 1000 / 60
            self.assertAlmostEqual(model.vo2_at(speed), value, places=6)

    def test_vma_for_a_20_minute_5k(self):
        # VDOT ≈ 50: Daniels' vVO2max is ≈ 3:51 /km, so a 5 km at ~96 % of VMA.
        self.assertAlmostEqual(model.vma_kmh(model.vdot(5000, 20)), 15.6, delta=0.1)


class HalfCooperTest(unittest.TestCase):
    def test_vdot_route_stays_close_to_the_field_rule(self):
        result = from_half_cooper(1620)
        self.assertEqual(result.method, "half_cooper")
        self.assertLess(abs(result.vma_kmh - 16.2) / 16.2, 0.06)
        self.assertEqual(result.extras["field_vma_kmh"], 16.2)
        self.assertEqual(result.notes[0].key, "level.note.field_vma")

    def test_out_of_range(self):
        with self.assertRaises(LevelInputError):
            from_half_cooper(300)


class CriticalSpeedTest(unittest.TestCase):
    def test_critical_speed_and_d_prime(self):
        result = from_critical_speed(1000, 3700)
        cs = (3700 - 1000) / 540
        self.assertAlmostEqual(result.extras["critical_pace_s_per_km"], round(1000 / cs, 1))
        self.assertEqual(result.extras["d_prime_m"], round(1000 - cs * 180))
        self.assertAlmostEqual(result.vdot, round(model.vdot(3700, 12), 1))

    def test_incoherent_distances(self):
        with self.assertRaises(LevelInputError) as caught:
            from_critical_speed(1000, 1500)  # D' far beyond 500 m
        self.assertEqual(caught.exception.key, "level.error.cs_inconsistent")
        with self.assertRaises(LevelInputError):
            from_critical_speed(1200, 1100)


class RecordsTest(unittest.TestCase):
    def test_median_from_three(self):
        rows = [(5000, 20 * 60), (10000, 41 * 60 + 30), (21097, 92 * 60)]
        result = from_records(rows)
        vdots = sorted(model.vdot(d, s / 60) for d, s in rows)
        self.assertAlmostEqual(result.vdot, round(vdots[1], 1))

    def test_mean_from_two(self):
        rows = [(5000, 20 * 60), (10000, 40 * 60)]
        result = from_records(rows)
        expected = (model.vdot(5000, 20) + model.vdot(10000, 40)) / 2
        self.assertAlmostEqual(result.vdot, round(expected, 1))

    def test_consistent_records_say_so(self):
        result = from_records([(5000, 20 * 60), (10000, 40 * 60 + 30), (21097, 89 * 60)])
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.notes[0].key, "level.note.records_consistent")

    def test_a_fast_5k_and_a_slow_marathon(self):
        result = from_records([(5000, 18 * 60), (10000, 38 * 60), (42195, 4 * 3600)])
        self.assertEqual(result.notes[0].key, "level.note.records_short_better")
        self.assertNotEqual(result.confidence, "high")

    def test_validation(self):
        with self.assertRaises(LevelInputError):
            from_records([(800, 150)])            # under 3 minutes
        with self.assertRaises(LevelInputError):
            from_records([(1000, 60)])            # 1:00 /km
        with self.assertRaises(LevelInputError):
            from_records([(100000, 7 * 3600)])    # over 6 hours
        with self.assertRaises(LevelInputError):
            from_records([])


class DispatchAndZonesTest(unittest.TestCase):
    def test_dispatch(self):
        result = estimate("records", {"records": [{"distance_m": 10000, "seconds": 2400}]})
        self.assertEqual(result.method, "records")
        with self.assertRaises(LevelInputError):
            estimate("records", {"records": [{"distance_m": "x", "seconds": 2400}]})
        with self.assertRaises(LevelInputError):
            estimate("nope", {})

    def test_zones_are_fastest_first(self):
        zones = pace_zones(240.0)  # 4:00 /km at VMA
        self.assertEqual([z.key for z in zones][0], "z2")
        endurance = zones[1]
        self.assertAlmostEqual(endurance.fast_s_per_km, 240 / 0.75)
        self.assertAlmostEqual(endurance.slow_s_per_km, 240 / 0.70)


if __name__ == "__main__":
    unittest.main()
