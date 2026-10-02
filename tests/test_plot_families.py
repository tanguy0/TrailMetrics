"""What each plot declares about its figure (charts.md § v1.2 — cas particuliers)."""

import unittest
from datetime import date

from src.domain.charts.ir import Trace, TraceKind
from src.domain.dataset.metrics import metric_or_default
from src.domain.dataset.resolved import ResolvedGroup
from src.domain.plots.metric_trend import _declare_family
from src.domain.spec.datasource import TimeWindow


def _group(index: int, end: date) -> ResolvedGroup:
    return ResolvedGroup(label=f"g{index}", index=index,
                         window=TimeWindow(name=f"g{index}", start=date(end.year, 1, 1), end=end))


def _line() -> Trace:
    return Trace(name="s", x=[1, 2], y=[1.0, 2.0])


class MetricTrendFamilyTest(unittest.TestCase):
    def test_a_sum_is_tracking(self) -> None:
        family = _declare_family([_line()], [_group(0, date(2026, 6, 1))],
                                 metric_or_default("distance_km"), "sum", TraceKind.LINE, "calendar")
        self.assertEqual(family, "tracking")

    def test_a_mean_or_a_ratio_is_an_oscillation(self) -> None:
        for key, agg in (("avg_hr", "mean"), ("power_to_hr", "mean"), ("distance_km", "max")):
            with self.subTest(metric=key):
                family = _declare_family([_line()], [_group(0, date(2026, 6, 1))],
                                         metric_or_default(key), agg, TraceKind.LINE, "calendar")
                self.assertEqual(family, "oscillation")

    def test_overlaid_windows_compare_with_the_latest_as_current(self) -> None:
        traces = [_line(), _line(), _line()]
        owners = [_group(0, date(2025, 12, 31)), _group(1, date(2026, 9, 30)), _group(2, date(2024, 12, 31))]
        family = _declare_family(traces, owners, metric_or_default("distance_km"), "sum",
                                 TraceKind.STEP, "elapsed")
        self.assertEqual(family, "comparison")
        self.assertEqual([t.end_label for t in traces], [None, True, None])

    def test_bars_are_left_to_the_renderer(self) -> None:
        self.assertIsNone(_declare_family([_line()], [_group(0, date(2026, 6, 1))],
                                          metric_or_default("distance_km"), "sum", TraceKind.BAR, "calendar"))


if __name__ == "__main__":
    unittest.main()
