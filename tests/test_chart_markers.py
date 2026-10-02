"""Today and races land on exactly the date-axis charts that cover them."""

import unittest
from datetime import date

from src.domain.charts.ir import Axis, AxisKind, ChartData, PlotOutput, Trace
from src.domain.charts.markers import Race, chart_span, pin_markers

WEEKS = ["2026-09-14", "2026-09-21", "2026-09-28"]


def _chart(x, kind=AxisKind.DATE) -> ChartData:
    return ChartData(x_axis=Axis(kind=kind), traces=[Trace(name="a", x=x, y=[1.0] * len(x))])


class ChartMarkersTest(unittest.TestCase):
    def test_weekly_bins_cover_their_last_week(self) -> None:
        self.assertEqual(chart_span(_chart(WEEKS)), (date(2026, 9, 14), date(2026, 10, 4)))

    def test_today_and_races_inside_the_window_only(self) -> None:
        output = PlotOutput(charts=[_chart(WEEKS)])
        races = [Race(date(2026, 9, 20), "UTMB"), Race(date(2026, 12, 1), "Later")]
        pinned = pin_markers(output, races, date(2026, 10, 2), "Today")
        self.assertEqual(
            [(m.kind, m.label) for m in pinned.charts[0].markers],
            [("race", "UTMB"), ("today", "Today")],
        )

    def test_cached_output_is_left_untouched(self) -> None:
        output = PlotOutput(charts=[_chart(WEEKS)])
        pin_markers(output, [], date(2026, 10, 2), "Today")
        self.assertEqual(output.charts[0].markers, [])

    def test_binned_axis_puts_today_on_the_current_period(self) -> None:
        chart = _chart(WEEKS)
        chart.x_bucket = "week"
        races = [Race(date(2026, 10, 1), "Trail"), Race(date(2026, 9, 16), "10 km")]
        pinned = pin_markers(PlotOutput(charts=[chart]), races, date(2026, 10, 2), "Today")
        markers = {(m.kind, m.label): m for m in pinned.charts[0].markers}
        # Thursday 2 October is in the week opening Monday 28 September — the last point.
        self.assertEqual(markers[("today", "Today")].x, "2026-09-28")
        # A race in the current week shares that x and stacks; an older one keeps its day.
        self.assertEqual((markers[("race", "Trail")].x, markers[("race", "Trail")].stacked), ("2026-09-28", True))
        self.assertEqual((markers[("race", "10 km")].x, markers[("race", "10 km")].stacked), ("2026-09-16", False))

    def test_daily_axis_stays_exact(self) -> None:
        chart = _chart(["2026-09-30", "2026-10-01", "2026-10-02"])
        chart.x_bucket = "day"
        pinned = pin_markers(PlotOutput(charts=[chart]), [], date(2026, 10, 2), "Today")
        self.assertEqual(pinned.charts[0].markers[0].x, "2026-10-02")

    def test_non_date_axes_get_nothing(self) -> None:
        output = PlotOutput(charts=[_chart([0, 5, 10], kind=AxisKind.LINEAR)])
        self.assertIs(pin_markers(output, [], date(2026, 10, 2), "Today"), output)


if __name__ == "__main__":
    unittest.main()
