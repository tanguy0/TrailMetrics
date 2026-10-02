"""The gradient area belongs to tracking figures and starts at zero (charts.md § v1.2)."""

import unittest

from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace
from src.domain.charts.plotly import area_y_range, render_chart
from src.domain.gap import theme


def _chart(y, *, family=None, y_range=None) -> ChartData:
    return ChartData(
        x_axis=Axis(kind=AxisKind.DATE), y_axis=Axis(range=y_range), family=family,
        traces=[Trace(name="you", x=list(range(len(y))), y=y, color=theme.CHART_YOU_1)],
    )


class AreaRangeTest(unittest.TestCase):
    def test_tracking_area_starts_at_zero(self) -> None:
        lo, hi, top = area_y_range(_chart([20, 30, 40], family="tracking"))
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 42.0)
        self.assertEqual(top, 40)

    def test_explicit_range_is_kept(self) -> None:
        self.assertEqual(area_y_range(_chart([1.2, 1.8], y_range=[1, 2]))[:2], (1, 2))

    def test_tracking_figure_fades_over_the_visible_range(self) -> None:
        fig = render_chart(_chart([20, 30, 40], family="tracking"))
        self.assertEqual(fig.layout.yaxis.range[0], 0.0)
        self.assertEqual(fig.data[0].fill, "tozeroy")
        self.assertEqual(fig.data[0].fillgradient.stop, 40)

    def test_a_level_far_from_zero_gets_no_area_and_stays_zoomed(self) -> None:
        fig = render_chart(_chart([1.2, 1.5, 1.8]))
        self.assertIsNone(fig.data[0].fill)
        self.assertIsNone(fig.layout.yaxis.range)


if __name__ == "__main__":
    unittest.main()
