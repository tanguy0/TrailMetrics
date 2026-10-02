"""Series 1's area never decides the y-axis: zero joins it only when it should."""

import unittest

from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace, TraceKind
from src.domain.charts.plotly import area_y_range, render_chart
from src.domain.gap import theme


def _chart(y, *, bars=None, y_range=None) -> ChartData:
    traces = [Trace(name="you", x=list(range(len(y))), y=y, color=theme.CHART_YOU_1)]
    if bars:
        traces.append(Trace(name="bars", x=list(range(len(bars))), y=bars, kind=TraceKind.BAR))
    return ChartData(x_axis=Axis(kind=AxisKind.DATE), y_axis=Axis(range=y_range), traces=traces)


class AreaRangeTest(unittest.TestCase):
    def test_narrow_band_far_from_zero_stays_zoomed(self) -> None:
        lo, hi, top = area_y_range(_chart([1.2, 1.5, 1.8]))
        self.assertAlmostEqual(lo, 1.17)
        self.assertAlmostEqual(hi, 1.83)
        self.assertEqual(top, 1.8)

    def test_data_near_zero_keeps_zero(self) -> None:
        self.assertEqual(area_y_range(_chart([5, 30, 60]))[0], 0.0)

    def test_bars_always_start_at_zero(self) -> None:
        self.assertEqual(area_y_range(_chart([40, 45, 50], bars=[42, 44, 48]))[0], 0.0)

    def test_explicit_range_is_kept(self) -> None:
        self.assertEqual(area_y_range(_chart([1.2, 1.8], y_range=[1, 2]))[:2], (1, 2))

    def test_figure_uses_the_range_and_fades_over_it(self) -> None:
        fig = render_chart(_chart([1.2, 1.5, 1.8]))
        self.assertAlmostEqual(fig.layout.yaxis.range[0], 1.17)
        self.assertAlmostEqual(fig.data[0].fillgradient.start, 1.17)
        self.assertEqual(fig.data[0].fillgradient.stop, 1.8)


if __name__ == "__main__":
    unittest.main()
