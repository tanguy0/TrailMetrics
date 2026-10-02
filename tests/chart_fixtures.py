"""The chart shapes the app produces, each with the family it must land in.

Shared by the family tests and the renderer parity test: one list of figures,
read by both engines. Shapes mirror what each plot actually builds; names say
which one.
"""

from typing import List, Tuple

from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace, TraceKind
from src.domain.gap import theme

_WEEKS = ["2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22"]
_DATE = Axis(kind=AxisKind.DATE)
_PACE = Axis(kind=AxisKind.DURATION, reversed=True)


def _lin(**kwargs) -> Axis:
    return Axis(kind=AxisKind.LINEAR, **kwargs)


def _trace(y, **kwargs) -> Trace:
    return Trace(name=kwargs.pop("name", "series"), x=kwargs.pop("x", list(_WEEKS)), y=y, **kwargs)


Y1, Y2, Y3, REF = theme.CHART_YOU_1, theme.CHART_YOU_2, theme.CHART_YOU_3, theme.CHART_REF

# (name, chart, expected family) — undeclared figures exercise the fallback.
FIXTURES: List[Tuple[str, ChartData, str]] = [
    ("volume_line_undeclared", ChartData(
        x_axis=_DATE, y_axis=_lin(), traces=[_trace([5, 35, 40, 30], color=Y1)]), "tracking"),
    ("volume_line_declared", ChartData(
        x_axis=_DATE, y_axis=_lin(), family="tracking",
        traces=[_trace([20, 35, 40, 30], color=Y1)]), "tracking"),
    ("power_to_hr", ChartData(
        x_axis=_DATE, y_axis=_lin(), traces=[_trace([1.4, 1.5, 1.55, 1.6], color=Y1)]),
     "oscillation"),
    ("average_pace", ChartData(
        x_axis=_DATE, y_axis=_PACE, traces=[_trace([330, 320, 315, 318], color=Y1)]),
     "oscillation"),
    ("cumulative_three_years", ChartData(
        x_axis=_DATE, y_axis=_lin(), traces=[
            _trace([0, 100, 200, 300], kind=TraceKind.STEP, color=Y1, name="2024"),
            _trace([0, 90, 180, 260], kind=TraceKind.STEP, color=Y2, name="2025"),
            _trace([0, 50, 120, None], kind=TraceKind.STEP, color=Y3, name="2026")]),
     "comparison"),
    ("cumulative_elapsed_declared", ChartData(
        x_axis=_lin(), y_axis=_lin(), family="comparison", traces=[
            _trace([0, 100, 200], x=[0, 1, 2], kind=TraceKind.STEP, color=Y1),
            _trace([0, 90, 180], x=[0, 1, 2], kind=TraceKind.STEP, color=Y2, end_label=True)]),
     "comparison"),
    ("gap_curves", ChartData(
        x_axis=_lin(), y_axis=_lin(), traces=[
            _trace([1.4, 1.0, 1.3, 2.0], x=[-20, 0, 10, 30], color=Y1),
            _trace([1.5, 1.0, 1.4, 2.1], x=[-20, 0, 10, 30], color=REF, dash="--")]),
     "function"),
    ("weekly_volume_bars", ChartData(
        x_axis=_DATE, y_axis=_lin(), traces=[_trace([20, 35, 40, 30], kind=TraceKind.BAR, color=Y1)]),
     "composition"),
    ("home_volume_dual_axis", ChartData(
        x_axis=_DATE, y_axis=_lin(), y2_axis=_lin(), traces=[
            _trace([20, 35, 40, 30], kind=TraceKind.BAR, color=Y1),
            _trace([300, 800, 600, 400], axis="y2", color=Y2)]),
     "composition"),
    ("fitness_fatigue_form", ChartData(
        x_axis=_DATE, y_axis=_lin(), family="oscillation", baseline=0.0, traces=[
            _trace([40, 42, 44, 45], color=Y1, width=1.5, area=False, end_label=True),
            _trace([50, 38, 60, 48], color=Y2, width=2.2, area=True, end_label=True),
            _trace([-10, 4, -16, -3], kind=TraceKind.BAR, color=theme.FORM)]),
     "oscillation"),
    ("gradient_map", ChartData(
        x_axis=_DATE, y_axis=_lin(range=[0, 100]), traces=[
            _trace([20, 25, 30, 20], kind=TraceKind.AREA, stack_group="bands", color=theme.MOSS)]),
     "composition"),
    ("scatter_with_trend", ChartData(
        x_axis=_lin(), y_axis=_PACE, traces=[
            _trace([330, 320, 315, 318], x=[1, 2, 3, 4], kind=TraceKind.SCATTER, color=Y1),
            _trace([330, 315], x=[1, 4], color=Y1, show_legend=False)]),
     "scatter"),
    ("weekly_feel", ChartData(
        x_axis=_DATE, y_axis=_lin(range=[0, 11.5]), y2_axis=_lin(range=[0.5, 3.5]), family="composition",
        traces=[
            _trace([5, 6, 8, 3], kind=TraceKind.BAR, color=theme.SUN,
                   point_colors=[theme.SUN, theme.SUN, theme.DANGER, theme.MOSS]),
            _trace([2, 3, 1, 2], axis="y2", color=theme.MOSS_INK, end_label=False)]),
     "composition"),
    ("records_two_distances", ChartData(
        x_axis=_DATE, y_axis=_PACE, family="comparison", traces=[
            _trace([300, 295, 290, 290], kind=TraceKind.STEP, color=Y1, end_label=True),
            _trace([320, 318, 310, 310], kind=TraceKind.STEP, color=Y2),
            _trace([300, 290], x=_WEEKS[:2], kind=TraceKind.SCATTER, color=Y1, show_legend=False)]),
     "comparison"),
    ("stream_with_altitude_background", ChartData(
        x_axis=_lin(), y_axis=_PACE, y2_axis=_lin(), family="function", traces=[
            _trace([500, 900, 1500, 800], x=[0, 1, 2, 3], axis="y2", background=True, color=REF),
            _trace([330, 320, 340, 310], x=[0, 1, 2, 3], color=Y1)]),
     "function"),
]
