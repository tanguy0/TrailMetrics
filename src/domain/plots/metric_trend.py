"""Trend of any activity metric over time — the workhorse activity-level plot.

One plot type covers what used to be four hand-written figures (mileage,
elevation gain, average gradient, power-to-HR) and a great deal more, because
every axis of variation is a parameter: which metric, how to aggregate, at what
granularity, cumulative or per-period, on the calendar or aligned to each group's
start, as a line / step / bar / area, optionally smoothed and split by sport.

The group-aligned x-axis (``x_mode="elapsed"``) is the old "season overlay":
several time windows in the panel's data source, each drawn from a common 0, so
blocks of different lengths compare directly.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    CellFormat,
    ChartData,
    Column,
    PlotOutput,
    TableData,
    Trace,
    TraceKind,
    empty_output,
)
from src.domain.dataset.binning import (
    bin_start,
    elapsed_bin_index,
    elapsed_bin_months,
    max_window_months,
    to_date,
)
from src.domain.dataset.metrics import NO_METRIC, metric_or_default, optional_metric
from src.domain.dataset.resolved import DataLevel, ResolvedGroup, ResolvedPanelData
from src.domain.plots.base import (
    PlotDefinition,
    group_color,
    hover_template,
    metric_axis,
    metric_cell_format,
    metric_hover_texts,
    metric_label,
    register,
    series_color,
)
from src.domain.races.smoothing import smooth_uniform_series
from src.domain.spec.params import (
    Choice,
    ParamSpec,
    boolean,
    choice,
    integer,
    when,
)
from src.translations import translate

_CHART_KINDS = {
    "line": TraceKind.LINE,
    "step": TraceKind.STEP,
    "bar": TraceKind.BAR,
    "area": TraceKind.AREA,
}

# Dual-metric charts encode the *metric* in colour (see `_metric_traces`), so the
# two need to be unmistakable and to match their axis tints. Both come from the
# shared curve palette, so they agree with every other figure in the app.
_PRIMARY_COLOR = series_color(0)
_SECONDARY_COLOR = series_color(1)

# Group identity moves to the dash pattern when colour is spent on the metric.
_GROUP_DASHES = ["-", "--", "-.", ":"]


PARAMS: List[ParamSpec] = [
    choice("metric", "param.metric", "distance_km",
           choices_from="activity_metrics", help_key="param.metric.help"),
    # Ratios and counts fix their own aggregation, so the control is hidden.
    choice("aggregation", "param.aggregation", "sum",
           choices_from="aggregations",
           visible_when=when.metric_allows_agg("metric")),
    # A second metric on the same figure, against its own right-hand axis. Both
    # sub-controls stay hidden until one is actually chosen.
    choice("metric2", "param.metric2", NO_METRIC,
           choices_from="activity_metrics_optional", help_key="param.metric2.help"),
    choice("aggregation2", "param.aggregation2", "sum",
           choices_from="aggregations",
           visible_when=when.all_of(
               when.ne("metric2", NO_METRIC),
               when.metric_allows_agg("metric2"),
           )),
    choice("chart2", "param.chart2", "line", choices=[
        Choice("line", "param.chart.line"),
        Choice("step", "param.chart.step"),
        Choice("bar", "param.chart.bar"),
        Choice("area", "param.chart.area"),
    ], visible_when=when.ne("metric2", NO_METRIC)),
    choice("granularity", "param.granularity", "week", choices_from="granularities"),
    choice("x_mode", "param.x_mode", "calendar", choices=[
        Choice("calendar", "param.x_mode.calendar"),
        Choice("elapsed", "param.x_mode.elapsed"),
    ], help_key="param.x_mode.help"),
    boolean("cumulative", "param.cumulative", False),
    choice("chart", "param.chart", "line", choices=[
        Choice("line", "param.chart.line"),
        Choice("step", "param.chart.step"),
        Choice("bar", "param.chart.bar"),
        Choice("area", "param.chart.area"),
    ]),
    boolean("markers", "param.markers", True),
    choice("split_by", "param.split_by", "none", choices=[
        Choice("none", "param.split_by.none"),
        Choice("sport_type", "param.split_by.sport"),
    ]),
    integer("smooth_rolling", "param.smooth_rolling", 0, min=0, max=52,
            help_key="param.smooth_rolling.help"),
    integer("smooth_savgol", "param.smooth_savgol", 0, min=0, max=101,
            help_key="param.smooth_savgol.help"),
    boolean("show_totals", "param.show_totals", False,
            help_key="param.show_totals.help"),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    metric = metric_or_default(params.get("metric"))
    granularity = params.get("granularity") or "week"
    x_mode = params.get("x_mode") or "calendar"
    chart_kind = _CHART_KINDS.get(params.get("chart") or "line", TraceKind.LINE)
    aggregation = params.get("aggregation") or metric.default_agg
    if metric.is_fixed_agg:
        aggregation = metric.default_agg

    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    notes: List[str] = []
    cumulative = bool(params.get("cumulative"))
    if cumulative and (metric.is_ratio or aggregation != "sum"):
        # A running total of averages is meaningless; say so rather than draw it.
        cumulative = False
        notes.append(translate("plot.trend.cumulative_ignored", lang))

    split_by_sport = (params.get("split_by") or "none") == "sport_type"
    smoothing = dict(
        rolling_window=int(params.get("smooth_rolling") or 0) or None,
        savgol_window=int(params.get("smooth_savgol") or 0) or None,
    )

    # Decided before drawing anything: the second metric changes how the *first*
    # one is encoded, so it cannot be an afterthought.
    metric2 = optional_metric(params.get("metric2"))
    dual = metric2 is not None and metric2.key != metric.key

    traces = _metric_traces(
        resolved, metric, aggregation, granularity, x_mode, chart_kind,
        cumulative=cumulative,
        split_by_sport=split_by_sport,
        markers=bool(params.get("markers", True)),
        smoothing=smoothing,
        lang=lang,
        axis="y",
        unify_color=_PRIMARY_COLOR if dual else None,
    )

    # The optional second metric, drawn against its own right-hand axis.
    y2_axis = None
    if dual:
        aggregation2 = params.get("aggregation2") or metric2.default_agg
        if metric2.is_fixed_agg:
            aggregation2 = metric2.default_agg
        chart_kind2 = _CHART_KINDS.get(params.get("chart2") or "line", TraceKind.LINE)

        traces += _metric_traces(
            resolved, metric2, aggregation2, granularity, x_mode, chart_kind2,
            cumulative=cumulative,
            split_by_sport=split_by_sport,
            markers=bool(params.get("markers", True)),
            smoothing=smoothing,
            lang=lang,
            axis="y2",
            unify_color=_SECONDARY_COLOR,
        )
        y2_axis = metric_axis(metric2, lang)
        y2_axis.color = _SECONDARY_COLOR

    left_axis = metric_axis(metric, lang)
    if dual:
        # Only tint when there is a second axis to tell it apart from.
        left_axis.color = _PRIMARY_COLOR

    chart = ChartData(
        title=_title(metric, lang, cumulative) if not dual
        else f"{_title(metric, lang, cumulative)} · {metric_label(metric2, lang)}",
        x_axis=_x_axis(resolved, x_mode, lang),
        y_axis=left_axis,
        y2_axis=y2_axis,
        traces=traces,
        caption=_caption(metric, aggregation, granularity, lang),
    )
    output = PlotOutput(charts=[chart], notes=notes)
    if params.get("show_totals"):
        output.tables.append(_totals_table(resolved, metric, aggregation, lang))
    return output


def _metric_traces(
    resolved: ResolvedPanelData,
    metric,
    aggregation: str,
    granularity: str,
    x_mode: str,
    chart_kind: TraceKind,
    *,
    cumulative: bool,
    split_by_sport: bool,
    markers: bool,
    smoothing: Dict[str, Optional[int]],
    lang: str,
    axis: str,
    unify_color: Optional[str] = None,
) -> List[Trace]:
    """Every series for one metric, bound to one y-axis.

    Extracted so the primary and the optional second metric go through exactly the
    same binning, cumulation and smoothing — the two cannot drift apart.

    ``unify_color`` switches the encoding for the dual-metric case: instead of one
    colour per group, every series of a metric takes that metric's colour and groups
    are told apart by dash. On a dual-axis chart the reader's first question is which
    axis a line is measured against, so colour has to answer that one — and it then
    matches the tint on the axis itself.
    """
    traces: List[Trace] = []
    series_index = 0

    for group in resolved.groups:
        group_frame = resolved.group_features(group)
        if group_frame.empty:
            continue
        for label, subset in _subsets(group, group_frame, split_by_sport):
            points = _bin_points(subset, group, metric, aggregation,
                                 granularity, x_mode)
            if not points:
                continue
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]

            if cumulative:
                y_values = _running_total(y_values)
                if x_mode == "elapsed":
                    # Start every window's cumulative line at the origin so the
                    # comparison is fair from the first day.
                    x_values = [0.0] + x_values
                    y_values = [0.0] + y_values

            y_values = smooth_uniform_series(y_values, **smoothing)

            traces.append(Trace(
                name=label,
                x=x_values,
                y=y_values,
                kind=chart_kind,
                color=(
                    series_color(series_index) if split_by_sport
                    else group_color(group.index)
                ),
                axis=axis,
                markers=markers and chart_kind is not TraceKind.BAR,
                hover_text=metric_hover_texts(metric, y_values),
                hover_template=hover_template(_x_hover(x_mode, lang)),
            ))
            series_index += 1

    if unify_color is not None:
        # Only in the two-metric case, so a single-metric chart keeps the group
        # colours it has always had.
        label = metric_label(metric, lang)
        for index, trace in enumerate(traces):
            trace.color = unify_color
            trace.dash = _GROUP_DASHES[index % len(_GROUP_DASHES)]
            # Name carries the metric, since colour now encodes it rather than group.
            trace.name = label if len(traces) == 1 else f"{trace.name} · {label}"
    return traces


def _subsets(
    group: ResolvedGroup, frame, split_by_sport: bool
) -> List[Tuple[str, Any]]:
    """The series to draw for one group: itself, or one per sport type."""
    if not split_by_sport:
        return [(group.label, frame)]
    return [
        (f"{group.label} · {sport}", subset)
        for sport, subset in frame.groupby("sport_type", sort=True)
    ]


def _bin_points(
    frame, group: ResolvedGroup, metric, aggregation: str,
    granularity: str, x_mode: str,
) -> List[Tuple[Any, float]]:
    """Bin the rows and aggregate each bin, returning ``(x, y)`` in x order."""
    dates = [to_date(d) for d in frame["date"]]
    if not dates:
        return []

    if x_mode == "elapsed":
        # Windows align to their own definition; a hand-picked selection has no
        # window, so it aligns to its own first activity.
        start = group.window.start if group.window else min(dates)
        keys = [elapsed_bin_index(d, start, granularity) for d in dates]
        pairs = metric.series_by_bin(frame, keys, aggregation)
        return [(elapsed_bin_months(int(k), granularity), v) for k, v in pairs]

    keys = [bin_start(d, granularity) for d in dates]
    pairs = metric.series_by_bin(frame, keys, aggregation)
    return [(datetime(k.year, k.month, k.day), v) for k, v in pairs]


def _running_total(values: List[float]) -> List[float]:
    total, out = 0.0, []
    for value in values:
        total += value
        out.append(total)
    return out


def _x_axis(resolved: ResolvedPanelData, x_mode: str, lang: str) -> Axis:
    if x_mode == "elapsed":
        windows = [g.window for g in resolved.groups if g.window]
        return Axis(
            title=translate("plot.x.months_since_start", lang),
            kind=AxisKind.LINEAR,
            dtick=1,
            range=[0, max_window_months(windows)] if windows else None,
        )
    return Axis(title=translate("plot.x.time", lang), kind=AxisKind.DATE)


def _x_hover(x_mode: str, lang: str) -> str:
    if x_mode == "elapsed":
        return f"%{{x:.1f}} {translate('plot.months', lang)}"
    return "%{x|%d %b %Y}"


def _title(metric, lang: str, cumulative: bool) -> str:
    label = metric_label(metric, lang)
    if cumulative:
        return f"{label} — {translate('param.cumulative', lang)}"
    return label


def _caption(metric, aggregation: str, granularity: str, lang: str) -> str:
    granularity_label = translate(f"gran.{granularity}", lang)
    if metric.is_fixed_agg:
        return granularity_label
    return f"{translate(f'agg.{aggregation}', lang)} · {granularity_label}"


def _totals_table(resolved: ResolvedPanelData, metric, aggregation: str, lang: str) -> TableData:
    """Per-group aggregate beside the curves — the old per-season totals table."""
    rows = []
    for group in resolved.groups:
        group_frame = resolved.group_features(group)
        if group_frame.empty:
            continue
        rows.append({
            "group": group.label,
            "value": metric.aggregate(group_frame, aggregation),
        })
    return TableData(
        title=translate("plot.trend.totals", lang),
        columns=[
            Column(key="group", label=translate("panel.group", lang),
                   format=CellFormat(kind="text")),
            Column(key="value", label=metric_label(metric, lang),
                   format=metric_cell_format(metric)),
        ],
        rows=list(reversed(rows)),  # newest window first reads better
        download_name="totals",
    )


register(PlotDefinition(
    key="metric_trend",
    label_key="plot.metric_trend.label",
    description_key="plot.metric_trend.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    category_key="plotcat.trends",
))
