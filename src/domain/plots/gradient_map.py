"""Where the time actually goes: share of moving time per gradient band.

A 100%-stacked area over the ``time_*`` feature columns, green (descent) → red
(ascent). It answers "am I really training for a vertical race, or just running
hills occasionally" in a way a single average-gradient number hides — two athletes
with the same mean gradient can have completely different band profiles.

One chart per data-source group, since stacking two groups in one figure would be
unreadable.
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from src.domain.charts.ir import (
    Axis,
    AxisKind,
    ChartData,
    PlotOutput,
    Trace,
    TraceKind,
    empty_output,
)
from src.domain.dataset.binning import bin_start, to_date
from src.domain.dataset.features import band_column
from src.domain.dataset.resolved import DataLevel, ResolvedGroup, ResolvedPanelData
from src.domain.plots.base import PlotDefinition, register
from src.domain.progress.models import GRADIENT_BAND_KEYS, GRADIENT_BANDS
from src.domain.spec.params import ParamSpec, choice, multichoice
from src.translations import translate

# Green (descent) → red (ascent). Kept here, next to the only plot that uses it.
BAND_COLORS = {
    "steep_descent": "#1B7A3D",
    "gentle_descent": "#7FB069",
    "flat": "#E8A33D",
    "gentle_ascent": "#C65D3B",
    "steep_ascent": "#8E2C18",
}

PARAMS: List[ParamSpec] = [
    choice("granularity", "param.granularity", "week", choices_from="granularities"),
    multichoice("bands", "param.bands", list(GRADIENT_BAND_KEYS),
                choices_from="gradient_bands", help_key="param.bands.help"),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    granularity = params.get("granularity") or "week"
    bands = [b for b in (params.get("bands") or GRADIENT_BAND_KEYS)
             if b in BAND_COLORS] or list(GRADIENT_BAND_KEYS)

    frame = resolved.features
    if frame.empty:
        return empty_output(translate("plot.no_data", lang))

    charts: List[ChartData] = []
    for group in resolved.groups:
        group_frame = resolved.group_features(group)
        if group_frame.empty:
            continue
        chart = _group_chart(group, group_frame, bands, granularity, lang,
                             titled=resolved.has_multiple_groups)
        if chart is not None:
            charts.append(chart)

    if not charts:
        return empty_output(translate("plot.gradient_map.no_stream_data", lang))
    return PlotOutput(charts=charts)


def _group_chart(
    group: ResolvedGroup, frame, bands: List[str], granularity: str,
    lang: str, *, titled: bool,
) -> ChartData:
    """Per-bin share of moving time in each band; bins with no time are dropped."""
    totals: Dict[Any, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        when = row.get("date")
        if not isinstance(when, datetime):
            continue
        key = bin_start(to_date(when), granularity)
        bucket = totals.setdefault(key, {band: 0.0 for band in bands})
        for band in bands:
            value = row.get(band_column(band))
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(seconds):
                bucket[band] += seconds

    x: List[datetime] = []
    shares: Dict[str, List[float]] = {band: [] for band in bands}
    for key in sorted(totals):
        total = sum(totals[key].values())
        if total <= 0:
            continue
        x.append(datetime(key.year, key.month, key.day))
        for band in bands:
            shares[band].append(totals[key][band] / total * 100.0)

    title = translate("plot.ltp.gradient_map.title", lang)
    if titled:
        title = f"{title} — {group.label}"

    traces = [
        Trace(
            name=translate(f"ltp.band.{band}", lang),
            x=list(x),
            y=shares[band],
            kind=TraceKind.AREA,
            color=BAND_COLORS[band],
            stack_group="bands",
            hover_template="%{x|%d %b %Y}<br>%{y:.0f} %<extra>%{fullData.name}</extra>",
        )
        # Keep the legend/stack in physical order (descent at the bottom).
        for band, _, _ in GRADIENT_BANDS if band in bands
    ]

    return ChartData(
        title=title,
        x_axis=Axis(title=translate("plot.ltp.gradient_map.x", lang), kind=AxisKind.DATE),
        y_axis=Axis(title=translate("plot.ltp.gradient_map.y", lang),
                    kind=AxisKind.LINEAR, range=[0, 100], suffix=" %"),
        traces=traces,
        hover_mode="x unified",
    )


register(PlotDefinition(
    key="gradient_map",
    label_key="plot.gradient_map.label",
    description_key="plot.gradient_map.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    requires_streams=True,
    category_key="plotcat.trends",
))
