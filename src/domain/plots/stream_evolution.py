"""Within-activity signal traces — the STREAM-level plot.

One line per activity, over elapsed time or distance covered: GAP pace, raw pace,
heart rate, power, power-to-HR, altitude or gradient. This is the whole race
comparator's set of evolution figures as a single plot type where the signal is a
parameter — and it generalizes, because any activity the panel selected can be
overlaid, not just races.

Per-signal smoothing is exposed because these traces are unusable raw: per-second
GPS pace and altitude are dominated by noise. The two-stage filter (time-domain
rolling mean, then a distance-domain Savitzky–Golay) is the same one the metrics
pipeline uses.
"""

from typing import Any, Dict, List, Optional, Tuple

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
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.plots.base import (
    SERIES_BY_ACTIVITY,
    PlotDefinition,
    register,
    series_color,
    weight_note,
)
from src.domain.plotting_common import fmt_hms, fmt_pace
from src.domain.races.metrics import compute_race
from src.domain.races.smoothing import FilterConfig, SmoothingParams, default_smoothing_params
from src.domain.spec.params import (
    Choice,
    ParamSpec,
    boolean,
    choice,
    group,
    integer,
    number,
    when,
)
from src.translations import translate

# signal key -> (RaceSeries attribute, y-label key, value kind, decimals, needs weight)
SIGNALS: Dict[str, Tuple[str, str, str, int, bool]] = {
    "gap_pace": ("gap_pace_s_per_km", "plot.races.gap_pace.y", "pace", 0, False),
    "pace": ("pace_s_per_km", "signal.pace.y", "pace", 0, False),
    "heartrate": ("heartrate", "plot.races.hr.y", "number", 0, False),
    "power": ("power_w", "plot.races.power.y", "number", 0, True),
    "power_to_hr": ("power_to_hr", "plot.races.p2hr.y", "number", 2, True),
    "altitude": ("altitude_m", "signal.altitude.y", "number", 0, False),
    "gradient": ("gradient_pct", "signal.gradient.y", "number", 1, False),
}

# x-axis key -> (attribute, label key, scale from SI, hover unit)
_X_AXES = {
    "time": ("time_s", "plot.races.x.time", 1.0 / 60.0, "min"),
    "distance": ("distance_m", "plot.races.x.distance", 1.0 / 1000.0, "km"),
}


def _filter_group(key: str, label_key: str, default: FilterConfig) -> ParamSpec:
    """Two windows per signal; 0 disables that stage."""
    return group(key, label_key, [
        number("rolling_window_s", "param.filter.rolling_s",
               float(default.rolling_window_s or 0.0), min=0, max=600, step=5),
        number("savgol_window_m", "param.filter.savgol_m",
               float(default.savgol_window_m or 0.0), min=0, max=5000, step=50),
    ])


_SMOOTH_DEFAULTS = default_smoothing_params()

PARAMS: List[ParamSpec] = [
    choice("signal", "param.signal", "gap_pace", choices=[
        Choice(key, f"signal.{key}") for key in SIGNALS
    ]),
    choice("x_axis", "param.x_axis", "time", choices=[
        Choice("time", "races.xaxis.time"),
        Choice("distance", "races.xaxis.distance"),
    ], help_key="races.xaxis.help"),
    boolean("as_speed", "param.as_speed", False, help_key="param.as_speed.help",
            visible_when=when.one_of("signal", ["gap_pace", "pace"])),
    integer("max_series", "param.max_series", 8, min=1, max=30,
            help_key="param.max_series.help"),
    group("smoothing", "param.smoothing", [
        _filter_group("pace", "races.signal.pace", _SMOOTH_DEFAULTS.pace),
        _filter_group("altitude", "races.signal.altitude", _SMOOTH_DEFAULTS.altitude),
        _filter_group("heartrate", "races.signal.hr", _SMOOTH_DEFAULTS.heartrate),
        _filter_group("power", "races.signal.power", _SMOOTH_DEFAULTS.power),
    ]),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    signal_key = params.get("signal") or "gap_pace"
    attribute, y_label_key, value_kind, decimals, needs_weight = SIGNALS.get(
        signal_key, SIGNALS["gap_pace"]
    )
    x_key = params.get("x_axis") or "time"
    x_attribute, x_label_key, x_scale, x_unit = _X_AXES.get(x_key, _X_AXES["time"])
    as_speed = bool(params.get("as_speed")) and value_kind == "pace"

    if needs_weight and resolved.mass_kg is None:
        return empty_output(weight_note(lang))

    activity_ids = [
        aid for aid in resolved.activity_ids
        if (stream := resolved.stream(aid)) is not None
        and getattr(stream, "has_streams", True)
    ]
    if not activity_ids:
        return empty_output(translate("plot.stream.no_stream_data", lang))

    notes: List[str] = []
    limit = int(params.get("max_series") or 8)
    if len(activity_ids) > limit:
        # Say what was dropped: a silently truncated overlay reads as complete.
        notes.append(translate("plot.stream.truncated", lang).format(
            shown=limit, total=len(activity_ids)))
        activity_ids = activity_ids[:limit]

    smoothing = _smoothing_from(params.get("smoothing") or {})
    smoothing_key = _smoothing_key(smoothing)

    traces: List[Trace] = []
    for index, activity_id in enumerate(activity_ids):
        series = _series_for(resolved, activity_id, smoothing, smoothing_key)
        if series is None:
            continue
        raw = getattr(series, attribute, None)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=float)
        if not np.isfinite(values).any():
            continue
        x = np.asarray(getattr(series, x_attribute), dtype=float) * x_scale
        y = 3600.0 / values if as_speed else values

        traces.append(Trace(
            name=resolved.activity_label(activity_id),
            x=x.tolist(),
            y=[None if not np.isfinite(v) else float(v) for v in y],
            kind=TraceKind.LINE,
            color=series_color(index),
            width=2.2,
            hover_text=_hover_texts(y, value_kind, as_speed, decimals),
            hover_template=(
                f"%{{x:.2f}} {x_unit}<br>%{{customdata}}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    if not traces:
        return empty_output(translate("plot.metric_unavailable", lang).format(
            metric=translate(f"signal.{signal_key}", lang)))

    y_title = translate("plot.races.gap_speed.y", lang) if as_speed \
        else translate(y_label_key, lang)
    chart = ChartData(
        title=translate(f"signal.{signal_key}", lang),
        x_axis=Axis(title=translate(x_label_key, lang), kind=AxisKind.LINEAR,
                    tick_format=",.1f"),
        y_axis=_y_axis(y_title, value_kind, as_speed, decimals),
        traces=traces,
        height=420,
    )
    return PlotOutput(charts=[chart], notes=notes)


def _y_axis(title: str, value_kind: str, as_speed: bool, decimals: int) -> Axis:
    if value_kind == "pace" and not as_speed:
        return Axis(title=title, kind=AxisKind.DURATION, reversed=True,
                    tick_format="%M:%S")
    return Axis(title=title, kind=AxisKind.LINEAR, tick_format=f",.{decimals}f")


def _hover_texts(values, value_kind: str, as_speed: bool, decimals: int) -> List[str]:
    if value_kind == "pace" and not as_speed:
        return [fmt_pace(v) if np.isfinite(v) else "—" for v in values]
    if as_speed:
        return [f"{v:,.1f} km/h" if np.isfinite(v) else "—" for v in values]
    return [f"{v:,.{decimals}f}" if np.isfinite(v) else "—" for v in values]


def _series_for(
    resolved: ResolvedPanelData, activity_id: int,
    smoothing: SmoothingParams, smoothing_key: tuple,
):
    """The per-second series for one activity, computed at most once per smoothing."""
    stream = resolved.stream(activity_id)
    if stream is None:
        return None

    def build():
        try:
            _, series = compute_race(
                stream, str(activity_id),
                mass_kg=resolved.mass_kg, smoothing=smoothing,
            )
            return series
        except ValueError:
            return None  # too few samples to analyse

    return resolved.memo(("stream_series", activity_id, resolved.mass_kg, smoothing_key), build)


def _smoothing_from(raw: Dict[str, Any]) -> SmoothingParams:
    """Rebuild a :class:`SmoothingParams` from the plot's nested parameter values."""
    def config(key: str, default: FilterConfig) -> FilterConfig:
        values = raw.get(key) or {}
        rolling = _positive(values.get("rolling_window_s"), default.rolling_window_s)
        savgol = _positive(values.get("savgol_window_m"), default.savgol_window_m)
        return FilterConfig(rolling_window_s=rolling, savgol_window_m=savgol)

    return SmoothingParams(
        pace=config("pace", _SMOOTH_DEFAULTS.pace),
        altitude=config("altitude", _SMOOTH_DEFAULTS.altitude),
        heartrate=config("heartrate", _SMOOTH_DEFAULTS.heartrate),
        power=config("power", _SMOOTH_DEFAULTS.power),
    )


def _positive(value: Any, fallback: Optional[float]) -> Optional[float]:
    """0 means "filter off"; a missing value falls back to the default."""
    if value is None:
        return fallback
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return fallback
    return number_value if number_value > 0 else None


def _smoothing_key(smoothing: SmoothingParams) -> tuple:
    return tuple(
        (config.rolling_window_s, config.savgol_window_m)
        for config in (smoothing.pace, smoothing.altitude,
                       smoothing.heartrate, smoothing.power)
    )


register(PlotDefinition(
    key="stream_evolution",
    label_key="plot.stream_evolution.label",
    description_key="plot.stream_evolution.description",
    level=DataLevel.STREAM,
    compute=compute,
    params=PARAMS,
    series_level=SERIES_BY_ACTIVITY,
    requires_streams=True,
    category_key="plotcat.within",
))
