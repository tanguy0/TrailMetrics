"""Within-activity signal traces — the STREAM-level plot.

One line per activity per signal, over elapsed time or distance covered: GAP pace,
raw pace, heart rate, power, power-to-HR, altitude or gradient. This is the whole
race comparator's set of evolution figures as a single plot type where the signals
are a parameter — and it generalizes, because any activity the panel selected can
be overlaid, not just races.

Any number of signals can be selected at once. There are only two y-axes in the
chart IR, so signals are bucketed onto them by ``value_kind`` (their unit) rather
than one-per-axis: every "pace" signal shares the axis GAP would use alone, every
"number" signal shares the other. That is a real simplification — heart rate and
altitude sharing an axis despite different units — but it is the one this plot
type already made for its dual-axis case, just no longer capped at exactly two
signals. A panel that wants heart rate read against its own untouched axis should
give it a panel of its own.

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
from src.domain.gap import theme
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
    multichoice,
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
    "power_per_kg": ("power_per_kg", "plot.races.power_per_kg.y", "number", 2, True),
    "power_to_hr": ("power_to_hr", "plot.races.p2hr.y", "number", 2, True),
    "altitude": ("altitude_m", "signal.altitude.y", "number", 0, False),
    "gradient": ("gradient_pct", "signal.gradient.y", "number", 1, False),
}

_DEFAULT_SIGNALS = ["gap_pace"]

# Heart rate reads as the athlete's *response* to an effort, never the effort
# itself, so it keeps one fixed identity everywhere it appears: its own axis,
# always on the right, always this red. The one signal paired against it on a
# per-sport session view — pace, GAP or power — gets the brand green in that
# exact pairing, so the two-line "effort vs response" chart reads the same way
# for a run, a ride, a hike or a swim. A custom multi-signal panel that adds a
# third signal alongside them falls back to the ordinary palette for it, since
# there is no longer one single "other metric" to force a color onto.
_EFFORT_SIGNALS = frozenset({"gap_pace", "pace", "power"})

# x-axis key -> (attribute, label key, scale from SI, hover unit)
_X_AXES = {
    "time": ("time_s", "plot.races.x.time", 1.0 / 60.0, "min"),
    "distance": ("distance_m", "plot.races.x.distance", 1.0 / 1000.0, "km"),
}

# Dashes distinguish the *activities* once colour is spent on the signal.
_ACTIVITY_DASHES = ["-", "--", "-.", ":"]

# One entry per signal sharing a y-axis: its legend label, unit, tick decimals,
# whether it is being shown as speed rather than pace, and its signal key (so the
# axis title can tell a plain "pace" from "gap_pace" — see _combined_axis).
_AxisEntry = Tuple[str, str, int, bool, str]


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
    multichoice("signals", "param.signals", list(_DEFAULT_SIGNALS), choices=[
        Choice(key, f"signal.{key}") for key in SIGNALS
    ], help_key="param.signals.help"),
    choice("x_axis", "param.x_axis", "time", choices=[
        Choice("time", "races.xaxis.time"),
        Choice("distance", "races.xaxis.distance"),
    ], help_key="races.xaxis.help"),
    boolean("as_speed", "param.as_speed", False, help_key="param.as_speed.help",
            visible_when=when.any_of(
                when.contains("signals", "gap_pace"),
                when.contains("signals", "pace"),
            )),
    integer("max_series", "param.max_series", 8, min=1, max=30,
            help_key="param.max_series.help"),
    group("smoothing", "param.smoothing", [
        _filter_group("pace", "races.signal.pace", _SMOOTH_DEFAULTS.pace),
        _filter_group("altitude", "races.signal.altitude", _SMOOTH_DEFAULTS.altitude),
        _filter_group("heartrate", "races.signal.hr", _SMOOTH_DEFAULTS.heartrate),
        _filter_group("power", "races.signal.power", _SMOOTH_DEFAULTS.power),
    ]),
]


def _selected_signals(params: Dict[str, Any]) -> List[str]:
    """The chosen signals, valid and de-duplicated, in the order picked."""
    raw = params.get("signals") or []
    ordered: List[str] = []
    for key in raw:
        if key in SIGNALS and key not in ordered:
            ordered.append(key)
    return ordered or list(_DEFAULT_SIGNALS)


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    signal_keys = _selected_signals(params)
    x_key = params.get("x_axis") or "time"
    x_attribute, x_label_key, x_scale, x_unit = _X_AXES.get(x_key, _X_AXES["time"])
    as_speed = bool(params.get("as_speed"))

    notes: List[str] = []
    limit = int(params.get("max_series") or 8)

    # Stop fetching once `limit` activities have a usable stream. The panel's
    # selection can be a whole multi-year window while the overlay draws eight
    # lines, and every stream this touches is decoded and then held in the render
    # memo — so loading the selection to pick a prefix of it costs hundreds of
    # megabytes to throw almost all of them away.
    # One past the limit, so "there are more than we drew" is known without
    # fetching the rest.
    activity_ids: List[int] = []
    for aid in resolved.activity_ids:
        stream = resolved.stream(aid)
        if stream is None or not getattr(stream, "has_streams", True):
            continue
        activity_ids.append(aid)
        if len(activity_ids) > limit:
            break
    if not activity_ids:
        return empty_output(translate("plot.stream.no_stream_data", lang))

    if len(activity_ids) > limit:
        # Say what was dropped: a silently truncated overlay reads as complete.
        # `total` counts the selection, not the streams fetched — stopping early
        # is what this loop is for, so the number of *usable* ones past the limit
        # is deliberately never established.
        notes.append(translate("plot.stream.truncated", lang).format(
            shown=limit, total=len(resolved.activity_ids)))
        activity_ids = activity_ids[:limit]

    # A single activity's own name is not useful information on every one of its
    # traces — the legend only needs to say *which signal* a line is, not repeat the
    # one session everything on the chart already belongs to.
    single_activity = len(activity_ids) == 1

    smoothing = _smoothing_from(params.get("smoothing") or {})
    smoothing_key = _smoothing_key(smoothing)

    # A signal that needs a weight we don't have is dropped, with one note covering
    # all of them — not one per signal, which would repeat the same explanation.
    resolved_signals = []
    missing_weight = False
    for key in signal_keys:
        attribute, y_label_key, value_kind, decimals, needs_weight = SIGNALS[key]
        if needs_weight and resolved.mass_kg is None:
            missing_weight = True
            continue
        resolved_signals.append((key, attribute, y_label_key, value_kind, decimals))
    if missing_weight:
        notes.append(weight_note(lang))
    if not resolved_signals:
        return empty_output(weight_note(lang))

    multi_signal = len(resolved_signals) > 1
    resolved_keys = [key for key, *_ in resolved_signals]
    # Heart rate never shares an axis with anything else it's plotted alongside —
    # forced to the right regardless of its (coincidentally shared) "number"
    # value_kind — as long as there is another signal there to leave the left
    # axis to. Selected on its own, it has nowhere else to go.
    has_other_signal = any(key != "heartrate" for key in resolved_keys)
    # Exactly one effort signal paired one-to-one against heart rate — the shape
    # of every per-sport session chart — gets fixed colors instead of the
    # general palette, so "effort vs response" reads the same everywhere.
    paired_with_hr = len(resolved_signals) == 2 and "heartrate" in resolved_keys

    # Bucket onto (at most) two axes by unit, in the order the signals were picked —
    # see the module docstring for why a third distinct unit would still land on
    # the second axis rather than being dropped. Heart rate's kind is skipped here
    # whenever something else is selected too, so it can never claim the primary
    # axis out from under the signal it is forced off of above.
    axis_kinds: List[str] = []
    for key, _, _, value_kind, _ in resolved_signals:
        if key == "heartrate" and has_other_signal:
            continue
        if value_kind not in axis_kinds:
            axis_kinds.append(value_kind)
    primary_kind = axis_kinds[0] if axis_kinds else SIGNALS["heartrate"][2]

    traces: List[Trace] = []
    primary_entries: List[_AxisEntry] = []
    secondary_entries: List[_AxisEntry] = []
    # A single-colour axis tint only still means something when exactly one signal
    # is on that axis; two differently-coloured signals sharing it can't be tinted
    # to either one, so it stays untinted.
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None

    for index, (key, attribute, y_label_key, value_kind, decimals) in enumerate(resolved_signals):
        on_primary = value_kind == primary_kind and not (key == "heartrate" and has_other_signal)
        signal_as_speed = as_speed and value_kind == "pace"
        if key == "heartrate" and has_other_signal:
            color = theme.DANGER
        elif paired_with_hr and key in _EFFORT_SIGNALS:
            color = theme.PRIMARY
        else:
            color = series_color(index) if multi_signal else None
        label = translate(f"signal.{key}", lang)

        traces += _signal_traces(
            resolved, activity_ids, attribute, x_attribute, x_scale, x_unit,
            value_kind, decimals, as_speed=signal_as_speed,
            smoothing=smoothing, smoothing_key=smoothing_key,
            axis="y" if on_primary else "y2",
            unify_color=color,
            signal_label=label,
            single_activity=single_activity,
        )

        entry: _AxisEntry = (label, value_kind, decimals, signal_as_speed, key)
        if on_primary:
            primary_entries.append(entry)
            primary_color = color if len(primary_entries) == 1 else None
        else:
            secondary_entries.append(entry)
            secondary_color = color if len(secondary_entries) == 1 else None

    if not traces:
        return empty_output(translate("plot.metric_unavailable", lang).format(
            metric=translate(f"signal.{signal_keys[0]}", lang)))

    left_axis = _combined_axis(primary_entries, lang)
    y2_axis = _combined_axis(secondary_entries, lang) if secondary_entries else None
    if y2_axis is not None:
        left_axis.color = primary_color
        y2_axis.color = secondary_color

    title = " · ".join(label for label, *_ in primary_entries + secondary_entries)

    chart = ChartData(
        title=title,
        x_axis=Axis(title=translate(x_label_key, lang), kind=AxisKind.LINEAR,
                    tick_format=",.1f"),
        y_axis=left_axis,
        y2_axis=y2_axis,
        traces=traces,
        height=420,
    )
    return PlotOutput(charts=[chart], notes=notes)


def _combined_axis(entries: List[_AxisEntry], lang: str) -> Axis:
    """One y-axis for every signal sharing it, formatted like the first of them.

    Several signals of the same ``value_kind`` can still differ in decimals or
    natural unit (altitude in metres, gradient in percent); the first signal
    assigned to the axis decides the tick formatting, and the title lists all of
    them so the reader knows what else is drawn against it.
    """
    _, value_kind, decimals, as_speed, _ = entries[0]
    has_gap = any(key == "gap_pace" for *_, key in entries)
    if as_speed:
        title = translate(
            "plot.races.gap_speed.y" if has_gap else "plot.races.speed.y", lang,
        )
    else:
        title = " · ".join(label for label, *_ in entries)
    return _y_axis(title, value_kind, as_speed, decimals)


def _signal_traces(
    resolved: ResolvedPanelData,
    activity_ids: List[int],
    attribute: str,
    x_attribute: str,
    x_scale: float,
    x_unit: str,
    value_kind: str,
    decimals: int,
    *,
    as_speed: bool,
    smoothing: SmoothingParams,
    smoothing_key: tuple,
    axis: str,
    unify_color: Optional[str],
    signal_label: str,
    single_activity: bool,
) -> List[Trace]:
    """One line per activity for a single signal, bound to one y-axis.

    With more than one signal selected, colour has to say *which signal* a line
    is, so activities are told apart by dash instead and the trace name carries
    both. With one signal, per-activity colours are kept.

    ``single_activity`` overrides all of that: with only one activity on the whole
    panel, every trace already belongs to the one session on the chart, so the name
    is just the signal — repeating the session's own label on every line would be
    the only thing in the legend, telling the reader nothing.
    """
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

        if single_activity:
            name = signal_label
        else:
            activity_name = resolved.activity_label(activity_id)
            name = f"{activity_name} · {signal_label}" if unify_color else activity_name
        traces.append(Trace(
            name=name,
            x=x.tolist(),
            y=[None if not np.isfinite(v) else float(v) for v in y],
            kind=TraceKind.LINE,
            color=unify_color or series_color(index),
            axis=axis,
            dash=_ACTIVITY_DASHES[index % len(_ACTIVITY_DASHES)] if unify_color else "-",
            width=8.8,
            hover_text=_hover_texts(y, value_kind, as_speed, decimals),
            hover_template=(
                f"%{{x:.2f}} {x_unit}<br>%{{customdata}}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))
    return traces


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
