"""Personalized GAP curves — the SPLIT-level plot.

Fits the athlete's own gradient-adjusted-pace models on the splits pooled across
each data-source group, and overlays the published reference curves. Colour
encodes the group (this block vs that one), line style encodes the model and the
heart-rate band, so a single figure can carry "efficiency vs auto-learning, easy
vs hard, 2024 vs 2025" without a legend the reader has to decode twice.

Optional HR bands replace what used to be two extra hard-wired figures: leave the
list empty for one curve per model, or add named bands to stratify by intensity.

This is the app's one **expensive** plot — it fits models rather than aggregating
rows — so the panel editor gives it a manual refresh and every fit is memoized on
its inputs.
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
from src.domain.dataset.resolved import DataLevel, ResolvedGroup, ResolvedPanelData
from src.domain.gap import theme
from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.preprocessing import DefaultStreamPreprocessor
from src.domain.gap.reference_curves import balanced_runner, kilian_jornet
from src.domain.gap.smoothing import LoessCurveSmoother
from src.domain.models.gap import DownsampledDataset, GapCurve
from src.domain.plots.base import (
    EXPENSIVE,
    PlotDefinition,
    group_color,
    register,
)
from src.domain.spec.params import (
    Choice,
    ParamSpec,
    boolean,
    integer,
    multichoice,
    number,
    rows,
    text,
    when,
)
from src.translations import translate

EFFICIENCY = "efficiency"
AUTO_LEARNING = "auto_learning"

# Line styles distinguish model × HR band within one group's colour.
_DASHES = ["-", "--", "-.", ":"]

_REFERENCES = {
    "balanced_runner": (balanced_runner, theme.BALANCED_RUNNER, "gap.refs.balanced"),
    "kilian": (kilian_jornet, theme.KILIAN, "gap.refs.kilian"),
}


_USES_EFFICIENCY = when.contains("models", EFFICIENCY)
_USES_AUTO_LEARNING = when.contains("models", AUTO_LEARNING)
_STRATIFIED = when.nonempty("hr_bands")

PARAMS: List[ParamSpec] = [
    multichoice("models", "param.gap_models", [EFFICIENCY, AUTO_LEARNING], choices=[
        Choice(EFFICIENCY, "gap.models.efficiency"),
        Choice(AUTO_LEARNING, "gap.models.auto"),
    ], help_key="gap.models.caption"),
    multichoice("references", "param.gap_references",
                ["balanced_runner", "kilian"], choices=[
                    Choice("balanced_runner", "gap.refs.balanced"),
                    Choice("kilian", "gap.refs.kilian"),
                ], help_key="gap.refs.caption"),
    boolean("show_std", "gap.display.show_std", True,
            help_key="gap.display.show_std_help"),
    rows("hr_bands", "param.hr_bands", [
        text("name", "param.hr_band.name"),
        integer("hr_min", "param.hr_band.min", 120, min=60, max=250),
        integer("hr_max", "param.hr_band.max", 150, min=60, max=250),
    ], default=[], max_items=4, help_key="param.hr_bands.help"),
    number("split_min_time", "gap.params.split_min_time", 10.0, min=1, max=300, step=1),
    integer("efficiency_min_samples", "gap.params.eff_min_samples", 250,
            min=10, max=5000, visible_when=_USES_EFFICIENCY),
    integer("efficiency_band_min_samples", "gap.params.eff_subset_min_samples", 50,
            min=5, max=2000, help_key="gap.params.eff_subset_help",
            visible_when=when.all_of(_USES_EFFICIENCY, _STRATIFIED)),
    number("hr_tolerance", "gap.params.hr_tol", 3.0, min=1, max=30, step=1,
           visible_when=_USES_AUTO_LEARNING),
    number("xgb_bin_width", "gap.params.bin_width", 20.0, min=1, max=200, step=1,
           visible_when=_USES_AUTO_LEARNING),
]

_PREPROCESSOR = DefaultStreamPreprocessor()
_SMOOTHER = LoessCurveSmoother(bandwidth_fraction=0.4, polyorder=2)


class CurveUnavailable(Exception):
    """One curve could not be produced, with a translatable reason.

    Raised rather than returning ``None`` so that a requested model or band always
    gets an explanation. A selected model that quietly draws nothing is worse than
    an error: the reader assumes the two models agree.
    """

    def __init__(self, reason_key: str):
        super().__init__(reason_key)
        self.reason_key = reason_key


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    models = [m for m in (params.get("models") or []) if m in (EFFICIENCY, AUTO_LEARNING)]
    bands = _bands(params.get("hr_bands") or [])
    show_std = bool(params.get("show_std", True))

    traces: List[Trace] = []
    notes: List[str] = []
    summaries: List[str] = []

    if models:
        for group in resolved.groups:
            dataset = _dataset(resolved, group, float(params.get("split_min_time") or 10.0))
            if dataset is None or dataset.speed.size == 0:
                notes.append(translate("gap.group_no_splits", lang).format(
                    label=group.label))
                continue
            summaries.append(translate("gap.summary.item", lang).format(
                label=group.label, n=int(dataset.speed.size)))
            traces.extend(_group_traces(
                resolved, group, dataset, models, bands, params, show_std, lang, notes
            ))

    for key in (params.get("references") or []):
        reference = _REFERENCES.get(key)
        if reference is None:
            continue
        factory, color, label_key = reference
        traces.append(_curve_trace(
            factory(), translate(label_key, lang), color, "--", show_std
        ))

    if not traces:
        return empty_output(translate("gap.nothing_to_plot", lang))

    if summaries:
        notes.insert(0, translate("gap.summary", lang).format(
            summary=", ".join(summaries)))

    chart = ChartData(
        title=translate("plot.gap.title_std" if show_std else "plot.gap.title", lang),
        x_axis=Axis(title=translate("plot.gap.xlabel", lang), kind=AxisKind.LINEAR,
                    tick_format=",.0f"),
        y_axis=Axis(title=translate("plot.gap.ylabel", lang), kind=AxisKind.LINEAR,
                    tick_format=".3f"),
        traces=traces,
        caption=translate("gap.caption.main", lang) if models else None,
    )
    return PlotOutput(charts=[chart], notes=notes)


# --- Model fitting ---------------------------------------------------------

def _dataset(
    resolved: ResolvedPanelData, group: ResolvedGroup, split_min_time: float
) -> Optional[DownsampledDataset]:
    """Pooled split-level samples for one group, computed once per (group, window)."""
    streams = [
        s for s in resolved.group_streams(group)
        if getattr(s, "has_streams", True)
    ]
    if not streams:
        return None

    def build():
        try:
            return _PREPROCESSOR.process_many(
                streams, split_min_time=split_min_time, verbose=False
            )
        except (ValueError, IndexError):
            # No activity yielded a usable split (too short, no HR, all mixed).
            return None

    key = ("gap_dataset", tuple(sorted(s.activity_id for s in streams)), split_min_time)
    return resolved.memo(key, build)


def _group_traces(
    resolved: ResolvedPanelData, group: ResolvedGroup, dataset: DownsampledDataset,
    models: List[str], bands: List[Tuple[str, float, float]], params: Dict[str, Any],
    show_std: bool, lang: str, notes: List[str],
) -> List[Trace]:
    """Every curve for one group: each selected model, each HR band (or none)."""
    out: List[Trace] = []
    color = group_color(group.index)
    band_slots = bands or [(None, None, None)]

    for model_index, model in enumerate(models):
        model_label = translate(
            "gap.models.efficiency" if model == EFFICIENCY else "gap.models.auto", lang
        )
        for band_index, (band_name, hr_min, hr_max) in enumerate(band_slots):
            dash = _DASHES[(model_index * len(band_slots) + band_index) % len(_DASHES)]
            hr_range = (hr_min, hr_max) if band_name else None
            name = _series_name(group.label, model_label, band_name)
            try:
                curve = _curve(resolved, group, dataset, model, hr_range, params)
                if not _has_points(curve):
                    # An HR band outside the athlete's actual range fits without
                    # raising and yields empty arrays; that is a real answer, but
                    # it must be said rather than drawn as an empty legend entry.
                    raise CurveUnavailable("gap.reason.empty_curve")
            except CurveUnavailable as unavailable:
                notes.append(translate("gap.curve_unavailable", lang).format(
                    label=name, error=translate(unavailable.reason_key, lang),
                ))
                continue
            except Exception as error:
                notes.append(translate("gap.curve_unavailable", lang).format(
                    label=name, error=error,
                ))
                continue
            out.append(_curve_trace(curve, name, color, dash, show_std))
    return out


def _has_points(curve: Optional[GapCurve]) -> bool:
    """Whether a curve carries at least one finite point worth drawing."""
    if curve is None:
        return False
    means = np.asarray(curve.means, dtype=float)
    return means.size > 0 and bool(np.isfinite(means).any())


def _series_name(group_label: str, model_label: str, band_name: Optional[str]) -> str:
    parts = [group_label, model_label]
    if band_name:
        parts.append(band_name)
    return " – ".join(p for p in parts if p)


def _curve(
    resolved: ResolvedPanelData, group: ResolvedGroup, dataset: DownsampledDataset,
    model: str, hr_range: Optional[Tuple[float, float]], params: Dict[str, Any],
) -> Optional[GapCurve]:
    ids = tuple(sorted(group.activity_ids))
    if model == EFFICIENCY:
        min_samples = int(
            params.get("efficiency_band_min_samples") if hr_range
            else params.get("efficiency_min_samples") or 250
        )
        key = ("gap_eff", ids, params.get("split_min_time"), min_samples, hr_range)

        def build_efficiency():
            fitted = EfficiencyGapModel(min_samples_per_bucket=min_samples)
            model_fit = (
                fitted.fit_on_subset(dataset, heartrate_range=hr_range)
                if hr_range else fitted.fit(dataset)
            )
            return _SMOOTHER.smooth(model_fit.gap_curve())

        return resolved.memo(key, build_efficiency)

    bin_width = float(params.get("xgb_bin_width") or 20.0)
    tolerance = float(params.get("hr_tolerance") or 3.0)
    fitted, reason = _xgboost_model(resolved, group, dataset, tolerance, params)
    if fitted is None:
        raise CurveUnavailable(reason or "gap.reason.no_calibration")
    key = ("gap_xgb_curve", ids, params.get("split_min_time"), tolerance,
           bin_width, hr_range)
    return resolved.memo(
        key,
        lambda: _SMOOTHER.smooth(
            fitted.gap_curve(bin_width=bin_width, heartrate_range=hr_range)
        ),
    )


def _xgboost_model(
    resolved: ResolvedPanelData, group: ResolvedGroup, dataset: DownsampledDataset,
    tolerance: float, params: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[str]]:
    """Fit (once) the auto-learning model, or say why it could not be fitted.

    Returns ``(model, reason_key)``. The failure is memoized along with the
    success, so a group that cannot be calibrated is established once per render
    context rather than re-derived from the splits on every parameter tweak.
    """
    from src.domain.gap.xgboost_model import XgboostGapModel

    ids = tuple(sorted(group.activity_ids))
    key = ("gap_xgb_model", ids, params.get("split_min_time"), tolerance)

    def build():
        features, targets, weights = _PREPROCESSOR.prepare_calibration_dataset(
            dataset, hr_tolerance=tolerance
        )
        if features.size == 0:
            # No flat section shares a heart rate with any climbing section, so
            # there is nothing to learn the adjustment from.
            return (None, "gap.reason.no_calibration")
        return (XgboostGapModel().fit(features, targets, sample_weight=weights), None)

    return resolved.memo(key, build)


# --- IR conversion ---------------------------------------------------------

def _curve_trace(
    curve: GapCurve, name: str, color: str, dash: str, show_std: bool
) -> Trace:
    centers = np.asarray(curve.bin_centers, dtype=float)
    means = np.asarray(curve.means, dtype=float)
    upper = lower = None
    if show_std:
        stds = np.asarray(curve.stds, dtype=float)
        if stds.size == means.size:
            upper = (means + stds).tolist()
            lower = (means - stds).tolist()
    return Trace(
        name=name,
        x=centers.tolist(),
        y=means.tolist(),
        kind=TraceKind.LINE,
        color=curve.color or color,
        dash=dash,
        width=11.2,
        band_upper=upper,
        band_lower=lower,
        hover_template="%{x:.0f} m/km<br>%{y:.3f}<extra>%{fullData.name}</extra>",
    )


def _bands(raw: List[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    """Named HR bands from the parameter rows, invalid/unnamed ones dropped."""
    out: List[Tuple[str, float, float]] = []
    for index, row in enumerate(raw):
        name = str(row.get("name") or "").strip()
        try:
            low = float(row.get("hr_min"))
            high = float(row.get("hr_max"))
        except (TypeError, ValueError):
            continue
        if low >= high:
            continue
        out.append((name or f"{low:.0f}–{high:.0f} bpm", low, high))
        if index >= 3:
            break
    return out


register(PlotDefinition(
    key="gap_curve",
    label_key="plot.gap_curve.label",
    description_key="plot.gap_curve.description",
    level=DataLevel.SPLIT,
    compute=compute,
    params=PARAMS,
    requires_streams=True,
    cost=EXPENSIVE,
    category_key="plotcat.models",
))
