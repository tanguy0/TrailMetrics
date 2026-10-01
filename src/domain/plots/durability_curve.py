"""Durability curve — how the athlete's running cost drifts over a long effort.

Fits the same hierarchical durability model the race plan uses
(:mod:`src.domain.durability`) on each group's long runs, **restricted to the past
year** whatever the data source says (older runs say little about current
durability), and draws two charts:

1. *Observed vs modelled drift.* Every steady 5-minute segment's cost drift since
   the start of its run (HR reserve per GAP speed, corrected for population
   cardiac drift), binned by elapsed time: the median with its interquartile band,
   against what the population prior and the personalized model predict for the
   very same segments. This is the evidence, and the check on the fit.
2. *Projected durability.* ``phi(t) − 1`` for a steady flat run at the athlete's
   typical long-run intensity, population vs personalized — the curve the race
   plan's pacing bends along.

A table of coefficients and the model's confidence notes close the output.
"""

from dataclasses import replace
from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np

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
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.durability.capability import sustainable_fraction
from src.domain.durability.config import (
    COMPONENTS,
    DEFAULT_CONFIG,
    DURATION,
    DurabilityCoefficients,
    DurabilityConfig,
)
from src.domain.durability.model import accumulate_exposures, durability_profile
from src.domain.durability.personalization import POPULATION_ONLY, AthleteDurabilityModel
from src.domain.durability.report import durability_notes
from src.domain.durability.segments import IDENTIFIABLE, design_matrix
from src.domain.gap import theme
from src.domain.plots.base import EXPENSIVE, PlotDefinition, group_color, register
from src.domain.plotting_common import fmt_pace
from src.domain.spec.params import ParamSpec, boolean, integer
from src.translations import translate
from src.domain.durability.history import fit_athlete_durability

PARAMS: List[ParamSpec] = [
    integer("lookback_days", "param.durability.lookback", 365, min=30, max=365,
            help_key="param.durability.lookback.help"),
    integer("min_run_minutes", "param.durability.min_run", 45, min=30, max=240),
    integer("bin_minutes", "param.durability.bin", 20, min=5, max=60),
    boolean("show_observed", "param.durability.show_observed", True),
]

# Each run's drift is measured from the mean of its first few valid segments.
_REFERENCE_SEGMENTS = 3
# A bin is drawn only with this much evidence behind it.
_MIN_BIN_SEGMENTS = 5
_MIN_BIN_ACTIVITIES = 2
_PROJECTION_MIN_H = 4.0
_PROJECTION_MAX_H = 12.0
# Intensity for the projection when there are no segments: race effort for 3 h.
_FALLBACK_PROJECTION_S = 3 * 3600.0


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    lang = resolved.lang
    config = _config(params)
    today = date.today()
    segment_memo = resolved.memo(("durability_segment_memo",), dict)

    fitted: List[Tuple[Any, AthleteDurabilityModel]] = []
    for group in resolved.groups:
        key = ("durability_model", tuple(sorted(group.activity_ids)), today,
               config.personalization.lookback_days, config.personalization.min_activity_moving_s)
        model = resolved.memo(key, lambda g=group: fit_athlete_durability(
            resolved.data, today, config, activity_ids=g.activity_ids, memo=segment_memo,
        ))
        fitted.append((group, model))
    if not fitted:
        return empty_output(translate("durability.nothing", lang))

    drift = _drift_chart(fitted, config, params, lang)
    projection = _projection_chart(fitted, config, lang)
    notes: List[str] = [translate("durability.note.past_year", lang).format(
        days=config.personalization.lookback_days)]
    tables: List[TableData] = []
    for group, model in fitted:
        prefix = f"{group.label} — " if len(fitted) > 1 else ""
        notes.extend(prefix + n for n in _summary_notes(model, lang))
        tables.append(_coefficient_table(model, prefix, lang))
    charts = [c for c in (drift, projection) if c is not None]
    return PlotOutput(charts=charts, tables=tables, notes=notes)


def _config(params: Dict[str, Any]) -> DurabilityConfig:
    settings = replace(
        DEFAULT_CONFIG.personalization,
        lookback_days=int(min(365, params.get("lookback_days") or 365)),
        min_activity_moving_s=float(params.get("min_run_minutes") or 45) * 60.0,
    )
    return replace(DEFAULT_CONFIG, personalization=settings)


# --- 1. Observed vs modelled drift -------------------------------------------

def _drift_chart(fitted, config: DurabilityConfig, params: Dict[str, Any], lang: str):
    bin_h = float(params.get("bin_minutes") or 20) / 60.0
    show_observed = bool(params.get("show_observed", True))
    traces: List[Trace] = []
    for group, model in fitted:
        if not model.segments:
            continue
        color = group_color(group.index)
        label = group.label if len(fitted) > 1 else ""
        hours, observed, population, personal, activity = _anchored(model)
        bins = np.floor(hours / bin_h).astype(int)
        centers, obs_q, pop_mean, ind_mean = [], [], [], []
        for b in np.unique(bins):
            inside = bins == b
            if inside.sum() < _MIN_BIN_SEGMENTS or \
                    np.unique(activity[inside]).size < _MIN_BIN_ACTIVITIES:
                continue
            centers.append((b + 0.5) * bin_h)
            obs_q.append(np.percentile(observed[inside], [25, 50, 75]) * 100)
            pop_mean.append(float(np.mean(population[inside])) * 100)
            ind_mean.append(float(np.mean(personal[inside])) * 100)
        if not centers:
            continue
        obs_q = np.array(obs_q)
        if show_observed:
            traces.append(Trace(
                name=_name(label, translate("durability.series.observed", lang)),
                x=[round(float(c), 3) for c in centers],
                y=obs_q[:, 1].round(2).tolist(),
                kind=TraceKind.LINE,
                color=color,
                markers=True,
                width=1.5,
                band_upper=obs_q[:, 2].round(2).tolist(),
                band_lower=obs_q[:, 0].round(2).tolist(),
                hover_template="%{x:.2f} h<br>%{y:+.1f} %<extra>%{fullData.name}</extra>",
            ))
        traces.append(Trace(
            name=_name(label, translate("durability.series.population", lang)),
            x=[round(float(c), 3) for c in centers], y=[round(v, 2) for v in pop_mean],
            kind=TraceKind.LINE, color=theme.BALANCED_RUNNER, dash="--", width=1.5,
            hover_template="%{x:.2f} h<br>%{y:+.1f} %<extra>%{fullData.name}</extra>",
        ))
        if model.confidence != POPULATION_ONLY:
            traces.append(Trace(
                name=_name(label, translate("durability.series.personal", lang)),
                x=[round(float(c), 3) for c in centers], y=[round(v, 2) for v in ind_mean],
                kind=TraceKind.LINE, color=color, width=2.0,
                hover_template="%{x:.2f} h<br>%{y:+.1f} %<extra>%{fullData.name}</extra>",
            ))
    if not traces:
        return None
    return ChartData(
        title=translate("durability.chart.drift", lang),
        x_axis=Axis(title=translate("durability.axis.elapsed", lang), kind=AxisKind.LINEAR,
                    tick_format=",.1f"),
        y_axis=Axis(title=translate("durability.axis.drift", lang), kind=AxisKind.LINEAR,
                    tick_format="+,.1f", suffix=" %"),
        traces=traces,
        height=420,
        caption=translate("durability.caption.drift", lang),
    )


def _anchored(model: AthleteDurabilityModel):
    """Drift since each run's first segments: observed, population, personalized."""
    segments = model.segments
    y, X, groups = design_matrix(segments, IDENTIFIABLE)
    theta_pop = np.array([model.population.get(n) for n in IDENTIFIABLE])
    theta_ind = np.array([model.coefficients.get(n) for n in IDENTIFIABLE])
    pop, ind = X @ theta_pop, X @ theta_ind
    hours = np.array([s.elapsed_s for s in segments]) / 3600.0
    out = [np.empty_like(y) for _ in range(3)]
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        rows = rows[np.argsort(hours[rows])]
        head = rows[:_REFERENCE_SEGMENTS]
        for target, values in zip(out, (y, pop, ind)):
            target[rows] = values[rows] - values[head].mean()
    return hours, out[0], out[1], out[2], groups


# --- 2. Projected durability ---------------------------------------------------

def _projection_chart(fitted, config: DurabilityConfig, lang: str):
    traces: List[Trace] = []
    for group, model in fitted:
        color = group_color(group.index)
        label = group.label if len(fitted) > 1 else ""
        u = _typical_intensity(model, config)
        longest = max((s.elapsed_s for s in model.segments), default=0.0) / 3600.0
        horizon = float(np.clip(np.ceil(longest + 1), _PROJECTION_MIN_H, _PROJECTION_MAX_H))
        hours = np.linspace(0.0, horizon, 97)

        def curve(coefficients: DurabilityCoefficients) -> np.ndarray:
            elapsed = hours * 3600.0
            n = len(elapsed) - 1
            exposures = accumulate_exposures(elapsed, np.full(n, u), np.zeros(n),
                                             np.zeros(n), config.exposure)
            profile = durability_profile(exposures, coefficients, config.exposure)
            return (profile.multiplier - 1) * 100

        traces.append(Trace(
            name=_name(label, translate("durability.series.population", lang)),
            x=hours.round(3).tolist(), y=curve(model.population).round(2).tolist(),
            kind=TraceKind.LINE, color=theme.BALANCED_RUNNER, dash="--", width=1.5,
            hover_template="%{x:.1f} h<br>+%{y:.1f} %<extra>%{fullData.name}</extra>",
        ))
        if model.confidence != POPULATION_ONLY:
            sd = model.posterior_sd.get(DURATION, 0.0)
            base = model.coefficients.get(DURATION)
            upper = curve(model.coefficients.with_values({DURATION: base + sd}))
            lower = curve(model.coefficients.with_values({DURATION: max(0.0, base - sd)}))
            traces.append(Trace(
                name=_name(label, translate("durability.series.personal", lang)),
                x=hours.round(3).tolist(), y=curve(model.coefficients).round(2).tolist(),
                kind=TraceKind.LINE, color=color, width=2.0,
                band_upper=upper.round(2).tolist(), band_lower=lower.round(2).tolist(),
                hover_template="%{x:.1f} h<br>+%{y:.1f} %<extra>%{fullData.name}</extra>",
            ))
    if not traces:
        return None
    u_text = ", ".join(
        f"{_typical_intensity(m, config) * 100:.0f} %" for _, m in fitted
    )
    return ChartData(
        title=translate("durability.chart.projection", lang),
        x_axis=Axis(title=translate("durability.axis.elapsed", lang), kind=AxisKind.LINEAR,
                    tick_format=",.0f"),
        y_axis=Axis(title=translate("durability.axis.extra_cost", lang), kind=AxisKind.LINEAR,
                    tick_format=",.1f", suffix=" %"),
        traces=traces,
        height=420,
        caption=translate("durability.caption.projection", lang).format(intensity=u_text),
    )


def _typical_intensity(model: AthleteDurabilityModel, config: DurabilityConfig) -> float:
    """Median intensity of the athlete's long-run segments, or race effort for 3 h."""
    if model.segments:
        return float(np.median([s.intensity for s in model.segments]))
    return sustainable_fraction(_FALLBACK_PROJECTION_S, config.capability)


# --- Table and notes ------------------------------------------------------------

def _coefficient_table(model: AthleteDurabilityModel, prefix: str, lang: str) -> TableData:
    rows = [{
        "component": translate(f"durability.component.{name}", lang),
        "population": model.population.get(name),
        "applied": model.coefficients.get(name),
        "sd": model.posterior_sd.get(name),
        "weight": model.personal_weight.get(name, 0.0) * 100,
    } for name in COMPONENTS]
    return TableData(
        title=prefix + translate("durability.table.title", lang),
        columns=[
            Column("component", translate("durability.col.component", lang)),
            Column("population", translate("durability.col.population", lang),
                   CellFormat("number", 4)),
            Column("applied", translate("durability.col.applied", lang), CellFormat("number", 4)),
            Column("sd", translate("durability.col.posterior_sd", lang), CellFormat("number", 4)),
            Column("weight", translate("durability.col.weight", lang), CellFormat("number", 0, "%")),
        ],
        rows=rows,
        download_name="durability_coefficients",
        caption=translate("durability.table.caption", lang),
    )


def _summary_notes(model: AthleteDurabilityModel, lang: str) -> List[str]:
    notes = durability_notes(model, lang)
    if model.reference.known:
        detail = model.reference.detail
        key = ("durability.note.reference_gap" if detail.get("gap_adjusted")
               else "durability.note.reference")
        notes.append(translate(key, lang).format(
            pace=fmt_pace(1000.0 / model.reference.speed_m_per_s),
            distance=detail.get("distance", "—"),
        ))
        outliers = detail.get("outliers") or []
        if outliers:
            notes.append(translate("durability.note.outliers", lang).format(
                count=len(outliers),
                distances=", ".join(sorted({o["distance"] for o in outliers})),
            ))
    if model.excluded:
        notes.append(translate("durability.note.excluded", lang).format(details=", ".join(
            f"{translate(f'durability.excluded.{k}', lang)}: {v}"
            for k, v in sorted(model.excluded.items())
        )))
    return notes


def _name(prefix: str, label: str) -> str:
    return f"{prefix} – {label}" if prefix else label


register(PlotDefinition(
    key="durability_curve",
    label_key="plot.durability_curve.label",
    description_key="plot.durability_curve.description",
    level=DataLevel.STREAM,
    compute=compute,
    params=PARAMS,
    requires_streams=True,
    cost=EXPENSIVE,
    category_key="plotcat.models",
))
