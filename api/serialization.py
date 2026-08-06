"""Domain objects → JSON payloads.

The important one is :func:`registry_payload`. It ships the *whole* plot catalogue —
every plot type, its parameter schema, the metric vocabulary and the dynamic choice
lists — so the web app builds its forms from data rather than from hand-written
components. Registering a plot type or a metric in Python makes it appear in the
UI with no frontend change, which is the property the whole design is chasing.

Field names stay ``snake_case`` all the way to the browser. Consistency with the
Python side is worth more here than JavaScript convention: it removes an entire
class of mapping bug.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from src.domain.dataset.binning import GRANULARITIES
from src.domain.dataset.metrics import ACTIVITY_METRICS, AGGREGATIONS, NO_METRIC
from src.domain.plots import all_plots
from src.domain.progress.models import GRADIENT_BAND_KEYS, PR_DISTANCES
from src.domain.ports.activity_data import ActivitySummary
from src.domain.ports.storage import Athlete, SyncState
from src.domain.spec.pages import PageSpec
from src.translations import translate
from src.usecases.render_page import PanelResult, PlotResult


# --- Registry --------------------------------------------------------------

def registry_payload(lang: str) -> Dict[str, Any]:
    """Everything the client needs to render plot pickers and parameter forms."""
    return {
        "plots": [_plot_definition(definition, lang) for definition in all_plots()],
        "metrics": {key: _metric(metric, lang)
                    for key, metric in ACTIVITY_METRICS.items()},
        "providers": _providers(lang),
        "source_modes": [
            {"value": "activities", "label": translate("param.rows.activity", lang)},
            {"value": "window", "label": translate("dash.window.all_history", lang)},
            {"value": "windows", "label": translate("param.hr_bands", lang)},
        ],
    }


def _plot_definition(definition, lang: str) -> Dict[str, Any]:
    return {
        "key": definition.key,
        "label": definition.label(lang),
        "description": definition.description(lang),
        "category": translate(definition.category_key, lang),
        "level": definition.level.value,
        "series_level": definition.series_level,
        "requires_streams": definition.requires_streams,
        "requires_weight": definition.requires_weight,
        # False for content blocks (prose, an image): the client hides the data-source
        # affordances that mean nothing for them.
        "requires_data": definition.requires_data,
        "cost": definition.cost,
        "params": [param.to_dict(lang) for param in definition.params],
    }


def _metric(metric, lang: str) -> Dict[str, Any]:
    return {
        "key": metric.key,
        "label": translate(metric.label_key, lang),
        "unit": metric.unit,
        "value_kind": metric.value_kind,
        "decimals": metric.decimals,
        "default_agg": metric.default_agg,
        # Empty means the metric fixes its own aggregation (ratios, counts), which
        # is how the client knows to hide the control.
        "allowed_aggs": list(metric.allowed_aggs),
        "higher_is_better": metric.higher_is_better,
        "needs_streams": metric.needs_streams,
        "needs_weight": metric.needs_weight,
    }


def _providers(lang: str) -> Dict[str, List[Dict[str, str]]]:
    """Resolved option lists for every ``choices_from`` a parameter can name."""
    metrics = [
        {"value": key, "label": translate(metric.label_key, lang)}
        for key, metric in ACTIVITY_METRICS.items()
    ]
    return {
        "activity_metrics": metrics,
        # For the "plot a second metric too" controls: same list, plus an explicit
        # opt-out, so a single-metric chart stays the default.
        "activity_metrics_optional": [
            {"value": NO_METRIC, "label": translate("param.metric2.none", lang)},
            *metrics,
        ],
        "aggregations": [
            {"value": agg, "label": translate(f"agg.{agg}", lang)}
            for agg in AGGREGATIONS
        ],
        "granularities": [
            {"value": gran, "label": translate(f"gran.{gran}", lang)}
            for gran in GRANULARITIES
        ],
        "pr_distances": [{"value": label, "label": label} for label, _ in PR_DISTANCES],
        "gradient_bands": [
            {"value": key, "label": translate(f"ltp.band.{key}", lang)}
            for key in GRADIENT_BAND_KEYS
        ],
        # date sorts by the activity's own timestamp; the rest are metric columns.
        "sortable_columns": [
            {"value": "date", "label": translate("races.col.date", lang)},
            *metrics,
        ],
    }


# --- Render results --------------------------------------------------------

def panel_payload(result: PanelResult) -> Dict[str, Any]:
    """One panel's outcome: how the source resolved, plus each plot's chart IR."""
    return {
        "panel_id": result.spec.id,
        "title": result.spec.title,
        "description": result.spec.description,
        "columns": result.spec.columns,
        "error": result.error,
        "groups": [
            {"label": group.label, "index": group.index, "size": group.size}
            for group in (result.resolved.groups if result.resolved else [])
        ],
        "activity_count": len(result.resolved.activity_ids) if result.resolved else 0,
        "plots": [_plot_payload(plot) for plot in result.plots],
    }


def _plot_payload(result: PlotResult) -> Dict[str, Any]:
    return {
        "plot_id": result.spec.id,
        "plot_type": result.spec.plot_type,
        "title": result.spec.title,
        # Coerced parameters, so the client's form shows what actually ran rather
        # than whatever partial values were stored.
        "params": result.params,
        "error": result.error,
        "pending": result.pending,
        "cost": result.definition.cost if result.definition else "cheap",
        "output": result.output.to_dict(),
    }


# --- Activities & athlete --------------------------------------------------

def summary_payload(summary: ActivitySummary) -> Dict[str, Any]:
    return {
        "activity_id": summary.activity_id,
        "start_date": summary.start_date.isoformat(),
        "sport_type": summary.sport_type,
        "has_streams": summary.has_streams,
        "distance_m": summary.distance_m,
        "moving_s": summary.moving_s,
        "label": summary.label,
    }


def athlete_payload(
    athlete: Athlete,
    sync: SyncState,
    date_range: Optional[tuple],
    activity_count: int,
    sport_types: List[str],
) -> Dict[str, Any]:
    oldest, newest = date_range if date_range else (None, None)
    return {
        "id": athlete.id,
        "firstname": athlete.firstname,
        "lastname": athlete.lastname,
        "display_name": athlete.display_name,
        "profile_url": athlete.profile_url,
        "weight_kg": athlete.weight_kg,
        "birthdate": athlete.birthdate.isoformat() if athlete.birthdate else None,
        "height_cm": athlete.height_cm,
        "email": athlete.email,
        "hr_zone1_end": athlete.hr_zone1_end,
        "hr_zone2_end": athlete.hr_zone2_end,
        "hr_zone3_end": athlete.hr_zone3_end,
        "hr_zone4_end": athlete.hr_zone4_end,
        "hr_max": athlete.hr_max,
        "vma_pace_s_per_km": athlete.vma_pace_s_per_km,
        # Derived rather than left to the client to infer from a null email, so
        # "have they answered?" has one definition.
        "needs_email": athlete.needs_email,
        # Derived here rather than in the browser so every client agrees on it.
        "age": athlete.age_on(date.today()),
        "activity_count": activity_count,
        "sport_types": sport_types,
        "oldest_activity": oldest.isoformat() if oldest else None,
        "newest_activity": newest.isoformat() if newest else None,
        "sync": {
            "status": sync.status,
            "done": sync.done,
            "total": sync.total,
            "message": sync.message,
            "last_synced_at": sync.last_synced_at.isoformat()
            if sync.last_synced_at else None,
        },
    }


def page_payload(page: PageSpec) -> Dict[str, Any]:
    return page.to_dict()


def page_summary_payload(page: PageSpec) -> Dict[str, Any]:
    """Enough to list analyses without shipping every panel."""
    return {
        "id": page.id,
        "name": page.name,
        "description": page.description,
        "icon": page.icon,
        "builtin_key": page.builtin_key,
        # One of the analyses every athlete gets: editable like any other, but the
        # client hides Delete for it.
        "is_default": page.is_default,
        "panel_count": len(page.panels),
        "plot_count": sum(len(panel.plots) for panel in page.panels),
    }
