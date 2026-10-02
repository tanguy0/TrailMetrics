"""Rendering: a page or panel spec in, chart IR out.

The spec comes in the request body rather than by id, which is the detail that makes
a live builder possible — the client can render edits that have not been saved, and
:post:`/render/panel` lets it re-render just the panel being worked on instead of
the whole page.

Nothing here draws anything. The response is the same JSON-serializable chart IR the
domain produced, so the browser owns presentation and the server owns computation.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import current_athlete, get_planned_item_repository, language, render_context_for
from api.serialization import panel_payload
from src.domain.charts.markers import Race, outputs_span, pin_markers
from src.domain.ports.storage import Athlete
from src.domain.spec.pages import PageSpec, PanelSpec
from src.translations import translate
from src.usecases.render_page import PanelResult, RenderPage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["render"])

_renderer = RenderPage()


class RenderPageRequest(BaseModel):
    spec: Dict[str, Any]
    # Plot ids the user explicitly asked to compute. Expensive plots (model fits)
    # stay pending until they appear here, so opening a page is always fast.
    force_plot_ids: List[str] = Field(default_factory=list)
    # Compute everything up front, ignoring the cost hint.
    compute_all: bool = False
    # Recompute and overwrite the cache — the "recompute" button.
    refresh: bool = False


class RenderPanelRequest(BaseModel):
    panel: Dict[str, Any]
    force_plot_ids: List[str] = Field(default_factory=list)
    compute_all: bool = False
    refresh: bool = False


@router.post("/render")
def render_page(
    payload: RenderPageRequest,
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    page = _parse(PageSpec, payload.spec, "page")
    context = render_context_for(
        athlete, lang,
        defer_expensive=not payload.compute_all,
        force_plot_ids=set(payload.force_plot_ids),
        refresh=payload.refresh,
    )
    results = _renderer.execute(page, context)
    _pin_markers(results, athlete, lang)
    return {
        "page_id": page.id,
        "panels": [panel_payload(result) for result in results],
    }


@router.post("/render/panel")
def render_panel(
    payload: RenderPanelRequest,
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    """Render one panel — what the editor calls on every parameter change."""
    panel = _parse(PanelSpec, payload.panel, "panel")
    context = render_context_for(
        athlete, lang,
        defer_expensive=not payload.compute_all,
        force_plot_ids=set(payload.force_plot_ids),
        refresh=payload.refresh,
    )
    result = _renderer.render_panel(panel, context)
    _pin_markers([result], athlete, lang)
    return {"panel": panel_payload(result)}


def _pin_markers(results: List[PanelResult], athlete: Athlete, lang: str) -> None:
    """Mark today and the athlete's races on every time chart that covers them.

    At serve time, after the cache (see :mod:`src.domain.charts.markers`): one
    calendar query for the whole response. A calendar that cannot be read costs
    the race dots, never the charts.
    """
    plots = [plot for result in results for plot in result.plots]
    span = outputs_span(plot.output for plot in plots)
    if span is None:
        return
    try:
        items = get_planned_item_repository(athlete.id).list_range(*span)
    except Exception:
        logger.warning("could not read the calendar for chart markers", exc_info=True)
        items = []
    races = [
        Race(day=date.fromisoformat(str(item["date"])[:10]), name=item["title"])
        for item in items if item["kind"] == "goal"
    ]
    today_label = translate("ui.chart.today", lang)
    for plot in plots:
        plot.output = pin_markers(plot.output, races, date.today(), today_label)


def _parse(model, raw: Dict[str, Any], what: str):
    """Build a spec from untrusted JSON, turning any malformed input into a 422."""
    if not isinstance(raw, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected a {what} object.",
        )
    try:
        return model.from_dict(raw)
    except Exception as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed {what} spec: {error}",
        )
