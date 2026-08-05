"""Compute the expensive plots ahead of the reader.

One plot type in this app fits models rather than aggregating rows, and it is the
centrepiece of an analysis every athlete gets. Left alone, the first athlete to open
that page waits for a per-year GAP fit over their whole per-second history — and waits
again after every deploy, because the only cache was in the worker's memory.

This router closes both gaps. The web app starts a pass as soon as the athlete
connects; it renders their stored analyses exactly as a browser would, which fills the
persistent output cache (``plot_outputs``), so opening the page later is a database
read. Progress goes through ``precompute_jobs`` and is polled, for the same reason
the Strava import is: the work outlives any sensible HTTP timeout.

Nothing here is GAP-specific, and nothing here knows about default analyses. It walks
the athlete's *stored* pages and skips any panel with no expensive plot in it — so a
page of trends costs nothing, and a new model-fitting plot type is covered the day it
is registered.
"""

import logging
from datetime import datetime, timezone
from typing import List, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from api.deps import (
    current_athlete,
    get_page_repository,
    get_plot_output_repository,
    get_precompute_repository,
    language,
    render_context_for,
)
from src.domain.plots import all_plots
from src.domain.plots.base import EXPENSIVE
from src.domain.ports.storage import Athlete
from src.domain.spec.pages import PageSpec, PanelSpec
from src.infrastructure.postgres.precompute_repository import PrecomputeState
from src.usecases.render_page import RenderPage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/precompute", tags=["precompute"])

# The job kind: fill the output cache for the athlete's stored analyses. A named
# kind rather than one row per athlete, so a second kind of background pass can be
# added later without a migration.
JOB_KIND = "analyses"

# A pass that reports "running" but has not touched its row for this long is taken
# to be dead — the worker was killed mid-fit. Without this a single interrupted pass
# would block every later one forever, and the athlete's page would stay pending
# with no way back other than a manual database edit.
_STALE_AFTER_S = 30 * 60

_renderer = RenderPage()


class PrecomputeRequest(BaseModel):
    # Recompute even what is already cached. Same meaning as `refresh` on /render.
    force: bool = False


@router.get("")
def precompute_status(athlete: Athlete = Depends(current_athlete)) -> dict:
    """Where the background pass got to. Polled while it runs."""
    return _payload(get_precompute_repository(athlete.id).get(JOB_KIND))


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def start_precompute(
    background: BackgroundTasks,
    payload: PrecomputeRequest = PrecomputeRequest(),
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    """Start (or restart) the pass. Returns immediately; poll ``GET /precompute``.

    Idempotent in the way that matters: a pass over an already-filled cache reads it
    and finishes in milliseconds, so the web app can fire this on every connect
    without asking whether it is needed.
    """
    jobs = get_precompute_repository(athlete.id)
    state = jobs.get(JOB_KIND)
    if state.is_running and not _is_stale(state):
        # Two passes would fit the same models twice over the same streams.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="A precompute pass is already running for this athlete.",
        )

    jobs.set(JOB_KIND, PrecomputeState(status="running", message="starting"))
    background.add_task(_run_precompute, athlete, lang, payload.force)
    return {"status": "running"}


def _run_precompute(athlete: Athlete, lang: str, force: bool) -> None:
    """Render each panel worth warming, reporting progress as it goes."""
    jobs = get_precompute_repository(athlete.id)
    try:
        if force:
            # Only on an explicit recompute. Otherwise the whole point is to *reuse*
            # what is stored, and a pass that cleared first would rebuild a decade
            # of fits on every connect.
            get_plot_output_repository(athlete.id).clear()

        panels = _panels_to_warm(athlete)
        total = len(panels)
        jobs.set(JOB_KIND, PrecomputeState(
            status="running", done=0, total=total, message="computing",
        ))

        # One context for the whole pass, so the per-second series a panel loads are
        # reused by the next one instead of being fetched from storage again.
        context = render_context_for(
            athlete, lang, defer_expensive=False, refresh=force,
        )

        failures = 0
        for index, (page, panel) in enumerate(panels, start=1):
            # Announced *before* the work, not after. Progress is counted in panels and
            # the first one is by far the slowest — a whole history of per-second data
            # to fetch and fit — so for several minutes the only honest thing the bar
            # can say is which panel it is on. Reporting after each panel instead left
            # it reading "computing" for the entire time that mattered.
            jobs.set(JOB_KIND, PrecomputeState(
                status="running", done=index - 1, total=total,
                message=f"{page.name} — {panel.title}" if panel.title else page.name,
            ))
            try:
                _renderer.render_panel(panel, context)
            except Exception:
                # One panel that cannot be computed must not cost the others their
                # cached result — and it will surface its own error when read.
                logger.exception(
                    "precompute failed for panel %s of %s", panel.id, page.name
                )
                failures += 1

        jobs.set(JOB_KIND, PrecomputeState(
            status="done", done=total, total=total,
            message=f"{total - failures}/{total} panels ready",
            finished_at=datetime.now(timezone.utc),
        ))
    except Exception as error:
        logger.exception("precompute failed for athlete %s", athlete.id)
        jobs.set(JOB_KIND, PrecomputeState(
            status="error", message=str(error)[:400],
            finished_at=datetime.now(timezone.utc),
        ))


# --- Helpers ---------------------------------------------------------------

def _panels_to_warm(athlete: Athlete) -> List[Tuple[PageSpec, PanelSpec]]:
    """Every stored panel that contains a plot worth precomputing.

    Reading the *stored* pages is what makes the cache actually get used: the render
    signature covers the panel's source, so warming a page built from anything other
    than what the browser will send would file the result under a key nothing reads.

    Panels with no expensive plot are skipped rather than rendered. They are already
    fast, and including them would make the progress count mostly noise — 7 panels of
    which 2 matter reads as stalled while the 2 do all the work.
    """
    expensive = {
        definition.key for definition in all_plots() if definition.cost == EXPENSIVE
    }
    pages = get_page_repository(athlete.id).list_pages()
    return [
        (page, panel)
        for page in pages
        for panel in page.panels
        if any(plot.plot_type in expensive for plot in panel.plots)
    ]


def _is_stale(state: PrecomputeState) -> bool:
    if state.updated_at is None:
        return True
    updated = state.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated).total_seconds() > _STALE_AFTER_S


def _payload(state: PrecomputeState) -> dict:
    return {
        "status": state.status,
        "done": state.done,
        "total": state.total,
        "message": state.message,
        "finished_at": state.finished_at.isoformat() if state.finished_at else None,
    }
