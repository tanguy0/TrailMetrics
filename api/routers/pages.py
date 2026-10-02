"""Analysis CRUD.

Analyses are the athlete's own, made to measure (design/tagg/access.md § Outils).
The analyses the product ships (GAP simulator, race comparator, long-term
progress, durability) are no longer seeded into every account: they are
**templates**, offered by "New analysis → from a template", and a page made from
one is an ordinary page — editable, deletable. The two that drew people in (the
GAP curve, durability) became designed pages in the Tools tab.

Athletes seeded before keep their defaults, ``builtin_key`` and all, which still
makes them undeletable — nothing they built on is taken away.
"""

from datetime import date, datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import (
    current_athlete,
    get_activity_repository,
    get_page_repository,
    language,
)
from api.serialization import page_payload, page_summary_payload
from src.domain.ports.storage import Athlete
from src.domain.spec.pages import PageSpec, PanelSpec
from src import dashboards

router = APIRouter(prefix="/pages", tags=["pages"])


class SavePageRequest(BaseModel):
    """A whole page document. Validated by the domain, not by a mirrored schema."""

    spec: Dict[str, Any]


class NewPageRequest(BaseModel):
    name: str = Field(default="New page", min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    icon: str = Field(default="📊", max_length=8)


class FromTemplateRequest(BaseModel):
    key: str = Field(min_length=1, max_length=60)


class DuplicateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)


@router.get("")
def list_pages(
    athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)
) -> dict:
    """Every analysis the athlete has — older accounts' defaults first."""
    pages = get_page_repository(athlete.id).list_pages()
    return {"pages": [page_summary_payload(page) for page in pages]}


@router.get("/templates")
def list_templates(
    athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)
) -> dict:
    """The analyses a new one can start from: name, description, icon."""
    oldest, newest = _athlete_range(athlete)
    templates = []
    for key in dashboards.BUILDERS:
        page = dashboards.build(key, oldest, newest, lang)
        templates.append({
            "key": key, "name": page.name, "description": page.description, "icon": page.icon,
        })
    return {"templates": templates}


@router.post("/from-template", status_code=status.HTTP_201_CREATED)
def create_from_template(
    payload: FromTemplateRequest,
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    """A new analysis built from a template over the athlete's current history.

    An ordinary page from then on: no ``builtin_key``, so it can be deleted, and
    making a second one from the same template is fine.
    """
    oldest, newest = _athlete_range(athlete)
    page = dashboards.build(payload.key, oldest, newest, lang)
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such template.")
    page.builtin_key = None
    get_page_repository(athlete.id).save(page)
    return page_payload(page)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_page(
    payload: NewPageRequest = NewPageRequest(),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """A new page with one empty panel, so it is immediately editable."""
    page = PageSpec(
        name=payload.name,
        description=payload.description,
        icon=payload.icon,
        panels=[PanelSpec(source=_default_source(athlete))],
    )
    get_page_repository(athlete.id).save(page)
    return page_payload(page)


@router.get("/{page_id}")
def get_page(page_id: str, athlete: Athlete = Depends(current_athlete)) -> dict:
    page = get_page_repository(athlete.id).get(page_id)
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such page.")
    return page_payload(page)


@router.put("/{page_id}")
def save_page(
    page_id: str,
    payload: SavePageRequest,
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Replace a page. The id in the path wins over anything in the body."""
    repository = get_page_repository(athlete.id)
    if repository.get(page_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such page.")

    existing = repository.get(page_id)
    page = _parse_spec(payload.spec)
    page.id = page_id
    # Whether this is a default analysis is decided by what is stored, never by the
    # client: a page cannot promote itself into one, and editing one cannot demote it
    # into something deletable.
    page.builtin_key = existing.builtin_key if existing else None
    repository.save(page)
    return page_payload(page)


@router.delete("/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_page(page_id: str, athlete: Athlete = Depends(current_athlete)) -> None:
    """Delete an analysis. The defaults every athlete gets are refused."""
    repository = get_page_repository(athlete.id)
    page = repository.get(page_id)
    if page is not None and page.is_default:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="This analysis ships with the app and cannot be deleted. "
                   "Duplicate it if you want a version you can remove.",
        )
    repository.delete(page_id)


@router.post("/{page_id}/duplicate", status_code=status.HTTP_201_CREATED)
def duplicate_page(
    page_id: str,
    payload: DuplicateRequest = DuplicateRequest(),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    repository = get_page_repository(athlete.id)
    source = repository.get(page_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such page.")
    clone = source.copy_as_custom(payload.name or f"{source.name} (copy)")
    repository.save(clone)
    return page_payload(clone)


# --- Helpers -------------------------------------------------------------

def _parse_spec(raw: Dict[str, Any]) -> PageSpec:
    try:
        return PageSpec.from_dict(raw)
    except Exception as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed page spec: {error}",
        )


def _athlete_range(athlete: Athlete) -> tuple:
    """The athlete's data range, or a sensible window when they have no data yet."""
    stored = get_activity_repository().date_range(athlete.id)
    if stored is None:
        today = date.today()
        return date(today.year, 1, 1), today
    oldest, newest = stored
    return _as_date(oldest), _as_date(newest)


def _as_date(value) -> date:
    return value.date() if isinstance(value, datetime) else value


def _default_source(athlete: Athlete):
    """A new panel covers the athlete's whole history — never empty on first open."""
    from src.domain.spec.datasource import DataSourceSpec, SourceMode, TimeWindow
    from src.translations import translate

    oldest, newest = _athlete_range(athlete)
    return DataSourceSpec(
        mode=SourceMode.WINDOW,
        windows=[TimeWindow(
            name=translate("dash.window.all_history", "en"),
            start=oldest, end=newest,
        )],
    )
