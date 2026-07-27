"""Page CRUD, plus the built-in example pages.

Built-ins are generated per request rather than stored, because their default time
windows depend on the athlete's own date range — a page whose window predates every
activity would open empty. They are read-only; ``POST /pages/builtin/{key}/duplicate``
hands the athlete an editable copy, which is what makes them useful as examples
rather than decoration.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import (
    current_athlete,
    get_activity_repository,
    get_page_repository,
    language,
)
from api.serialization import page_payload, page_summary_payload
from src import dashboards
from src.domain.ports.storage import Athlete
from src.domain.spec.pages import PageSpec, PanelSpec

router = APIRouter(prefix="/pages", tags=["pages"])


class SavePageRequest(BaseModel):
    """A whole page document. Validated by the domain, not by a mirrored schema."""

    spec: Dict[str, Any]


class NewPageRequest(BaseModel):
    name: str = Field(default="New page", min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    icon: str = Field(default="📊", max_length=8)


class DuplicateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)


# --- Built-ins -------------------------------------------------------------

@router.get("/builtin")
def list_builtin(
    athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)
) -> dict:
    pages = _builtin_pages(athlete, lang)
    return {"pages": [page_summary_payload(page) for page in pages]}


@router.get("/builtin/{key}")
def get_builtin(
    key: str,
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    page = _builtin(key, athlete, lang)
    return page_payload(page)


@router.post("/builtin/{key}/duplicate", status_code=status.HTTP_201_CREATED)
def duplicate_builtin(
    key: str,
    payload: DuplicateRequest = DuplicateRequest(),
    athlete: Athlete = Depends(current_athlete),
    lang: str = Depends(language),
) -> dict:
    """Copy an example into the athlete's own pages, fully editable."""
    source = _builtin(key, athlete, lang)
    clone = source.copy_as_custom(payload.name or source.name)
    get_page_repository(athlete.id).save(clone)
    return page_payload(clone)


# --- User pages ----------------------------------------------------------

@router.get("")
def list_pages(athlete: Athlete = Depends(current_athlete)) -> dict:
    pages = get_page_repository(athlete.id).list_pages()
    return {"pages": [page_summary_payload(page) for page in pages]}


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

    page = _parse_spec(payload.spec)
    page.id = page_id
    # A stored page is never a built-in, whatever the client claims.
    page.builtin_key = None
    repository.save(page)
    return page_payload(page)


@router.delete("/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_page(page_id: str, athlete: Athlete = Depends(current_athlete)) -> None:
    get_page_repository(athlete.id).delete(page_id)


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


def _builtin_pages(athlete: Athlete, lang: str) -> List[PageSpec]:
    oldest, newest = _athlete_range(athlete)
    return dashboards.build_all(oldest, newest, lang)


def _builtin(key: str, athlete: Athlete, lang: str) -> PageSpec:
    oldest, newest = _athlete_range(athlete)
    page = dashboards.build(key, oldest, newest, lang)
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such example page.")
    return page


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
