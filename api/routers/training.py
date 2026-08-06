"""The training diary: planned workouts/goals and completed sessions, by date.

Coach and athlete will eventually see the same calendar with different write
permissions; that layer is not built yet, so today every row is scoped to the
signed-in athlete like the rest of the app. The calendar itself lives on the web
app as a fixed screen, not a page-builder page — see ``api/routers/home.py`` for
the same "fixed screen, borrowed rendering pipeline" pattern.
"""

from datetime import date as date_type
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.deps import (
    current_athlete,
    get_activity_repository,
    get_planned_item_repository,
)
from api.routers.home import _activity_card, _row_date
from src.domain.ports.storage import Athlete

router = APIRouter(prefix="/training", tags=["training"])

_KINDS = {"workout", "goal"}
_IMPORTANCES = {"primary", "secondary"}


class CreatePlannedItemRequest(BaseModel):
    kind: str
    date: date_type
    title: str = Field(default="", max_length=200)
    body: str = Field(default="", max_length=10000)
    importance: str = "primary"


class UpdatePlannedItemRequest(BaseModel):
    # The type is imported under an alias (`date_type`) rather than `date`: with a
    # field named `date` and a plain `= None` default, Pydantic's annotation
    # resolution uses the class's own namespace as part of its lookup, and that
    # namespace's `date` attribute is by then the default value, not the type —
    # so `Optional[date]` silently resolves to `Optional[None]` and every update
    # is rejected. Naming the type differently from the field sidesteps it.
    date: Optional[date_type] = None
    title: Optional[str] = Field(default=None, max_length=200)
    body: Optional[str] = Field(default=None, max_length=10000)
    importance: Optional[str] = None


def _validate_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"kind must be one of {sorted(_KINDS)}",
        )


def _validate_importance(importance: Optional[str]) -> None:
    if importance is not None and importance not in _IMPORTANCES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"importance must be one of {sorted(_IMPORTANCES)}",
        )


@router.get("/calendar")
def calendar(
    start: date_type = Query(...),
    end: date_type = Query(...),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Everything the calendar draws between ``start`` and ``end``, inclusive."""
    planned = get_planned_item_repository(athlete.id).list_range(start, end)

    rows = get_activity_repository().rows(athlete.id)
    activities: List[Dict[str, Any]] = [
        _activity_card(row) for row in rows
        if (row_date := _row_date(row)) is not None and start <= row_date.date() <= end
    ]

    return {"planned_items": planned, "activities": activities}


@router.post("/planned-items", status_code=status.HTTP_201_CREATED)
def create_planned_item(
    payload: CreatePlannedItemRequest,
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    _validate_kind(payload.kind)
    _validate_importance(payload.importance)
    return get_planned_item_repository(athlete.id).create(
        payload.kind, payload.date, payload.title, payload.body, payload.importance,
    )


@router.patch("/planned-items/{item_id}")
def update_planned_item(
    item_id: str,
    payload: UpdatePlannedItemRequest,
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    _validate_importance(payload.importance)
    updated = get_planned_item_repository(athlete.id).update(
        item_id, date=payload.date, title=payload.title, body=payload.body,
        importance=payload.importance,
    )
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="planned item not found")
    return updated


@router.delete("/planned-items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_planned_item(
    item_id: str,
    athlete: Athlete = Depends(current_athlete),
) -> None:
    get_planned_item_repository(athlete.id).delete(item_id)
