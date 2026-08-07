"""Postgres store for the training diary's planned workouts, goals, and notes.

One row per planned item, scoped to one athlete like every other table here.
Plain dicts in and out — a planned item is a handful of scalar columns, not a
document, so a domain dataclass here would be pure ceremony.
"""

from datetime import date as date_type
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.infrastructure.postgres.pool import Database

_SELECT = "select id, kind, date, end_date, title, body, importance from planned_items"


class PostgresPlannedItemRepository:
    """Planned items for one athlete. Scoping happens here, not in the caller."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def list_range(self, start: date_type, end: date_type) -> List[Dict[str, Any]]:
        """Every item whose span overlaps `[start, end]` — not just ones that
        start in it, so a multi-day note keeps showing up on the days it
        covers even once the calendar has scrolled past where it started."""
        rows = self.db.fetch_all(
            f"{_SELECT} where athlete_id = %s and date <= %s "
            f"and coalesce(end_date, date) >= %s order by date",
            (self.athlete_id, end, start),
        )
        return [_payload(row) for row in rows]

    def create(
        self, kind: str, date: date_type, title: str, body: str,
        importance: str = "primary", end_date: Optional[date_type] = None,
    ) -> Dict[str, Any]:
        item_id = f"plan_{uuid4().hex[:8]}"
        end_date = end_date if end_date is not None else date
        self.db.execute(
            "insert into planned_items (id, athlete_id, kind, date, end_date, "
            "title, body, importance) values (%s, %s, %s, %s, %s, %s, %s, %s)",
            (item_id, self.athlete_id, kind, date, end_date, title, body, importance),
        )
        return {
            "id": item_id, "kind": kind, "date": date.isoformat(),
            "end_date": end_date.isoformat(), "title": title, "body": body,
            "importance": importance,
        }

    def update(
        self,
        item_id: str,
        *,
        date: Optional[date_type] = None,
        end_date: Optional[date_type] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        importance: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Partial update. A date-only call is what a drag-and-drop move makes."""
        fields: List[str] = []
        params: List[Any] = []
        if date is not None:
            fields.append("date = %s")
            params.append(date)
        if end_date is not None:
            fields.append("end_date = %s")
            params.append(end_date)
        if title is not None:
            fields.append("title = %s")
            params.append(title)
        if body is not None:
            fields.append("body = %s")
            params.append(body)
        if importance is not None:
            fields.append("importance = %s")
            params.append(importance)
        if fields:
            fields.append("updated_at = now()")
            self.db.execute(
                f"update planned_items set {', '.join(fields)} "
                f"where athlete_id = %s and id = %s",
                (*params, self.athlete_id, item_id),
            )
        row = self.db.fetch_one(
            f"{_SELECT} where athlete_id = %s and id = %s",
            (self.athlete_id, item_id),
        )
        return _payload(row) if row else None

    def delete(self, item_id: str) -> None:
        self.db.execute(
            "delete from planned_items where athlete_id = %s and id = %s",
            (self.athlete_id, item_id),
        )


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    date_value = _iso(row["date"])
    return {
        "id": row["id"],
        "kind": row["kind"],
        "date": date_value,
        # NULL for a row from before `end_date` existed, or any single-day item —
        # both read the same as "just `date`".
        "end_date": _iso(row["end_date"]) if row["end_date"] is not None else date_value,
        "title": row["title"],
        "body": row["body"],
        "importance": row["importance"],
    }
