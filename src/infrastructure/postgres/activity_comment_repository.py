"""Postgres store for free-text comments an athlete attaches to one activity.

One row per comment, scoped to one athlete like every other table here. Plain
dicts in and out — a comment is a handful of scalar columns, not a document.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.infrastructure.postgres.pool import Database

_COLUMNS = "id, activity_id, body, created_at, updated_at"


class PostgresActivityCommentRepository:
    """Comments for one athlete's activities. Scoping happens here, not in the caller."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def list_for_activity(self, activity_id: int) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            f"select {_COLUMNS} from activity_comments "
            f"where athlete_id = %s and activity_id = %s order by created_at",
            (self.athlete_id, activity_id),
        )
        return [_payload(row) for row in rows]

    def create(self, activity_id: int, body: str) -> Dict[str, Any]:
        comment_id = f"cmt_{uuid4().hex[:8]}"
        row = self.db.fetch_one(
            "insert into activity_comments (id, athlete_id, activity_id, body) "
            f"values (%s, %s, %s, %s) returning {_COLUMNS}",
            (comment_id, self.athlete_id, activity_id, body),
        )
        return _payload(row)

    def update(self, comment_id: str, body: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            "update activity_comments set body = %s, updated_at = now() "
            f"where athlete_id = %s and id = %s returning {_COLUMNS}",
            (body, self.athlete_id, comment_id),
        )
        return _payload(row) if row else None

    def delete(self, comment_id: str) -> None:
        self.db.execute(
            "delete from activity_comments where athlete_id = %s and id = %s",
            (self.athlete_id, comment_id),
        )


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "activity_id": row["activity_id"],
        "body": row["body"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }
