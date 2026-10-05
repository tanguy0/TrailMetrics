"""Postgres store for an athlete's fitted models (GAP curves, durability).

One row per ``(athlete, kind)``: the model as last fitted, the runs it was fitted
on and when. Payloads are plain JSON; the domain object ↔ dict mapping lives with
the caller. Non-finite floats are stored as ``null`` (``jsonb`` has no NaN), so a
reader turns ``None`` back into ``nan`` where a number is expected.
"""

import json
import math
from datetime import date, datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.infrastructure.postgres.pool import Database


class PostgresAthleteModelRepository:
    """Fitted models for one athlete."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def get(self, kind: str, version: str) -> Optional[Dict[str, Any]]:
        """``{payload, activity_ids, computed_at}``, or ``None`` when absent or stale."""
        row = self.db.fetch_one(
            "select payload, activity_ids, computed_at from athlete_models "
            "where athlete_id = %s and kind = %s and version = %s",
            (self.athlete_id, kind, version),
        )
        if row is None:
            return None
        payload = row["payload"]
        return {
            "payload": json.loads(payload) if isinstance(payload, str) else payload,
            "activity_ids": tuple(int(i) for i in row["activity_ids"] or ()),
            "computed_at": row["computed_at"],
        }

    def put(self, kind: str, version: str, payload: Dict[str, Any],
            activity_ids: Sequence[int]) -> datetime:
        """Store (or replace) a fitted model; returns its ``computed_at``."""
        row = self.db.fetch_one(
            """
            insert into athlete_models (athlete_id, kind, version, payload, activity_ids)
            values (%s, %s, %s, %s, %s)
            on conflict (athlete_id, kind) do update set
                version = excluded.version,
                payload = excluded.payload,
                activity_ids = excluded.activity_ids,
                computed_at = now()
            returning computed_at
            """,
            (self.athlete_id, kind, version, json.dumps(plain(payload)),
             [int(i) for i in activity_ids]),
        )
        return row["computed_at"]


def plain(value: Any) -> Any:
    """``value`` as JSON-ready Python: numpy scalars unwrapped, NaN/inf → ``None``."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
