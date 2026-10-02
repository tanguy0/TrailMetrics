"""Postgres store for saved race plans ("Plan de course").

Plain dicts in and out, like the planned items: a saved plan is a title, a GPX and
a small JSON of parameters. The GPX is compressed here and nowhere else, so callers
only ever see the original bytes.
"""

import gzip
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.infrastructure.postgres.pool import Database

_SUMMARY = (
    "select id, title, gpx_name, params, distance_m, elevation_gain_m, "
    "created_at, updated_at from race_plans"
)


class PostgresRacePlanRepository:
    """Race plans for one account. Scoping happens here, not in the caller."""

    def __init__(self, db: Database, account_id: str):
        self.db = db
        self.account_id = account_id

    def list(self) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            f"{_SUMMARY} where account_id = %s order by updated_at desc",
            (self.account_id,),
        )
        return [_payload(row) for row in rows]

    def get(self, plan_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            f"{_SUMMARY} where account_id = %s and id = %s", (self.account_id, plan_id)
        )
        return _payload(row) if row else None

    def gpx(self, plan_id: str) -> Optional[bytes]:
        row = self.db.fetch_one(
            "select gpx_gz from race_plans where account_id = %s and id = %s",
            (self.account_id, plan_id),
        )
        return gzip.decompress(bytes(row["gpx_gz"])) if row else None

    def create(
        self, title: str, gpx_name: str, gpx: bytes, params: Dict[str, Any],
        distance_m: float, elevation_gain_m: float,
    ) -> Dict[str, Any]:
        plan_id = f"race_{uuid4().hex[:10]}"
        self.db.execute(
            "insert into race_plans (id, account_id, title, gpx_name, gpx_gz, params, "
            "distance_m, elevation_gain_m) values (%s, %s, %s, %s, %s, %s, %s, %s)",
            (plan_id, self.account_id, title, gpx_name, gzip.compress(gpx),
             json.dumps(params), distance_m, elevation_gain_m),
        )
        return self.get(plan_id)

    def update(
        self, plan_id: str, *, title: str, params: Dict[str, Any],
        distance_m: float, elevation_gain_m: float,
        gpx: Optional[bytes] = None, gpx_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Replace the inputs; the GPX only when a new one is given."""
        fields = ["title = %s", "params = %s", "distance_m = %s",
                  "elevation_gain_m = %s", "updated_at = now()"]
        values: List[Any] = [title, json.dumps(params), distance_m, elevation_gain_m]
        if gpx is not None:
            fields += ["gpx_gz = %s", "gpx_name = %s"]
            values += [gzip.compress(gpx), gpx_name or ""]
        self.db.execute(
            f"update race_plans set {', '.join(fields)} where account_id = %s and id = %s",
            (*values, self.account_id, plan_id),
        )
        return self.get(plan_id)

    def delete(self, plan_id: str) -> None:
        self.db.execute(
            "delete from race_plans where account_id = %s and id = %s",
            (self.account_id, plan_id),
        )


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    params = row["params"]
    return {
        "id": row["id"],
        "title": row["title"],
        "gpx_name": row["gpx_name"],
        "params": json.loads(params) if isinstance(params, str) else params,
        "distance_m": row["distance_m"],
        "elevation_gain_m": row["elevation_gain_m"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }
