"""Postgres store for level assessments, one account at a time."""

import json
from typing import Any, Dict, Optional

from src.infrastructure.postgres.pool import Database


class PostgresLevelRepository:
    def __init__(self, db: Database, account_id: str):
        self.db = db
        self.account_id = account_id

    def save(self, method: str, inputs: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        row = self.db.fetch_one(
            "insert into level_estimates (account_id, method, inputs, result) "
            "values (%s, %s, %s, %s) returning id, method, inputs, result, created_at",
            (self.account_id, method, json.dumps(inputs), json.dumps(result)),
        )
        return _payload(row)

    def latest(self) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            "select id, method, inputs, result, created_at from level_estimates "
            "where account_id = %s order by created_at desc limit 1",
            (self.account_id,),
        )
        return _payload(row) if row else None


def _payload(row) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "method": row["method"],
        "inputs": row["inputs"],
        "result": row["result"],
        "created_at": row["created_at"].isoformat(),
    }
