"""Progress of a background compute pass.

Deliberately the same shape as ``sync_state``, and for the same reason: the work
takes far longer than any sensible HTTP timeout, so the client starts it and polls.
Keeping the progress in the database rather than in memory means it survives a
container restart — on Railway that happens on every deploy, which is exactly when
a long job is most likely to be interrupted.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.infrastructure.postgres.pool import Database


@dataclass
class PrecomputeState:
    """Where a background compute pass got to."""

    status: str = "idle"  # idle | running | done | error
    done: int = 0
    total: int = 0
    message: str = ""
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self.status == "running"


class PostgresPrecomputeRepository:
    """Background-job state for one athlete."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def get(self, kind: str) -> PrecomputeState:
        row = self.db.fetch_one(
            "select status, done, total, message, finished_at, updated_at "
            "from precompute_jobs where athlete_id = %s and kind = %s",
            (self.athlete_id, kind),
        )
        if row is None:
            return PrecomputeState()
        return PrecomputeState(
            status=row["status"],
            done=row["done"],
            total=row["total"],
            message=row["message"] or "",
            finished_at=row["finished_at"],
            updated_at=row["updated_at"],
        )

    def set(self, kind: str, state: PrecomputeState) -> None:
        self.db.execute(
            """
            insert into precompute_jobs
                (athlete_id, kind, status, done, total, message, finished_at,
                 updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, now())
            on conflict (athlete_id, kind) do update set
                status = excluded.status,
                done = excluded.done,
                total = excluded.total,
                message = excluded.message,
                -- Kept when the new state doesn't carry one, so "last completed"
                -- survives a subsequent run that is still in progress.
                finished_at = coalesce(excluded.finished_at,
                                       precompute_jobs.finished_at),
                updated_at = now()
            """,
            (self.athlete_id, kind, state.status, state.done, state.total,
             state.message, state.finished_at),
        )
