"""Postgres connection pool and schema bootstrap.

Synchronous on purpose. The expensive work in this app is CPU-bound — pandas
aggregation, Savitzky–Golay filtering, XGBoost fits — which blocks an event loop
whether the database driver is async or not. Sync handlers running in FastAPI's
threadpool keep the model honest and the code free of async/sync bridging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """A pooled Postgres connection, with dict rows and helper queries."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8):
        # Opened lazily so importing the API never requires a reachable database.
        #
        # ``prepare_threshold=None`` disables server-side prepared statements.
        # psycopg3 would otherwise prepare any query after its fifth execution,
        # which breaks behind a *transaction* pooler (Supabase's port 6543): the
        # statement is prepared on one backend connection and the next
        # transaction may land on another, failing with "prepared statement does
        # not exist". The cost is a re-plan per execution on queries we run in a
        # loop; the alternative is an intermittent production error.
        self.pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=False,
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
        )
        self._opened = False

    def open(self) -> None:
        if not self._opened:
            self.pool.open(wait=True, timeout=30.0)
            self._opened = True

    def close(self) -> None:
        if self._opened:
            self.pool.close()
            self._opened = False

    def connection(self) -> Iterator[Connection]:
        self.open()
        return self.pool.connection()

    # --- Query helpers -----------------------------------------------------

    def fetch_all(self, sql: str, params: Sequence = ()) -> List[Dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def fetch_one(self, sql: str, params: Sequence = ()) -> Optional[Dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def execute(self, sql: str, params: Sequence = ()) -> int:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def execute_many(self, sql: str, rows: Sequence[Sequence]) -> int:
        if not rows:
            return 0
        with self.connection() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            return len(rows)

    def apply_schema(self) -> None:
        """Run ``schema.sql``. Idempotent — every statement is ``if not exists``."""
        sql = SCHEMA_PATH.read_text()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
        logger.info("schema applied")
