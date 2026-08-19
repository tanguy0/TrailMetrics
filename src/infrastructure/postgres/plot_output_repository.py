"""Computed plot outputs, kept in Postgres.

The API already memoizes outputs per process (``api.deps.AthleteCaches``). That is
the right cache for editing a page — it is keyed identically and costs nothing —
but it dies with the worker, and on Railway a worker dies on every deploy. For the
cheap plots that is invisible. For a GAP curve, which is a model fit over
per-second data, it means the athlete waits again for a number that has not
changed.

So outputs are also written here, keyed by the render signature. Two properties
make that safe:

* **The key contains the resolved activity ids.** A new import changes the key, so
  a stale entry is never read rather than needing to be invalidated for *that*
  reason. It doesn't cover a change to how a plot is drawn from the same
  activities, though — a recolored trace, a new default line width — which is
  what ``render_page.RENDER_VERSION`` is for: bump it and every cached row
  misses once, everywhere, on the next render after deploy.
* **The payload is the IR**, not a figure, so it is rebuilt with
  :meth:`~src.domain.charts.ir.PlotOutput.from_dict` and rendered by the same code
  as a fresh computation. A cached page and a computed page cannot diverge.

Signatures are hashed rather than stored verbatim: the raw one embeds every
activity id in the panel, which for a decade of running is tens of kilobytes of
primary key.
"""

import hashlib
import json
import logging
from typing import Optional

from src.domain.charts.ir import PlotOutput
from src.infrastructure.postgres.pool import Database

logger = logging.getLogger(__name__)


def signature_key(signature: str) -> str:
    """Fixed-length key for a render signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


class PostgresPlotOutputRepository:
    """Stored plot outputs for one athlete."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def get(self, signature: str) -> Optional[PlotOutput]:
        row = self.db.fetch_one(
            "select payload from plot_outputs "
            "where athlete_id = %s and signature = %s",
            (self.athlete_id, signature_key(signature)),
        )
        if row is None:
            return None
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return PlotOutput.from_dict(payload)
        except Exception as error:
            # A payload written by an incompatible version is a cache miss, not an
            # error: recomputing is always correct.
            logger.warning("unreadable cached output: %s", error)
            return None

    def put(self, signature: str, plot_type: str, output: PlotOutput) -> None:
        self.db.execute(
            """
            insert into plot_outputs (athlete_id, signature, plot_type, payload)
            values (%s, %s, %s, %s)
            on conflict (athlete_id, signature) do update set
                payload = excluded.payload,
                plot_type = excluded.plot_type,
                created_at = now()
            """,
            (self.athlete_id, signature_key(signature), plot_type,
             json.dumps(output.to_dict())),
        )

    def clear(self) -> int:
        """Drop every stored output for this athlete — what "recompute" does."""
        return self.db.execute(
            "delete from plot_outputs where athlete_id = %s", (self.athlete_id,)
        )

    def count(self) -> int:
        row = self.db.fetch_one(
            "select count(*) as n from plot_outputs where athlete_id = %s",
            (self.athlete_id,),
        )
        return int(row["n"]) if row else 0
