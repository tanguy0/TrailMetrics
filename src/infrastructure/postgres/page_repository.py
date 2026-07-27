"""Postgres :class:`PageRepository` — a page is a JSONB document.

``name``/``description``/``icon`` are denormalized out of the spec so listing an
athlete's pages is one cheap indexed query that never parses JSON.
"""

import json
from typing import List, Optional

from src.domain.ports.page_repository import PageRepository
from src.domain.spec.pages import PageSpec
from src.infrastructure.postgres.pool import Database


class PostgresPageRepository(PageRepository):
    """Pages for one athlete. Scoping happens here, not in the caller."""

    def __init__(self, db: Database, athlete_id: int):
        self.db = db
        self.athlete_id = athlete_id

    def list_pages(self) -> List[PageSpec]:
        rows = self.db.fetch_all(
            "select spec from pages where athlete_id = %s order by updated_at desc",
            (self.athlete_id,),
        )
        return [_spec(row) for row in rows if _spec(row) is not None]

    def get(self, page_id: str) -> Optional[PageSpec]:
        row = self.db.fetch_one(
            "select spec from pages where athlete_id = %s and id = %s",
            (self.athlete_id, page_id),
        )
        return _spec(row) if row else None

    def save(self, page: PageSpec) -> PageSpec:
        self.db.execute(
            """
            insert into pages
                (id, athlete_id, name, description, icon, spec, schema_version)
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (id) do update set
                name = excluded.name,
                description = excluded.description,
                icon = excluded.icon,
                spec = excluded.spec,
                schema_version = excluded.schema_version,
                updated_at = now()
            where pages.athlete_id = excluded.athlete_id
            """,
            (page.id, self.athlete_id, page.name, page.description, page.icon,
             json.dumps(page.to_dict()), page.schema_version),
        )
        return page

    def delete(self, page_id: str) -> None:
        self.db.execute(
            "delete from pages where athlete_id = %s and id = %s",
            (self.athlete_id, page_id),
        )


def _spec(row) -> Optional[PageSpec]:
    raw = row.get("spec")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return None
    try:
        return PageSpec.from_dict(raw)
    except Exception:
        # One unreadable document must not break the page list.
        return None
