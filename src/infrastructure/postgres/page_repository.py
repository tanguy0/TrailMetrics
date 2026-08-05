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
        # Defaults first, then the athlete's own newest-first. The three analyses
        # everyone has are the stable landmarks of the screen, so they should not move
        # around as other pages are edited.
        rows = self.db.fetch_all(
            "select spec from pages where athlete_id = %s "
            "order by (builtin_key is null), builtin_key, updated_at desc",
            (self.athlete_id,),
        )
        return [spec for spec in (_spec(row) for row in rows) if spec is not None]

    def default_keys(self) -> set:
        """Which default analyses this athlete already has — what seeding checks."""
        rows = self.db.fetch_all(
            "select builtin_key from pages "
            "where athlete_id = %s and builtin_key is not null",
            (self.athlete_id,),
        )
        return {row["builtin_key"] for row in rows}

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
                (id, athlete_id, name, description, icon, spec, schema_version,
                 builtin_key)
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (id) do update set
                name = excluded.name,
                description = excluded.description,
                icon = excluded.icon,
                spec = excluded.spec,
                schema_version = excluded.schema_version,
                -- Never cleared on update: a default analysis stays one however it is
                -- edited, and losing the key would make it deletable.
                builtin_key = coalesce(excluded.builtin_key, pages.builtin_key),
                updated_at = now()
            where pages.athlete_id = excluded.athlete_id
            """,
            (page.id, self.athlete_id, page.name, page.description, page.icon,
             json.dumps(page.to_dict()), page.schema_version, page.builtin_key),
        )
        return page

    def delete(self, page_id: str) -> None:
        """Delete one page. Default analyses are refused, in SQL.

        The API checks first so it can explain itself, but the rule is enforced here
        too: "cannot be deleted" is a property of the row, not of one code path.
        """
        self.db.execute(
            "delete from pages where athlete_id = %s and id = %s "
            "and builtin_key is null",
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
