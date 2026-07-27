"""Port for storing the pages a user builds.

Deliberately tiny and storage-agnostic: the app talks to this, and the
implementation behind it can be a local JSON file today and a Postgres table
(one row per page, the spec as JSONB) once the app is multi-user — without any
change above this line.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.spec.pages import PageSpec


class PageRepository(ABC):
    """CRUD over :class:`PageSpec` documents for one owner."""

    @abstractmethod
    def list_pages(self) -> List[PageSpec]:
        """Every stored page, in a stable order (most recently saved last)."""

    @abstractmethod
    def get(self, page_id: str) -> Optional[PageSpec]:
        ...

    @abstractmethod
    def save(self, page: PageSpec) -> PageSpec:
        """Insert or update by ``page.id``; returns what was stored."""

    @abstractmethod
    def delete(self, page_id: str) -> None:
        ...
