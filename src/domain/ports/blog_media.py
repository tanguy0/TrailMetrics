"""Port for the blog's PDF and rasterized-page objects.

Separate from :class:`~src.domain.ports.storage.StreamStore` despite the similar
shape: streams are private, one bucket per athlete's own data, read back through
the API. Blog media is public — anyone reads it, unauthenticated, ideally straight
from storage rather than through the API — so the adapter needs a public URL, not
just get/put/delete.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BlogMediaStore(ABC):
    @abstractmethod
    def put(self, path: str, payload: bytes, content_type: str) -> None:
        """Store one object (the PDF, or one rasterized page) at ``path``."""

    @abstractmethod
    def get(self, path: str) -> Optional[bytes]:
        ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Remove every object under ``prefix`` — a post's PDF and all its pages."""

    @abstractmethod
    def url_for(self, path: str) -> str:
        """What a browser should fetch to read this object."""

    def ensure_bucket(self) -> None:
        """No-op by default; the Supabase adapter overrides this."""
