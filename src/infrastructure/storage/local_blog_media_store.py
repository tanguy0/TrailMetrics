"""Filesystem :class:`BlogMediaStore` — for local development.

Unlike the Supabase adapter, there is no public bucket to point a browser at, so
``url_for`` names the API's own serving route instead (see
``api/routers/blog.py``'s ``GET /blog/media/{path}``).
"""

from pathlib import Path
from typing import Optional

from src.domain.ports.blog_media import BlogMediaStore


class LocalBlogMediaStore(BlogMediaStore):
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, path: str, payload: bytes, content_type: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def get(self, path: str) -> Optional[bytes]:
        target = self.root / path
        # Refuse anything that escapes the root: paths are built server-side, but
        # a store is exactly the wrong place to trust that.
        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        return target.read_bytes() if target.is_file() else None

    def delete_prefix(self, prefix: str) -> None:
        base = self.root / prefix
        try:
            base.resolve().relative_to(self.root.resolve())
        except ValueError:
            return
        if not base.is_dir():
            return
        for child in base.iterdir():
            if child.is_file():
                child.unlink()
        base.rmdir()

    def url_for(self, path: str) -> str:
        return f"/api/proxy/blog/media/{path}"
