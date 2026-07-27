"""Filesystem :class:`StreamStore` — for local development and tests.

Same interface as the Supabase one, so the whole stack (sync, GAP fits,
stream-level plots) can be exercised without any cloud account.
"""

from pathlib import Path
from typing import Optional

from src.domain.ports.storage import StreamStore
from src.infrastructure.storage.codec import object_path


class LocalStreamStore(StreamStore):
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, athlete_id: int, activity_id: int, payload: bytes) -> str:
        path = object_path(athlete_id, activity_id)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return path

    def get(self, path: str) -> Optional[bytes]:
        target = self.root / path
        # Refuse anything that escapes the root: paths are built server-side, but
        # a store is exactly the wrong place to trust that.
        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        return target.read_bytes() if target.is_file() else None

    def delete(self, path: str) -> None:
        target = self.root / path
        if target.is_file():
            target.unlink()
