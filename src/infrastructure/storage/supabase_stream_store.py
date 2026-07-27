"""Supabase Storage :class:`StreamStore`, over the plain REST API.

Deliberately just ``httpx`` against three endpoints rather than the Supabase SDK:
it is the whole surface we need, it keeps the dependency list small enough to
matter for container size, and the service-role key never leaves the server.
"""

import logging
from typing import Optional

import httpx

from src.domain.ports.storage import StreamStore
from src.infrastructure.storage.codec import object_path

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "activity-streams"


class SupabaseStreamStore(StreamStore):
    """Blob storage for per-second arrays in a Supabase Storage bucket."""

    def __init__(
        self,
        project_url: str,
        service_key: str,
        bucket: str = DEFAULT_BUCKET,
        timeout: float = 30.0,
    ):
        self.base = f"{project_url.rstrip('/')}/storage/v1"
        self.bucket = bucket
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {service_key}",
                "apikey": service_key,
            },
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it isn't there. Private — reads go through the API."""
        response = self._client.post(
            f"{self.base}/bucket",
            json={"id": self.bucket, "name": self.bucket, "public": False},
        )
        if response.status_code in (200, 201) or _is_duplicate(response):
            return
        logger.warning("could not ensure bucket %s: %s %s",
                       self.bucket, response.status_code, response.text[:200])

    def put(self, athlete_id: int, activity_id: int, payload: bytes) -> str:
        path = object_path(athlete_id, activity_id)
        response = self._client.post(
            f"{self.base}/object/{self.bucket}/{path}",
            content=payload,
            headers={
                "Content-Type": "application/octet-stream",
                # Overwrite: re-syncing an activity should replace its streams.
                "x-upsert": "true",
            },
        )
        response.raise_for_status()
        return path

    def get(self, path: str) -> Optional[bytes]:
        response = self._client.get(f"{self.base}/object/{self.bucket}/{path}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    def delete(self, path: str) -> None:
        response = self._client.delete(f"{self.base}/object/{self.bucket}/{path}")
        if response.status_code not in (200, 204, 404):
            response.raise_for_status()

    def close(self) -> None:
        self._client.close()


def _is_duplicate(response: httpx.Response) -> bool:
    """Did this bucket already exist?

    Supabase Storage answers a duplicate bucket with HTTP **400** and puts the
    real status in the body (``{"statusCode": "409", "error": "Duplicate"}``), so
    the transport code alone cannot tell "already there" — the normal case on
    every restart — from a genuine failure.
    """
    if response.status_code == 409:
        return True
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return str(body.get("statusCode")) == "409" or body.get("error") == "Duplicate"
