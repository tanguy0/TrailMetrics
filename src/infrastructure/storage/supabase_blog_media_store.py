"""Supabase Storage :class:`BlogMediaStore`, over the plain REST API.

Same ``httpx``-only shape as :mod:`supabase_stream_store`, but the bucket is
created **public**: a blog reader has no session, so pages must be fetched
directly from Supabase rather than proxied through the API.
"""

import logging
from typing import Optional

import httpx

from src.domain.ports.blog_media import BlogMediaStore

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "blog-media"


class SupabaseBlogMediaStore(BlogMediaStore):
    def __init__(
        self,
        project_url: str,
        service_key: str,
        bucket: str = DEFAULT_BUCKET,
        timeout: float = 30.0,
    ):
        self.project_url = project_url.rstrip("/")
        self.base = f"{self.project_url}/storage/v1"
        self.bucket = bucket
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {service_key}",
                "apikey": service_key,
            },
        )

    def ensure_bucket(self) -> None:
        response = self._client.post(
            f"{self.base}/bucket",
            json={"id": self.bucket, "name": self.bucket, "public": True},
        )
        if response.status_code in (200, 201) or _is_duplicate(response):
            return
        logger.warning("could not ensure bucket %s: %s %s",
                       self.bucket, response.status_code, response.text[:200])

    def put(self, path: str, payload: bytes, content_type: str) -> None:
        response = self._client.post(
            f"{self.base}/object/{self.bucket}/{path}",
            content=payload,
            headers={"Content-Type": content_type, "x-upsert": "true"},
        )
        response.raise_for_status()

    def get(self, path: str) -> Optional[bytes]:
        response = self._client.get(f"{self.base}/object/{self.bucket}/{path}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    def delete_prefix(self, prefix: str) -> None:
        listing = self._client.post(
            f"{self.base}/object/list/{self.bucket}",
            json={"prefix": prefix},
        )
        if listing.status_code != 200:
            return
        names = [f"{prefix}/{item['name']}" for item in listing.json()]
        if not names:
            return
        response = self._client.request(
            "DELETE", f"{self.base}/object/{self.bucket}", json={"prefixes": names}
        )
        if response.status_code not in (200, 204):
            logger.warning("could not delete blog media prefix %s: %s %s",
                           prefix, response.status_code, response.text[:200])

    def url_for(self, path: str) -> str:
        return f"{self.project_url}/storage/v1/object/public/{self.bucket}/{path}"

    def close(self) -> None:
        self._client.close()


def _is_duplicate(response: httpx.Response) -> bool:
    """Did this bucket already exist? See ``supabase_stream_store``'s twin."""
    if response.status_code == 409:
        return True
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return str(body.get("statusCode")) == "409" or body.get("error") == "Duplicate"
