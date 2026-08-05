"""Images an athlete puts in a panel.

Bytes live in Postgres, not in the stream bucket. That is a deliberate trade: these
are a handful of small files per athlete, and keeping them in the database means the
feature works identically on a local Postgres and on Supabase — no bucket to create,
no public-URL or signed-URL story, no second storage adapter. :data:`MAX_ASSET_BYTES`
is what keeps the assumption honest; raise it and the trade stops holding.

Serving goes through this API rather than a CDN so an image stays behind the
athlete's session, like every other piece of their data.
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from api.deps import current_athlete, get_database
from src.domain.ports.storage import Athlete

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assets", tags=["assets"])

# 4 MB. Comfortable for a screenshot or a course profile, and small enough that a
# page of images stays a fast read and a row stays a sane thing to keep in Postgres.
MAX_ASSET_BYTES = 4 * 1024 * 1024

# Only formats a browser renders inline. An allowlist rather than a blocklist: an
# uploaded SVG is a script-execution vector, and "any image/*" would admit it.
ALLOWED_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


@router.get("")
def list_assets(athlete: Athlete = Depends(current_athlete)) -> dict:
    """The athlete's uploads, newest first — so a second panel can reuse one."""
    rows = get_database().fetch_all(
        "select id, filename, content_type, byte_size, created_at from assets "
        "where athlete_id = %s order by created_at desc limit 200",
        (athlete.id,),
    )
    return {"assets": [_summary(row) for row in rows]}


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_asset(
    file: UploadFile = File(...),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Store one image and return the URL an image block should point at.

    ``async`` unlike the rest of the API: the work here is reading an upload off the
    socket, which is I/O, not the CPU-bound computation the sync handlers exist for.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type {content_type or 'unknown'}. "
                   f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}.",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Empty upload.")
    if len(payload) > MAX_ASSET_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is {len(payload) // 1024} kB; the limit is "
                   f"{MAX_ASSET_BYTES // 1024} kB.",
        )

    asset_id = f"img_{uuid4().hex[:16]}"
    get_database().execute(
        "insert into assets (id, athlete_id, filename, content_type, byte_size, data) "
        "values (%s, %s, %s, %s, %s, %s)",
        (asset_id, athlete.id, (file.filename or "")[:200], content_type,
         len(payload), payload),
    )
    return {
        "id": asset_id,
        "content_type": content_type,
        "byte_size": len(payload),
        # What goes into the image block's `src`. Relative on purpose: the same
        # document then works behind localhost, a preview deploy and production.
        "url": asset_url(asset_id),
    }


@router.get("/{asset_id}")
def get_asset(asset_id: str, athlete: Athlete = Depends(current_athlete)) -> Response:
    """The image bytes.

    Scoped to the signed-in athlete: an id is not a capability, so someone else's
    id is a 404 rather than a picture. Cached hard because the id is content —
    an asset is never rewritten, only replaced by a new upload with a new id.
    """
    row = get_database().fetch_one(
        "select content_type, data from assets where id = %s and athlete_id = %s",
        (asset_id, athlete.id),
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such image.")
    return Response(
        content=bytes(row["data"]),
        media_type=row["content_type"],
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset(asset_id: str, athlete: Athlete = Depends(current_athlete)) -> None:
    get_database().execute(
        "delete from assets where id = %s and athlete_id = %s",
        (asset_id, athlete.id),
    )


def asset_url(asset_id: str) -> str:
    """The path an image block stores. Resolved by the web app's own proxy."""
    return f"/api/proxy/assets/{asset_id}"


def _summary(row: dict) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "content_type": row["content_type"],
        "byte_size": row["byte_size"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "url": asset_url(row["id"]),
    }
