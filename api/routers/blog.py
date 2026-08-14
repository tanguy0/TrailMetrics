"""The blog: articles written by the master account, read by anyone.

Unlike every other router in this API, the read routes take **no** auth dependency
at all — the blog is public, meant to be shared and found without a TrailMetrics
account. Only the write routes are gated, by :func:`api.deps.require_master`.

An article's carousel is a single uploaded PDF, rasterized page-by-page into PNGs
at upload time (see ``src/infrastructure/pdf/rasterize.py``) — so the reader never
opens a PDF viewer, they just swipe through images.
"""

import logging
import mimetypes
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from api.config import get_settings
from api.deps import (
    get_athlete_repository,
    get_blog_media_store,
    get_database,
    require_master,
)
from api.security import read_session_token
from src.domain.ports.blog_media import BlogMediaStore
from src.domain.ports.storage import Athlete
from src.infrastructure.pdf.rasterize import TooManyPages, rasterize_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["blog"])

# 20 MB. A slide-deck export comfortably fits; well past that is almost certainly
# the wrong file.
MAX_PDF_BYTES = 20 * 1024 * 1024

EXCERPT_LENGTH = 220


@router.get("")
def list_posts() -> dict:
    """Published articles, newest first — the public blog index."""
    rows = get_database().fetch_all(
        "select id, slug, title, body_text, page_count, created_at from blog_posts "
        "where published order by created_at desc"
    )
    return {"posts": [_summary(row) for row in rows]}


@router.get("/admin")
def list_all_posts(_: Athlete = Depends(require_master)) -> dict:
    """Every article, drafts included — what the "write a post" screen lists."""
    rows = get_database().fetch_all(
        "select id, slug, title, body_text, page_count, published, created_at "
        "from blog_posts order by created_at desc"
    )
    return {"posts": [_summary(row) for row in rows]}


@router.get("/{slug}")
def get_post(slug: str, request: Request) -> dict:
    """One article. A draft 404s for anyone but the master account previewing it
    from the edit screen — see :func:`_is_master_request`."""
    query = "select * from blog_posts where slug = %s"
    if not _is_master_request(request):
        query += " and published"
    row = get_database().fetch_one(query, (slug,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such article.")
    return _detail(row)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_post(
    title: str = Form(...),
    body_text: str = Form(""),
    slug: str = Form(""),
    published: bool = Form(True),
    pdf: UploadFile = File(...),
    _: Athlete = Depends(require_master),
) -> dict:
    payload = await _read_pdf(pdf)
    pages = _rasterize(payload)

    post_id = f"blog_{uuid4().hex[:16]}"
    final_slug = _unique_slug(slug or title, exclude_id=None)
    store = get_blog_media_store()
    _store_pdf_and_pages(store, post_id, payload, pages)

    get_database().execute(
        "insert into blog_posts (id, slug, title, body_text, pdf_path, page_count, "
        "published) values (%s, %s, %s, %s, %s, %s, %s)",
        (post_id, final_slug, title.strip(), body_text, post_id, len(pages), published),
    )
    row = get_database().fetch_one("select * from blog_posts where id = %s", (post_id,))
    return _detail(row)


@router.patch("/{post_id}")
async def update_post(
    post_id: str,
    title: Optional[str] = Form(None),
    body_text: Optional[str] = Form(None),
    slug: Optional[str] = Form(None),
    published: Optional[bool] = Form(None),
    pdf: Optional[UploadFile] = File(None),
    _: Athlete = Depends(require_master),
) -> dict:
    row = get_database().fetch_one("select * from blog_posts where id = %s", (post_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such article.")

    updates: dict = {}
    if title is not None:
        updates["title"] = title.strip()
    if body_text is not None:
        updates["body_text"] = body_text
    if slug is not None and slug.strip() and slug.strip() != row["slug"]:
        updates["slug"] = _unique_slug(slug, exclude_id=post_id)
    if published is not None:
        updates["published"] = published

    store = get_blog_media_store()
    if pdf is not None:
        payload = await _read_pdf(pdf)
        pages = _rasterize(payload)
        store.delete_prefix(post_id)
        _store_pdf_and_pages(store, post_id, payload, pages)
        updates["page_count"] = len(pages)

    if updates:
        updates["updated_at"] = datetime.now(timezone.utc)
        set_clause = ", ".join(f"{key} = %s" for key in updates)
        get_database().execute(
            f"update blog_posts set {set_clause} where id = %s",
            (*updates.values(), post_id),
        )

    row = get_database().fetch_one("select * from blog_posts where id = %s", (post_id,))
    return _detail(row)


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: str, _: Athlete = Depends(require_master)) -> None:
    get_blog_media_store().delete_prefix(post_id)
    get_database().execute("delete from blog_posts where id = %s", (post_id,))


@router.get("/media/{path:path}")
def get_media(path: str) -> Response:
    """Serves blog media from local disk. Only reachable in practice when the app
    isn't using Supabase Storage — a Supabase-backed article's ``url_for`` points
    straight at its public bucket URL instead of this route."""
    data = get_blog_media_store().get(path)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such file.")
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --- Helpers -----------------------------------------------------------------

def _is_master_request(request: Request) -> bool:
    """Non-throwing version of :func:`api.deps.require_master`, for the one public
    route (the article page) that should still show a draft to its author.

    Mirrors ``current_athlete_id``'s ``DEV_ATHLETE_ID`` fallback (api/deps.py) so
    the preview also works against a local, session-less dev server.
    """
    settings = get_settings()
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    athlete_id = read_session_token(token, settings.session_secret)
    if athlete_id is None:
        if not settings.allow_dev_athlete:
            return False
        athlete_id = int(settings.dev_athlete_id)
    athlete = get_athlete_repository().get(athlete_id)
    return athlete is not None and settings.is_master(athlete.email)


async def _read_pdf(pdf: UploadFile) -> bytes:
    content_type = (pdf.content_type or "").split(";")[0].strip().lower()
    if content_type != "application/pdf" and not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are accepted."
        )
    payload = await pdf.read()
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Empty upload.")
    if len(payload) > MAX_PDF_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF is {len(payload) // 1024} kB; the limit is "
                   f"{MAX_PDF_BYTES // 1024} kB.",
        )
    return payload


def _rasterize(payload: bytes) -> list:
    try:
        return rasterize_pdf(payload)
    except TooManyPages as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error))
    except Exception as error:
        logger.warning("could not rasterize uploaded PDF: %s", error)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Could not read this PDF."
        )


def _store_pdf_and_pages(
    store: BlogMediaStore, post_id: str, pdf_payload: bytes, pages: list
) -> None:
    store.put(f"{post_id}/source.pdf", pdf_payload, "application/pdf")
    for index, page_bytes in enumerate(pages, start=1):
        store.put(f"{post_id}/page-{index:04d}.png", page_bytes, "image/png")


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "article"


def _unique_slug(source: str, exclude_id: Optional[str]) -> str:
    base = _slugify(source)
    candidate = base
    suffix = 2
    database = get_database()
    while True:
        query = "select id from blog_posts where slug = %s"
        params = (candidate,)
        if exclude_id is not None:
            query += " and id != %s"
            params = (candidate, exclude_id)
        if database.fetch_one(query, params) is None:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1


def _page_urls(post_id: str, page_count: int) -> list:
    store = get_blog_media_store()
    return [store.url_for(f"{post_id}/page-{i:04d}.png") for i in range(1, page_count + 1)]


def _summary(row: dict) -> dict:
    body = row["body_text"] or ""
    excerpt = body if len(body) <= EXCERPT_LENGTH else body[:EXCERPT_LENGTH].rstrip() + "…"
    urls = _page_urls(row["id"], row["page_count"])
    return {
        "id": row["id"],
        "slug": row["slug"],
        "title": row["title"],
        "excerpt": excerpt,
        "cover_url": urls[0] if urls else None,
        "page_count": row["page_count"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        **({"published": row["published"]} if "published" in row else {}),
    }


def _detail(row: dict) -> dict:
    return {
        "id": row["id"],
        "slug": row["slug"],
        "title": row["title"],
        "body_text": row["body_text"],
        "page_urls": _page_urls(row["id"], row["page_count"]),
        "page_count": row["page_count"],
        "published": row["published"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }
