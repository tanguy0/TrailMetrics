"""The TrailMetrics compute API.

Sits between the web app and the analytics. It owns Strava credentials, the stored
activity data, and every computation — and it returns chart IR, never rendered
figures, so the browser is purely presentation.

Handlers are **sync** on purpose: the work here is CPU-bound (pandas aggregation,
Savitzky–Golay filtering, XGBoost fits), so FastAPI runs them in its threadpool and
one athlete's model fit never stalls another's request. Async handlers over the same
work would block the event loop outright.
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import anyio.to_thread
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import (
    activities,
    assets,
    auth,
    blog,
    coach,
    home,
    pages,
    precompute,
    registry,
    render,
    training,
)
from api.security import read_session_token

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# How many CPU-bound handlers may run at once.
#
# Handlers here are sync, so FastAPI runs each in anyio's threadpool — which
# defaults to **40** threads. That default is sized for I/O, and this service is
# not: a render can hold a feature frame, a decoded history and a fitted model, so
# 40 of them at once is 40x the memory of one on a container sized for a handful.
# Requests past this queue instead of running, which is the right failure — a
# queued render is slow, a parallel one gets the worker OOM-killed. Raise it only
# alongside the memory to back it (see docs/DEPLOYMENT.md).
def _max_concurrent_requests() -> int:
    try:
        value = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "") or 4)
    except ValueError:
        value = 4
    return max(value, 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Open the pool and converge the schema on boot; close cleanly on shutdown."""
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = _max_concurrent_requests()
    logger.info("threadpool capped at %d concurrent handlers", limiter.total_tokens)

    if settings.has_database:
        from api.deps import get_blog_media_store, get_database, get_stream_store

        try:
            database = get_database()
            database.open()
            # Idempotent, so a fresh deploy converges without a migration step.
            database.apply_schema()
        except Exception:
            logger.exception("could not prepare the database")
        if settings.uses_supabase_storage:
            try:
                get_stream_store().ensure_bucket()
            except Exception:
                logger.exception("could not ensure the storage bucket")
            try:
                get_blog_media_store().ensure_bucket()
            except Exception:
                logger.exception("could not ensure the blog media bucket")
    else:
        logger.warning(
            "No DATABASE_URL — /registry works, but data endpoints will 503."
        )

    yield

    if settings.has_database:
        from api.deps import get_database

        try:
            get_database().close()
        except Exception:
            logger.exception("could not close the database pool")


app = FastAPI(
    title="TrailMetrics API",
    version="0.2.0",
    description="Composable running-data analysis: pages, panels, plots.",
    lifespan=lifespan,
)

# The web app normally proxies through its own origin, so CORS is a convenience for
# local development and any direct browser client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


# Per-caller request cap, in memory. In memory and per-process on purpose: this app
# runs one worker per container and scales by replicas, not workers (see
# docs/DEPLOYMENT.md), so a shared store isn't needed at the traffic this is sized
# for — scaling to several replicas would just make the limit per-replica rather
# than global, which is an acceptable trade at this scale.
#
# Keyed by athlete id when the caller has a valid session, not by IP: every browser
# call is proxied through the Next.js app (see web/app/api/proxy), so every athlete
# would otherwise appear to share whatever IP that proxy calls out from. Falling
# back to IP only covers the pre-session auth endpoints.
_RATE_WINDOW_S = 60.0
_DEFAULT_RATE_LIMIT = 120
_AUTH_RATE_LIMIT = 20
_AUTH_PATHS = {"/auth/strava/url", "/auth/strava/exchange"}

_hits: Dict[Tuple[str, str], deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limit_key(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        athlete_id = read_session_token(auth_header[7:], settings.session_secret)
        if athlete_id is not None:
            return f"athlete:{athlete_id}"
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else None
    if not ip:
        ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


# A caller whose bucket has aged out is forgotten rather than kept as an empty
# deque. Without this the dict only ever grows — one entry per athlete and per
# pre-session IP, for the life of the worker.
_HITS_SWEEP_EVERY = 512
_sweep_countdown = _HITS_SWEEP_EVERY


def _sweep_hits(now: float) -> None:
    """Drop buckets with nothing left in the window. Caller holds the lock."""
    stale = [
        key for key, hits in _hits.items()
        if not hits or now - hits[-1] > _RATE_WINDOW_S
    ]
    for key in stale:
        del _hits[key]


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    global _sweep_countdown
    is_auth_path = request.url.path in _AUTH_PATHS
    limit = _AUTH_RATE_LIMIT if is_auth_path else _DEFAULT_RATE_LIMIT
    bucket = "auth" if is_auth_path else "default"
    key = (_rate_limit_key(request), bucket)
    now = time.monotonic()
    with _hits_lock:
        hits = _hits[key]
        while hits and now - hits[0] > _RATE_WINDOW_S:
            hits.popleft()
        if len(hits) >= limit:
            return JSONResponse(
                {"detail": "Too many requests, slow down."}, status_code=429
            )
        hits.append(now)
        _sweep_countdown -= 1
        if _sweep_countdown <= 0:
            _sweep_countdown = _HITS_SWEEP_EVERY
            _sweep_hits(now)
    return await call_next(request)


app.include_router(registry.router)
app.include_router(auth.router)
app.include_router(activities.router)
app.include_router(pages.router)
app.include_router(render.router)
app.include_router(home.router)
app.include_router(training.router)
app.include_router(precompute.router)
app.include_router(assets.router)
app.include_router(coach.router)
app.include_router(blog.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness plus a readable view of what is configured.

    ``missing_config`` is the fastest way to diagnose a half-configured deploy;
    it names environment variables, never their values.
    """
    from api.deps import memo_stats

    return {
        "status": "ok",
        "database": settings.has_database,
        "storage": "supabase" if settings.uses_supabase_storage else "local",
        "strava": settings.has_strava,
        "dev_mode": settings.dev_mode,
        "missing_config": settings.missing_for_auth(),
        # What the worker is holding. The reason an OOM used to be un-diagnosable
        # was that nothing reported this: a container is killed from outside and
        # leaves no trace of what filled it.
        "memory": {**memo_stats(), "rss_bytes": _rss_bytes()},
    }


def _rss_bytes() -> Optional[int]:
    """Resident set size, read from procfs. ``None`` where that isn't available."""
    try:
        with open("/proc/self/statm", "r") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None
