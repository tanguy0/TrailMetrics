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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routers import (
    activities,
    assets,
    auth,
    home,
    pages,
    precompute,
    registry,
    render,
    training,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Open the pool and converge the schema on boot; close cleanly on shutdown."""
    if settings.has_database:
        from api.deps import get_database, get_stream_store

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

app.include_router(registry.router)
app.include_router(auth.router)
app.include_router(activities.router)
app.include_router(pages.router)
app.include_router(render.router)
app.include_router(home.router)
app.include_router(training.router)
app.include_router(precompute.router)
app.include_router(assets.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness plus a readable view of what is configured.

    ``missing_config`` is the fastest way to diagnose a half-configured deploy;
    it names environment variables, never their values.
    """
    return {
        "status": "ok",
        "database": settings.has_database,
        "storage": "supabase" if settings.uses_supabase_storage else "local",
        "strava": settings.has_strava,
        "dev_mode": settings.dev_mode,
        "missing_config": settings.missing_for_auth(),
    }
