"""Dependency wiring: one place that knows which adapter backs which port.

Also home to the per-athlete caches. Those matter more than they look: fitted
XGBoost models and per-second series are expensive enough that recomputing them on
every parameter tweak would make the builder unusable. They live here, keyed by
athlete and bounded, and are dropped when a sync changes the underlying data.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Query, Request, status

from api.config import Settings, get_settings
from api.security import constant_time_equals, read_session_token
from src.domain.ports.activity_data import ActivityDataSource
from src.domain.ports.page_repository import PageRepository
from src.domain.ports.storage import (
    ActivityRepository,
    Athlete,
    AthleteRepository,
    StreamStore,
)
from src.domain.charts.ir import PlotOutput
from src.infrastructure.postgres.activity_repository import PostgresActivityRepository
from src.infrastructure.postgres.athlete_repository import PostgresAthleteRepository
from src.infrastructure.postgres.page_repository import PostgresPageRepository
from src.infrastructure.postgres.plot_output_repository import (
    PostgresPlotOutputRepository,
)
from src.infrastructure.postgres.pool import Database
from src.infrastructure.postgres.precompute_repository import (
    PostgresPrecomputeRepository,
)
from src.infrastructure.postgres.stored_activity_data import StoredActivityData
from src.infrastructure.storage.local_stream_store import LocalStreamStore
from src.infrastructure.storage.supabase_stream_store import SupabaseStreamStore
from src.infrastructure.strava.token_service import StravaTokenService
from src.translations import DEFAULT_LANG, LANGUAGES
from src.usecases.render_page import OutputCache, RenderContext

logger = logging.getLogger(__name__)

# How many athletes keep warm caches in one process.
MAX_CACHED_ATHLETES = 16
# How many finished plot outputs to keep per athlete.
MAX_CACHED_OUTPUTS = 96


def _require_database(settings: Settings) -> None:
    if not settings.has_database:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No DATABASE_URL configured — this endpoint needs storage.",
        )


# --- Process singletons ----------------------------------------------------

@lru_cache(maxsize=1)
def get_database() -> Database:
    settings = get_settings()
    _require_database(settings)
    return Database(settings.database_url)


@lru_cache(maxsize=1)
def get_stream_store() -> StreamStore:
    """Supabase Storage when configured, otherwise the local filesystem."""
    settings = get_settings()
    if settings.uses_supabase_storage:
        return SupabaseStreamStore(
            settings.supabase_url,
            settings.supabase_service_key,
            bucket=settings.supabase_bucket,
        )
    logger.info("using local stream store at %s", settings.local_stream_root)
    return LocalStreamStore(settings.local_stream_root)


@lru_cache(maxsize=1)
def get_athlete_repository() -> AthleteRepository:
    settings = get_settings()
    if not settings.encryption_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No ENCRYPTION_KEY configured — Strava tokens cannot be stored.",
        )
    return PostgresAthleteRepository(get_database(), settings.encryption_key)


@lru_cache(maxsize=1)
def get_activity_repository() -> ActivityRepository:
    return PostgresActivityRepository(get_database())


@lru_cache(maxsize=1)
def get_token_service() -> StravaTokenService:
    settings = get_settings()
    if not settings.has_strava:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No Strava app credentials configured.",
        )
    return StravaTokenService(
        settings.strava_client_id,
        settings.strava_client_secret,
        get_athlete_repository(),
    )


def get_page_repository(athlete_id: int) -> PageRepository:
    return PostgresPageRepository(get_database(), athlete_id)


def get_plot_output_repository(athlete_id: int) -> PostgresPlotOutputRepository:
    return PostgresPlotOutputRepository(get_database(), athlete_id)


def get_precompute_repository(athlete_id: int) -> PostgresPrecomputeRepository:
    return PostgresPrecomputeRepository(get_database(), athlete_id)


# --- Per-athlete caches ----------------------------------------------------

@dataclass
class AthleteCaches:
    """Warm state for one athlete: fitted models, series, finished plot outputs."""

    memo: Dict[Any, Any] = field(default_factory=dict)
    outputs: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)

    def trim(self) -> None:
        while len(self.outputs) > MAX_CACHED_OUTPUTS:
            self.outputs.popitem(last=False)


_caches: "OrderedDict[int, AthleteCaches]" = OrderedDict()


def get_caches(athlete_id: int) -> AthleteCaches:
    caches = _caches.get(athlete_id)
    if caches is None:
        caches = AthleteCaches()
        _caches[athlete_id] = caches
        while len(_caches) > MAX_CACHED_ATHLETES:
            _caches.popitem(last=False)
    else:
        _caches.move_to_end(athlete_id)
    return caches


def invalidate_caches(athlete_id: int) -> None:
    """Drop an athlete's warm state — call whenever their stored data changes.

    Only the in-process state: the stored outputs in ``plot_outputs`` are keyed by
    the resolved activity ids, so new data produces new keys and the old rows are
    simply never read again. Deleting them is a separate, explicit act (the
    "recompute" action), not a side effect of importing a run.
    """
    _caches.pop(athlete_id, None)


class PersistentOutputCache(OutputCache):
    """Plot outputs in memory, backed by Postgres.

    Memory answers the editor's rapid re-renders; Postgres answers the case memory
    cannot — a fresh worker, or a page opened days after the fit — so an expensive
    curve is computed once per athlete rather than once per process.

    Database failures are swallowed on purpose. This is a cache: a read that fails
    is a miss, and a write that fails costs the *next* reader a recomputation. Either
    is strictly better than failing a render over it.
    """

    def __init__(self, athlete_id: int, store: Dict[str, PlotOutput]):
        super().__init__(store)
        self.athlete_id = athlete_id

    def get(self, signature: str) -> Optional[PlotOutput]:
        hit = super().get(signature)
        if hit is not None:
            return hit
        try:
            stored = get_plot_output_repository(self.athlete_id).get(signature)
        except Exception as error:
            logger.warning("could not read cached output: %s", error)
            return None
        if stored is not None:
            # Promote into memory so the next render in this process is free.
            super().set(signature, "", stored)
        return stored

    def set(self, signature: str, plot_type: str, output: PlotOutput) -> None:
        super().set(signature, plot_type, output)
        try:
            get_plot_output_repository(self.athlete_id).put(
                signature, plot_type, output
            )
        except Exception as error:
            logger.warning("could not store computed output: %s", error)


# --- Request-scoped -------------------------------------------------------

def current_athlete_id(request: Request) -> int:
    """The authenticated athlete, from the bearer session token.

    ``DEV_ATHLETE_ID`` bypasses this, but only when ``DEV_MODE`` is explicitly on,
    so a misconfigured production deploy can't accidentally authenticate everyone
    as one athlete.
    """
    settings = get_settings()
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.cookies.get("tm_session", "")

    athlete_id = read_session_token(token, settings.session_secret)
    if athlete_id is not None:
        return athlete_id
    if settings.allow_dev_athlete:
        return int(settings.dev_athlete_id)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not signed in.")


def require_service_token(request: Request) -> None:
    """Guard the OAuth exchange so only our own web app can call it."""
    settings = get_settings()
    presented = request.headers.get("x-service-token", "")
    if not constant_time_equals(presented, settings.service_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Bad service token.")


def language(lang: str = Query(DEFAULT_LANG)) -> str:
    """Requested UI language, falling back rather than erroring on a bad code."""
    return lang if lang in LANGUAGES else DEFAULT_LANG


def current_athlete(athlete_id: int = Depends(current_athlete_id)) -> Athlete:
    athlete = get_athlete_repository().get(athlete_id)
    if athlete is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown athlete.")
    return athlete


# --- Render plumbing ------------------------------------------------------

def data_source_for(athlete: Athlete) -> ActivityDataSource:
    return StoredActivityData(
        athlete.id,
        get_activity_repository(),
        get_stream_store(),
        mass_kg=athlete.weight_kg,
    )


def render_context_for(
    athlete: Athlete,
    lang: str,
    *,
    defer_expensive: bool = True,
    force_plot_ids: Optional[set] = None,
    refresh: bool = False,
) -> RenderContext:
    """A render context sharing this athlete's warm caches.

    ``defer_expensive`` leaves model fits uncomputed until asked for, so opening a
    page never blocks on an XGBoost fit the reader may not even scroll to.
    ``refresh`` does the opposite: recompute and overwrite, which is what the
    "recompute" action asks for.
    """
    if refresh:
        # The output cache is not the only memory: `memo` holds the fitted models and
        # the per-second series those outputs were built from. Recomputing without
        # dropping it would replay the same fit and return the same numbers, which
        # would make the button look broken.
        invalidate_caches(athlete.id)

    caches = get_caches(athlete.id)
    caches.trim()
    return RenderContext(
        data=data_source_for(athlete),
        lang=lang,
        mass_kg=athlete.weight_kg,
        memo=caches.memo,
        output_cache=PersistentOutputCache(athlete.id, caches.outputs),
        defer_expensive=defer_expensive,
        force_compute=set(force_plot_ids or ()),
        refresh=refresh,
    )
