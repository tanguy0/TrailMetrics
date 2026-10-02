"""Dependency wiring: one place that knows which adapter backs which port.

Also home to the per-athlete caches. Those matter more than they look: fitted
XGBoost models and per-second series are expensive enough that recomputing them on
every parameter tweak would make the builder unusable. They live here, keyed by
athlete and bounded, and are dropped when a sync changes the underlying data.

Two different bounds, because the two caches fail differently. Finished plot
outputs are small and numerous, so they are capped by *count* per athlete. The
memo holds decoded streams and fitted models — entries whose sizes differ by
orders of magnitude — so it is capped by *bytes*, and across the whole process
rather than per athlete: see :mod:`api.memo`.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Query, Request, status

from api.config import Settings, get_settings
from api.memo import AthleteMemo, MemoStore
from api.passwords import hash_password
from api.security import constant_time_equals, hash_token, new_token
from src.domain.ports.accounts import Account
from src.domain.ports.activity_data import ActivityDataSource
from src.domain.ports.page_repository import PageRepository
from src.domain.ports.storage import (
    ActivityRepository,
    Athlete,
    AthleteRepository,
    StreamStore,
)
from src.domain.charts.ir import PlotOutput
from src.infrastructure.postgres.activity_comment_repository import (
    PostgresActivityCommentRepository,
)
from src.infrastructure.postgres.account_repository import PostgresAccountRepository
from src.infrastructure.postgres.activity_repository import PostgresActivityRepository
from src.infrastructure.postgres.athlete_repository import PostgresAthleteRepository
from src.infrastructure.postgres.level_repository import PostgresLevelRepository
from src.infrastructure.postgres.page_repository import PostgresPageRepository
from src.infrastructure.postgres.planned_item_repository import (
    PostgresPlannedItemRepository,
)
from src.infrastructure.postgres.plot_output_repository import (
    PostgresPlotOutputRepository,
    signature_key,
)
from src.infrastructure.postgres.pool import Database
from src.infrastructure.postgres.precompute_repository import (
    PostgresPrecomputeRepository,
)
from src.infrastructure.postgres.race_plan_repository import PostgresRacePlanRepository
from src.infrastructure.postgres.stored_activity_data import StoredActivityData
from src.domain.ports.blog_media import BlogMediaStore
from src.infrastructure.storage.local_blog_media_store import LocalBlogMediaStore
from src.infrastructure.storage.local_stream_store import LocalStreamStore
from src.infrastructure.storage.supabase_blog_media_store import SupabaseBlogMediaStore
from src.infrastructure.storage.supabase_stream_store import SupabaseStreamStore
from src.infrastructure.strava.token_service import StravaTokenService
from src.translations import DEFAULT_LANG, LANGUAGES
from src.usecases.render_page import OutputCache, RenderContext

logger = logging.getLogger(__name__)

# The 409 detail an account with no Strava attached gets from any athlete route;
# the web app keys its "connect Strava" states on it.
STRAVA_NOT_CONNECTED = "strava_not_connected"

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
def get_blog_media_store() -> BlogMediaStore:
    """Supabase Storage (public bucket) when configured, otherwise local disk."""
    settings = get_settings()
    if settings.uses_supabase_storage:
        return SupabaseBlogMediaStore(
            settings.supabase_url,
            settings.supabase_service_key,
            bucket=settings.blog_media_bucket,
        )
    logger.info("using local blog media store at %s", settings.local_blog_media_root)
    return LocalBlogMediaStore(settings.local_blog_media_root)


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
def get_account_repository() -> PostgresAccountRepository:
    return PostgresAccountRepository(get_database())


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


def get_planned_item_repository(athlete_id: int) -> PostgresPlannedItemRepository:
    return PostgresPlannedItemRepository(get_database(), athlete_id)


def get_activity_comment_repository(athlete_id: int) -> PostgresActivityCommentRepository:
    return PostgresActivityCommentRepository(get_database(), athlete_id)


def get_plot_output_repository(athlete_id: int) -> PostgresPlotOutputRepository:
    return PostgresPlotOutputRepository(get_database(), athlete_id)


def get_precompute_repository(athlete_id: int) -> PostgresPrecomputeRepository:
    return PostgresPrecomputeRepository(get_database(), athlete_id)


def get_race_plan_repository(account_id: str) -> PostgresRacePlanRepository:
    return PostgresRacePlanRepository(get_database(), account_id)


def get_level_repository(account_id: str) -> PostgresLevelRepository:
    return PostgresLevelRepository(get_database(), account_id)


# --- Per-athlete caches ----------------------------------------------------

# One byte budget shared by every athlete this worker has served. Per-athlete
# would not bound the container: 16 athletes under a 192 MB cap each is 3 GB.
_memo_store = MemoStore()


@dataclass
class AthleteCaches:
    """Warm state for one athlete: fitted models, series, finished plot outputs."""

    memo: Any
    outputs: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)

    def trim(self) -> None:
        while len(self.outputs) > MAX_CACHED_OUTPUTS:
            self.outputs.popitem(last=False)


_caches: "OrderedDict[int, AthleteCaches]" = OrderedDict()


def get_caches(athlete_id: int) -> AthleteCaches:
    caches = _caches.get(athlete_id)
    if caches is None:
        caches = AthleteCaches(memo=AthleteMemo(_memo_store, athlete_id))
        _caches[athlete_id] = caches
        while len(_caches) > MAX_CACHED_ATHLETES:
            evicted, _ = _caches.popitem(last=False)
            # The memo outlives this dict, so dropping the athlete's entry here
            # would otherwise leak every byte they had memoized.
            _memo_store.discard_athlete(evicted)
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
    _memo_store.discard_athlete(athlete_id)


def memo_stats() -> Dict[str, Any]:
    """What the memo is holding — reported by ``GET /health`` for diagnosis."""
    return _memo_store.stats()


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
        hit = super().get(signature_key(signature))
        if hit is not None:
            return hit
        try:
            stored = get_plot_output_repository(self.athlete_id).get(signature)
        except Exception as error:
            logger.warning("could not read cached output: %s", error)
            return None
        if stored is not None:
            # Promote into memory so the next render in this process is free.
            super().set(signature_key(signature), "", stored)
        return stored

    def set(self, signature: str, plot_type: str, output: PlotOutput) -> None:
        # Hashed for the same reason the table hashes it: the raw signature embeds
        # every activity id in the panel, which is tens of kilobytes for a long
        # history — and this dict holds MAX_CACHED_OUTPUTS of them per athlete,
        # for MAX_CACHED_ATHLETES athletes, so the *keys* alone were tens of MB.
        super().set(signature_key(signature), plot_type, output)
        try:
            get_plot_output_repository(self.athlete_id).put(
                signature, plot_type, output
            )
        except Exception as error:
            logger.warning("could not store computed output: %s", error)


# --- Request-scoped -------------------------------------------------------

# How long a session lives without being used; each use pushes it out again.
# Re-written at most this often, so a burst of requests is not a burst of writes.
SESSION_TOUCH_EVERY = timedelta(hours=1)


def session_ttl_s() -> int:
    return get_settings().session_ttl_days * 24 * 60 * 60


def session_token(request: Request) -> str:
    """The presented session token: the proxy's bearer header, else the cookie."""
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return token or request.cookies.get("tm_session", "")


def optional_account(request: Request) -> Optional[Account]:
    """The signed-in account, or ``None`` — for the routes a visitor may call too.

    Resolved once per request and kept on ``request.state``, along with the
    attached Strava athlete (same query), which :func:`current_athlete_id` reads.

    ``DEV_ATHLETE_ID`` stands in for a session, but only when ``DEV_MODE`` is
    explicitly on, so a misconfigured production deploy can't accidentally
    authenticate everyone as one athlete.
    """
    if hasattr(request.state, "account"):
        return request.state.account
    account: Optional[Account] = None
    athlete_id: Optional[int] = None
    token = session_token(request)
    if token:
        found = get_account_repository().session_by_token(hash_token(token))
        if found is not None:
            session, account = found
            athlete_id = session.athlete_id
            now = datetime.now(timezone.utc)
            if session.last_seen_at is None or now - session.last_seen_at > SESSION_TOUCH_EVERY:
                get_account_repository().extend_session(session.id, session_ttl_s())
    if account is None and get_settings().allow_dev_athlete:
        account, athlete_id = _dev_account()
    request.state.account = account
    request.state.account_athlete_id = athlete_id
    return account


def current_account(account: Optional[Account] = Depends(optional_account)) -> Account:
    if account is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not signed in.")
    return account


def current_athlete_id(request: Request, account: Account = Depends(current_account)) -> int:
    """The Strava athlete this request acts as — almost always the account's own.

    An account with no Strava attached gets a 409 ``strava_not_connected``: every
    endpoint keyed on an athlete reads Strava data, so there is nothing for it
    to return yet, and the client tells that apart from "not signed in".

    A coach account (``role`` coach or master) can override this via the
    ``X-View-As-Athlete-Id`` header the web app attaches while browsing another
    athlete's account (see web/app/api/proxy). Every endpoint keyed on this
    dependency — which is nearly all of them — picks that up for free; the one
    exception is guarded explicitly with :func:`block_when_viewing_as`. The
    account's own athlete is still recorded on ``request.state`` for that check.
    """
    real_id: Optional[int] = request.state.account_athlete_id
    request.state.real_athlete_id = real_id

    view_as = request.headers.get("x-view-as-athlete-id", "").strip()
    if view_as and account.is_coach:
        try:
            target_id = int(view_as)
        except ValueError:
            target_id = None
        if target_id is not None and target_id != real_id:
            return target_id
    if real_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=STRAVA_NOT_CONNECTED)
    return real_id


def real_athlete_id(
    request: Request, _: int = Depends(current_athlete_id)
) -> Optional[int]:
    """The account's own athlete, ignoring any view-as override."""
    return request.state.real_athlete_id


def block_when_viewing_as(
    effective_id: int = Depends(current_athlete_id),
    real_id: Optional[int] = Depends(real_athlete_id),
) -> None:
    """Guard for the handful of actions only the athlete themself may take.

    Strava's tokens are theirs; a coach browsing their account should not be able
    to trigger a fetch of their private data from Strava on their behalf, even
    though the stored (encrypted) tokens would technically allow it.
    """
    if effective_id != real_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Not available while viewing another athlete's account.",
        )


def require_coach(account: Account = Depends(current_account)) -> Account:
    if not account.is_coach:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not a coach.")
    return account


def require_master(account: Account = Depends(current_account)) -> Account:
    """Gate for writing blog posts — the operator's account, by role."""
    if not account.is_master:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not the master account.")
    return account


def require_service_token(request: Request) -> None:
    """Guard the server-to-server routes so only our own web app can call them."""
    settings = get_settings()
    presented = request.headers.get("x-service-token", "")
    if not constant_time_equals(presented, settings.service_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Bad service token.")


def client_ip(request: Request) -> str:
    """The browser's IP, as the web app reports it on a service-token call.

    Every browser call reaches this API through the Next.js server, so the
    connection's own address is the web app's. The web app forwards the real
    one in ``X-Client-Ip``; only trust it alongside the service token.
    """
    presented = request.headers.get("x-service-token", "")
    if constant_time_equals(presented, get_settings().service_token):
        forwarded = request.headers.get("x-client-ip", "").strip()
        if forwarded:
            return forwarded[:64]
    return request.client.host if request.client else "unknown"


def language(lang: str = Query(DEFAULT_LANG)) -> str:
    """Requested UI language, falling back rather than erroring on a bad code."""
    return lang if lang in LANGUAGES else DEFAULT_LANG


def current_athlete(athlete_id: int = Depends(current_athlete_id)) -> Athlete:
    athlete = get_athlete_repository().get(athlete_id)
    if athlete is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown athlete.")
    return athlete


_dev_identity: Optional[Tuple[Account, Optional[int]]] = None


def _dev_account() -> Tuple[Account, Optional[int]]:
    """The account standing in for a session under ``DEV_MODE``.

    The one attached to ``DEV_ATHLETE_ID`` if there is one; otherwise one is made
    and attached, as ``master`` so every screen is reachable locally.
    """
    global _dev_identity
    if _dev_identity is not None:
        return _dev_identity
    athlete_id = int(get_settings().dev_athlete_id)
    accounts = get_account_repository()
    account_id = accounts.account_id_of_athlete(athlete_id)
    account = accounts.get(account_id) if account_id else None
    if account is None:
        email = f"dev-{athlete_id}@tagg.local"
        found = accounts.by_email(email)
        account = found[0] if found else accounts.create(
            email, hash_password(new_token()), "master", DEFAULT_LANG
        )
        if not accounts.link_athlete(athlete_id, account.id):
            logger.warning("DEV_ATHLETE_ID %s is not linkable to the dev account", athlete_id)
    _dev_identity = (account, accounts.athlete_id_for(account.id))
    return _dev_identity


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
