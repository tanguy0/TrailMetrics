"""Configuration, read once from the environment.

Plain dataclass rather than a settings library: the surface is small, and keeping
it dependency-free matters for container size on a service that already carries
pandas, scipy and XGBoost.

Two deployment shapes are supported by the same code:

* **local** — a Postgres you run yourself plus streams on disk. No cloud account
  needed to exercise the whole stack.
* **hosted** — Supabase Postgres plus Supabase Storage.

Which one is in play is decided by whether ``SUPABASE_URL`` is set, so there is no
"environment" flag to get wrong.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str) -> Optional[int]:
    raw = _env(name)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _env_int_list(name: str) -> List[int]:
    ids = []
    for part in _env(name).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            pass
    return ids


@dataclass
class Settings:
    # --- Data ---------------------------------------------------------------
    database_url: str = ""
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_bucket: str = "activity-streams"
    # Where streams go when Supabase isn't configured (local development).
    local_stream_root: Path = field(default_factory=lambda: Path(".streams"))
    # Blog PDFs and their rasterized pages. A separate, *public* bucket — unlike the
    # streams bucket, these are meant to be fetched directly by a browser with no
    # session.
    blog_media_bucket: str = "blog-media"
    local_blog_media_root: Path = field(
        default_factory=lambda: Path(".streams/blog-media")
    )

    # --- Strava -------------------------------------------------------------
    strava_client_id: str = ""
    strava_client_secret: str = ""

    # --- Sessions -----------------------------------------------------------
    # Signs the session token the web app stores in a first-party cookie.
    session_secret: str = ""
    session_ttl_days: int = 30
    # Fernet key; encrypts Strava tokens at rest.
    encryption_key: str = ""
    # Shared secret the web app presents to exchange an OAuth code. Prevents
    # anyone else from turning a stolen code into a session.
    service_token: str = ""

    # --- Web app ------------------------------------------------------------
    web_app_url: str = "http://localhost:3000"
    extra_cors_origins: List[str] = field(default_factory=list)

    # Set to bypass authentication and act as this athlete. Local use only —
    # refused unless the process is explicitly marked as development.
    dev_athlete_id: Optional[int] = None
    dev_mode: bool = False

    # Strava athlete ids allowed to browse another athlete's account read-mostly
    # (see api/deps.py's view-as override). Not a role stored in the database —
    # there is no admin UI for it, just this list, set once by whoever operates
    # the deployment.
    coach_athlete_ids: List[int] = field(default_factory=list)

    # The one account allowed to write blog posts. Not a role stored in the
    # database — same shape as `coach_athlete_ids`, but by email (an athlete id
    # differs per environment; this operator's email does not).
    master_email: str = "tanguy.blervacque@gmail.com"

    @classmethod
    def from_env(cls) -> "Settings":
        local_root = _env("LOCAL_STREAM_ROOT")
        local_blog_root = _env("LOCAL_BLOG_MEDIA_ROOT")
        origins = [o for o in _env("EXTRA_CORS_ORIGINS").split(",") if o.strip()]
        return cls(
            database_url=_env("DATABASE_URL"),
            supabase_url=_env("SUPABASE_URL"),
            supabase_service_key=_env("SUPABASE_SERVICE_KEY"),
            supabase_bucket=_env("SUPABASE_BUCKET", "activity-streams"),
            local_stream_root=Path(local_root) if local_root else Path(".streams"),
            blog_media_bucket=_env("BLOG_MEDIA_BUCKET", "blog-media"),
            local_blog_media_root=(
                Path(local_blog_root) if local_blog_root
                else Path(".streams/blog-media")
            ),
            master_email=_env("MASTER_EMAIL", "tanguy.blervacque@gmail.com"),
            strava_client_id=_env("STRAVA_CLIENT_ID"),
            strava_client_secret=_env("STRAVA_CLIENT_SECRET"),
            session_secret=_env("SESSION_SECRET"),
            session_ttl_days=int(_env("SESSION_TTL_DAYS", "30")),
            encryption_key=_env("ENCRYPTION_KEY"),
            service_token=_env("SERVICE_TOKEN"),
            web_app_url=_env("WEB_APP_URL", "http://localhost:3000"),
            extra_cors_origins=[o.strip() for o in origins],
            dev_athlete_id=_env_int("DEV_ATHLETE_ID"),
            dev_mode=_env("DEV_MODE").lower() in ("1", "true", "yes"),
            coach_athlete_ids=_env_int_list("COACH_ATHLETE_IDS"),
        )

    # --- Derived ------------------------------------------------------------

    @property
    def has_database(self) -> bool:
        return bool(self.database_url)

    @property
    def uses_supabase_storage(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def has_strava(self) -> bool:
        return bool(self.strava_client_id and self.strava_client_secret)

    @property
    def allow_dev_athlete(self) -> bool:
        """Impersonation is only ever allowed with DEV_MODE explicitly on."""
        return self.dev_mode and self.dev_athlete_id is not None

    def is_coach(self, athlete_id: int) -> bool:
        return athlete_id in self.coach_athlete_ids

    def is_master(self, email: Optional[str]) -> bool:
        return bool(email) and email.strip().lower() == self.master_email.lower()

    @property
    def cors_origins(self) -> List[str]:
        origins = [self.web_app_url, *self.extra_cors_origins]
        return [o for o in dict.fromkeys(origins) if o]

    def missing_for_auth(self) -> List[str]:
        """Which settings still have to be provided before login can work."""
        required = {
            "STRAVA_CLIENT_ID": self.strava_client_id,
            "STRAVA_CLIENT_SECRET": self.strava_client_secret,
            "SESSION_SECRET": self.session_secret,
            "ENCRYPTION_KEY": self.encryption_key,
            "SERVICE_TOKEN": self.service_token,
            "DATABASE_URL": self.database_url,
        }
        return [name for name, value in required.items() if not value]


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide settings, read from the environment on first use."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
