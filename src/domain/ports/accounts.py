"""The account: who signs in, as opposed to the Strava athlete whose data it is.

An account is an email and a password; a Strava athlete is an optional attachment
of it (``athletes.account_id``). The split is deliberate — see
design/specs/auth.md — so nothing Strava provides is ever stored here, and an
account exists, and can be used, before Strava is connected.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

ROLES = ("athlete", "coach", "master")


@dataclass
class Account:
    id: str
    email: str
    role: str = "athlete"
    lang: str = "en"
    # Whether the account has proven it holds `email` (a verification link, or
    # a completed password reset). Required before MASTER_EMAIL becomes master.
    email_verified: bool = False
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None

    @property
    def is_coach(self) -> bool:
        """Coach rights: accepting requests, browsing coached athletes.

        ``master`` (the operator) holds every right a coach has, plus the blog.
        """
        return self.role in ("coach", "master")

    @property
    def is_master(self) -> bool:
        return self.role == "master"


@dataclass
class Session:
    """A signed-in device. Only ever built from a token that was just presented."""

    id: str
    account_id: str
    expires_at: datetime
    last_seen_at: Optional[datetime] = None
    # The Strava athlete attached to the account, if any.
    athlete_id: Optional[int] = None
