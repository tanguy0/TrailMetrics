"""Session tokens.

The API issues its own short signed token identifying an athlete; the web app
stores it in a first-party ``httpOnly`` cookie and presents it as a bearer token
on each proxied call. Strava's tokens never reach the browser.

Keeping sessions separate from Strava credentials matters: a Strava access token
lives six hours and is a capability against a third party, while a session is ours
to expire and revoke.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from cryptography.fernet import Fernet

ALGORITHM = "HS256"
AUDIENCE = "trailmetrics-web"


def create_session_token(
    athlete_id: int, secret: str, ttl_days: int = 30
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(int(athlete_id)),
        "aud": AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=ttl_days)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def read_session_token(token: str, secret: str) -> Optional[int]:
    """The athlete id in a valid token, or ``None`` for anything invalid.

    Every failure mode — bad signature, expired, wrong audience, malformed — is
    deliberately collapsed to ``None`` so callers can't accidentally distinguish
    them and leak that difference to a client.
    """
    if not token or not secret:
        return None
    try:
        payload = jwt.decode(
            token, secret, algorithms=[ALGORITHM], audience=AUDIENCE
        )
        return int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        return None


def constant_time_equals(left: str, right: str) -> bool:
    """Compare shared secrets without leaking their length or content by timing."""
    if not left or not right:
        return False
    return secrets.compare_digest(left, right)


def generate_keys() -> dict:
    """Fresh secrets, for filling in a new deployment's environment."""
    return {
        "SESSION_SECRET": secrets.token_urlsafe(48),
        "SERVICE_TOKEN": secrets.token_urlsafe(48),
        "ENCRYPTION_KEY": Fernet.generate_key().decode(),
    }
