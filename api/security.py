"""Session and reset tokens.

A session is an **opaque** random token, not a signed one: the web app stores it
in a first-party ``httpOnly`` cookie and presents it as a bearer token on each
proxied call, and the API looks its hash up in ``sessions``. Unlike the JWT it
replaces, a session can be revoked — sign out, sign out everywhere, a password
reset — because it only means something while its row exists.

Only the sha256 of a token is ever stored. 32 random bytes need no salt or slow
hash: there is nothing to brute-force, and the database dump that would leak the
hashes cannot be turned back into a cookie.

Strava's tokens never reach the browser, and are not sessions: a Strava access
token is a capability against a third party; a session is ours to expire.
"""

import hashlib
import secrets

from cryptography.fernet import Fernet

TOKEN_BYTES = 32


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def constant_time_equals(left: str, right: str) -> bool:
    """Compare shared secrets without leaking their length or content by timing."""
    if not left or not right:
        return False
    return secrets.compare_digest(left, right)


def generate_keys() -> dict:
    """Fresh secrets, for filling in a new deployment's environment."""
    return {
        "SERVICE_TOKEN": secrets.token_urlsafe(48),
        "ENCRYPTION_KEY": Fernet.generate_key().decode(),
    }
