"""Password hashing and the password policy (design/specs/auth.md § Mots de passe).

argon2id with the library's defaults (m=64 MiB, t=3, p=4). The parameters are
encoded in each hash, so raising them later only needs :func:`verify` to report
that a stored hash is due for a re-hash — done at the next sign-in, when the
plain password is in hand.

The policy is length, not composition: 10 to 128 characters, and not one of the
10,000 most common passwords. Composition rules ("one digit, one symbol") push
people toward ``Password1!``, which every cracking list already has.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

MIN_LENGTH = 10
MAX_LENGTH = 128

# The 10,000 most frequent passwords of 10 characters or more, lowercased, from
# SecLists' Pwdb top-1,000,000. Shorter ones are refused by length already, so
# the list spends its whole budget on passwords the length rule lets through.
_COMMON_PATH = Path(__file__).with_name("data") / "common_passwords.txt"

_hasher = PasswordHasher()


@lru_cache(maxsize=1)
def _common() -> frozenset:
    return frozenset(_COMMON_PATH.read_text(encoding="utf-8").split())


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return _hasher.hash("not a real password, only spends the time a check would")


def policy_error(password: str) -> Optional[str]:
    """Why a new password is refused (a translation key suffix), or ``None``."""
    if len(password) < MIN_LENGTH:
        return "too_short"
    if len(password) > MAX_LENGTH:
        return "too_long"
    if password.lower() in _common():
        return "too_common"
    return None


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify(stored_hash: Optional[str], password: str) -> Tuple[bool, Optional[str]]:
    """Check a password; also return a fresh hash when the stored one is outdated.

    With no stored hash (unknown email), a dummy hash is verified anyway, so an
    unknown address costs the same time as a wrong password and the two cannot be
    told apart by timing.
    """
    if len(password) > MAX_LENGTH:
        # Don't hash megabytes for an attacker. No account can hold such a
        # password, so the answer is "no" for a known and an unknown email alike.
        return False, None
    if stored_hash is None:
        try:
            _hasher.verify(_dummy_hash(), password)
        except VerificationError:
            pass
        return False, None
    try:
        _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False, None
    if _hasher.check_needs_rehash(stored_hash):
        return True, _hasher.hash(password)
    return True, None
