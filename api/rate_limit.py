"""Rate limits for the auth endpoints (design/specs/auth.md § Limitation de débit).

Fixed windows counted in ``login_attempts``. Every attempt counts, successful or
not: the limits are generous enough for a person who mistypes, and counting only
failures would let an attacker reset the counter with one known-good login.

This is in addition to the coarse per-caller cap in ``api/main.py``, which is in
memory and per replica; these hold across replicas and restarts.
"""

import random
from typing import Tuple

from src.infrastructure.postgres.account_repository import PostgresAccountRepository

LOGIN_WINDOW_S = 15 * 60
LOGIN_PER_EMAIL = 10
LOGIN_PER_IP = 50
SIGNUP_WINDOW_S = 60 * 60
SIGNUP_PER_IP = 5
RESET_PER_IP = 5

# How often a call also sweeps out yesterday's counters.
_PRUNE_ONE_IN = 200

Check = Tuple[str, int, int]  # (key, limit, window seconds)


def over_limit(accounts: PostgresAccountRepository, *checks: Check) -> bool:
    """Count one attempt against every key; true if any is now past its limit.

    Every key is counted even once one is over, so the counters say what really
    happened rather than depending on the order of the checks.
    """
    if random.randrange(_PRUNE_ONE_IN) == 0:
        accounts.prune_attempts()
    exceeded = False
    for key, limit, window_s in checks:
        if accounts.hit(key, window_s) > limit:
            exceeded = True
    return exceeded
