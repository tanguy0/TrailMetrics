"""Coaching rules that are not storage (design/specs/coaching.md § Règles)."""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

# After a decline, a new request waits this long.
REQUEST_AGAIN_AFTER = timedelta(days=30)
# The offer page shows "N athletes coached" only from this many: fewer reads as
# a weakness, not a proof.
MIN_PROOF_COUNT = 3

_LOOKS_LIKE_PHONE = re.compile(r"^\+?[\d\s().\-]{6,24}$")


def looks_like_phone(raw: str) -> bool:
    """No strict validation beyond "looks like a number" (coaching.md)."""
    return bool(_LOOKS_LIKE_PHONE.match(raw.strip())) and sum(c.isdigit() for c in raw) >= 6


def normalize_phone(raw: str, default_country_code: str = "33") -> Optional[str]:
    """E.164 when the number allows it, else ``None`` (the raw text is kept too).

    International numbers (``+…`` or ``00…``) keep their country code; a national
    number with a leading 0 takes ``default_country_code`` — France, where the
    coach is.
    """
    digits = re.sub(r"\D", "", raw)
    stripped = raw.strip()
    if stripped.startswith("+"):
        number = digits
    elif digits.startswith("00"):
        number = digits[2:]
    elif digits.startswith("0") and len(digits) == 10:
        number = default_country_code + digits[1:]
    else:
        return None
    return f"+{number}" if 8 <= len(number) <= 15 else None


def can_request_again_at(declined_at: Optional[datetime]) -> Optional[datetime]:
    """When a declined athlete may ask again; ``None`` if they already can."""
    if declined_at is None:
        return None
    again = declined_at + REQUEST_AGAIN_AFTER
    return again if again > datetime.now(timezone.utc) else None
