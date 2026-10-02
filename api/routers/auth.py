"""Accounts, sessions, Strava attachment and the account's own profile.

Identity is an email + password account (design/specs/auth.md). The routes that
mint a session — register, login, reset — are **server-to-server**: the browser
posts its form to the web app, which calls here with the shared service token
and puts the returned session in a first-party ``httpOnly`` cookie. The session
token never reaches client-side JavaScript; neither does a Strava token.

Strava no longer signs anyone in. From a signed-in account, "Connect Strava"
runs the OAuth flow, and the callback (on the web app, for the same first-party
cookie reasons) posts the code here with the account's session; the athlete is
then attached to the account — see :func:`exchange`.

Roles are earned by proof, not by typing: the account registered with
``MASTER_EMAIL`` only becomes ``master`` once it proves it holds the address —
a verification link, or a completed password reset (which also takes the
address back from anyone who registered it first). Other roles are set with
``python -m api.roles``.

Errors that could tell an attacker whether an email has an account are the same
for both cases: one message for an unknown email and a wrong password, and a
dummy hash verified for the unknown one so the timing matches too. Registration
does say "this address already has an account" — a deliberate, documented
choice (design/specs/auth.md § Sessions): without email verification, a generic
answer would hide nothing, since only a new account gets signed in.
"""

import logging
import re
import threading
import time
from datetime import date
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api import passwords
from api.config import get_settings
from api.deps import (
    STRAVA_NOT_CONNECTED,
    client_ip,
    current_account,
    current_athlete_id,
    get_account_repository,
    get_activity_repository,
    get_athlete_repository,
    get_coaching_repository,
    get_level_repository,
    get_token_service,
    invalidate_caches,
    language,
    require_service_token,
    session_token,
    session_ttl_s,
)
from api.mail import get_mail_sender
from api.rate_limit import (
    LOGIN_PER_EMAIL,
    LOGIN_PER_IP,
    LOGIN_WINDOW_S,
    RESET_PER_IP,
    SIGNUP_PER_IP,
    SIGNUP_WINDOW_S,
    over_limit,
)
from api.security import hash_token, new_token
from api.serialization import account_without_strava_payload, athlete_payload
from src.domain.ports.accounts import Account
from src.domain.ports.storage import Athlete
from src.infrastructure.postgres.account_repository import AccountExists
from src.translations import LANGUAGES, translate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

RESET_TTL_S = 30 * 60
VERIFY_TTL_S = 48 * 60 * 60
VERIFY_RESEND_PER_ACCOUNT = 3

# A Strava authorization code can be exchanged exactly once, but the callback can
# easily arrive twice — a double-clicked button, a browser retrying a redirect, a
# platform replaying the request. The second call would then fail and *that* is the
# response the user sees, even though the first one connected them. Replaying the
# response for a code we just exchanged makes the callback idempotent.
#
# Deliberately in-process and short-lived: the retry we care about lands within
# seconds, on the same worker. Across replicas the duplicate still fails and the
# user retries, which is the status quo, not a regression.
_RECENT_EXCHANGE_TTL_S = 120.0
_recent_exchanges: Dict[str, Tuple[float, dict]] = {}
_recent_lock = threading.Lock()


def _remember_exchange(key: str, response: dict) -> None:
    now = time.monotonic()
    with _recent_lock:
        _recent_exchanges[key] = (now, response)
        stale = [k for k, (at, _) in _recent_exchanges.items()
                 if now - at > _RECENT_EXCHANGE_TTL_S]
        for stale_key in stale:
            del _recent_exchanges[stale_key]


def _replay_exchange(key: str) -> Optional[dict]:
    with _recent_lock:
        entry = _recent_exchanges.get(key)
        if entry is None:
            return None
        at, response = entry
        if time.monotonic() - at > _RECENT_EXCHANGE_TTL_S:
            del _recent_exchanges[key]
            return None
    return response


def _fail(code: int, key: str, lang: str) -> HTTPException:
    """An error whose detail is already in the reader's language."""
    return HTTPException(code, detail=translate(f"ui.auth.error.{key}", lang))


class AuthorizeUrlRequest(BaseModel):
    redirect_uri: str = Field(min_length=1, max_length=500)
    state: str = ""


class ExchangeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=500)


class Credentials(BaseModel):
    # No pattern on the way in for login: a malformed address simply matches no
    # account, and must get the same answer as any other failed attempt.
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


# Deliberately permissive: "something@something.something", no dots-in-local-part
# rules, no TLD list. A stricter pattern rejects real addresses, and the only thing
# this validation can honestly promise is that the value is shaped like an email —
# whether it *works* is a question only a sent message answers.
_EMAIL_PATTERN = r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$"


class Registration(Credentials):
    lang: str = "en"


class LoginRequest(Credentials):
    # The session the browser already holds, if any: revoked on success, so a
    # sign-in always rotates the token rather than piling up sessions.
    previous_token: str = Field(default="", max_length=200)


class ResetRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)


class VerifyConfirm(BaseModel):
    token: str = Field(min_length=1, max_length=200)


class ResetConfirm(BaseModel):
    token: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=1024)


class ProfileUpdate(BaseModel):
    """A partial update of the athlete's own self-reported fields.

    Every field is optional *and* nullable, which are different things here: an
    absent key leaves the stored value alone, an explicit ``null`` clears it. That
    distinction is what lets one endpoint back several independently-edited widgets
    without them overwriting each other.
    """

    # Wide but sane bounds; power is unmodellable from a nonsense weight.
    weight_kg: Optional[float] = Field(default=None, ge=25, le=250)
    birthdate: Optional[date] = None
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    # No `email`: it is the account's sign-in identifier now, and changing it is
    # an account operation (password re-check), not a profile field.

    # Self-reported zones and VMA pace — display-only, see the module docstring
    # on `Athlete` for why there's no cross-field validation between them.
    hr_zone1_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone2_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone3_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_zone4_end: Optional[int] = Field(default=None, ge=30, le=250)
    hr_max: Optional[int] = Field(default=None, ge=30, le=250)
    vma_pace_s_per_km: Optional[float] = Field(default=None, ge=90, le=900)

    # Unlike the fields above, there is no "unset" state to clear it back to —
    # see the `Athlete.lang` docstring — so this is validated against the
    # known languages rather than just typed as `Optional[str]` and trusted.
    lang: Optional[str] = None

    model_config = {"extra": "forbid"}


@router.get("/reset/options")
def reset_options() -> dict:
    """Whether a reset can be emailed; if not, whom the reset page says to write to."""
    return {
        "can_send": get_mail_sender() is not None,
        "contact": get_settings().master_email,
    }


@router.post("/register", dependencies=[Depends(require_service_token)])
def register(payload: Registration, request: Request, lang: str = Depends(language)) -> dict:
    """Create an account and sign it in. Service-to-service only."""
    accounts = get_account_repository()
    if over_limit(accounts, (f"signup-ip:{client_ip(request)}", SIGNUP_PER_IP, SIGNUP_WINDOW_S)):
        raise _fail(status.HTTP_429_TOO_MANY_REQUESTS, "too_many", lang)

    email = payload.email.strip()
    if not re.match(_EMAIL_PATTERN, email):
        raise _fail(status.HTTP_422_UNPROCESSABLE_ENTITY, "email_invalid", lang)
    problem = passwords.policy_error(payload.password)
    if problem:
        raise _fail(status.HTTP_422_UNPROCESSABLE_ENTITY, f"password_{problem}", lang)

    # Every account starts as an athlete — MASTER_EMAIL included, until it
    # proves it holds the address (see the module docstring).
    try:
        account = accounts.create(
            email,
            passwords.hash_password(payload.password),
            "athlete",
            payload.lang if payload.lang in LANGUAGES else lang,
        )
    except AccountExists:
        raise _fail(status.HTTP_409_CONFLICT, "exists", lang)
    _send_verification(account)
    return _open_session(account, request)


@router.post("/login", dependencies=[Depends(require_service_token)])
def login(payload: LoginRequest, request: Request, lang: str = Depends(language)) -> dict:
    """Trade an email and a password for a session. Service-to-service only."""
    accounts = get_account_repository()
    email = payload.email.strip()
    if over_limit(
        accounts,
        (f"login-email:{email.lower()}", LOGIN_PER_EMAIL, LOGIN_WINDOW_S),
        (f"login-ip:{client_ip(request)}", LOGIN_PER_IP, LOGIN_WINDOW_S),
    ):
        raise _fail(status.HTTP_429_TOO_MANY_REQUESTS, "too_many", lang)

    found = accounts.by_email(email)
    ok, rehashed = passwords.verify(found[1] if found else None, payload.password)
    if not found or not ok:
        raise _fail(status.HTTP_401_UNAUTHORIZED, "credentials", lang)

    account = found[0]
    if rehashed:
        accounts.set_password_hash(account.id, rehashed)
    accounts.touch_login(account.id)
    if payload.previous_token:
        accounts.delete_session(hash_token(payload.previous_token))
    return _open_session(account, request)


@router.post("/logout")
def logout(request: Request) -> dict:
    """Revoke the presented session. Idempotent: an unknown token is already out."""
    token = session_token(request)
    if token:
        get_account_repository().delete_session(hash_token(token))
    return {"ok": True}


@router.post("/logout-all")
def logout_all(account: Account = Depends(current_account)) -> dict:
    """Revoke every session of the account — this device's included."""
    get_account_repository().delete_sessions(account.id)
    return {"ok": True}


@router.post("/reset", dependencies=[Depends(require_service_token)])
def request_reset(payload: ResetRequest, request: Request, lang: str = Depends(language)) -> dict:
    """Email a reset link — or say whom to write to when no mail is configured.

    The answer is the same whether the address has an account or not; so is a
    failure to send, which is logged rather than reported.
    """
    accounts = get_account_repository()
    if over_limit(accounts, (f"reset-ip:{client_ip(request)}", RESET_PER_IP, SIGNUP_WINDOW_S)):
        raise _fail(status.HTTP_429_TOO_MANY_REQUESTS, "too_many", lang)

    sender = get_mail_sender()
    if sender is None:
        return {"sent": False, "contact": get_settings().master_email}

    found = accounts.by_email(payload.email.strip())
    if found:
        account = found[0]
        token = new_token()
        accounts.create_reset(hash_token(token), account.id, RESET_TTL_S)
        link = f"{get_settings().web_app_url.rstrip('/')}/reset/{token}"
        try:
            sender.send(
                account.email,
                translate("ui.auth.reset.mail.subject", account.lang),
                translate("ui.auth.reset.mail.body", account.lang).replace("{link}", link),
            )
        except Exception as error:
            logger.warning("could not send a reset email: %s", type(error).__name__)
    return {"sent": True}


@router.post("/reset/confirm", dependencies=[Depends(require_service_token)])
def confirm_reset(payload: ResetConfirm, request: Request, lang: str = Depends(language)) -> dict:
    """Set a new password from a reset link, sign out everywhere, sign in here."""
    problem = passwords.policy_error(payload.password)
    if problem:
        raise _fail(status.HTTP_422_UNPROCESSABLE_ENTITY, f"password_{problem}", lang)

    accounts = get_account_repository()
    account_id = accounts.consume_reset(hash_token(payload.token))
    account = accounts.get(account_id) if account_id else None
    if account is None:
        raise _fail(status.HTTP_400_BAD_REQUEST, "reset_invalid", lang)
    accounts.set_password_hash(account.id, passwords.hash_password(payload.password))
    accounts.delete_sessions(account.id)
    # The link came to the address, so following it is proof of holding it.
    _mark_verified(account)
    return _open_session(account, request)


@router.post("/verify/confirm")
def confirm_verification(payload: VerifyConfirm, lang: str = Depends(language)) -> dict:
    """Follow a verification link. Needs no session: the token is the proof,
    and the link may well be opened on another device than the one signed in."""
    accounts = get_account_repository()
    account_id = accounts.consume_verification(hash_token(payload.token))
    account = accounts.get(account_id) if account_id else None
    if account is None:
        raise _fail(status.HTTP_400_BAD_REQUEST, "verify_invalid", lang)
    _mark_verified(account)
    return {"ok": True, "email": account.email}


@router.post("/verify/resend")
def resend_verification(
    account: Account = Depends(current_account), lang: str = Depends(language)
) -> dict:
    """Send a fresh verification link to the signed-in account's address."""
    if account.email_verified:
        return {"sent": False, "verified": True}
    accounts = get_account_repository()
    if over_limit(
        accounts, (f"verify-account:{account.id}", VERIFY_RESEND_PER_ACCOUNT, SIGNUP_WINDOW_S)
    ):
        raise _fail(status.HTTP_429_TOO_MANY_REQUESTS, "too_many", lang)
    return {"sent": _send_verification(account), "verified": False}


@router.post("/strava/url")
def authorize_url(payload: AuthorizeUrlRequest) -> dict:
    """The Strava consent URL to send the user to."""
    service = get_token_service()
    return {"url": service.authorization_url(payload.redirect_uri, payload.state)}


@router.post("/strava/exchange", dependencies=[Depends(require_service_token)])
def exchange(
    payload: ExchangeRequest,
    account: Account = Depends(current_account),
    lang: str = Depends(language),
) -> dict:
    """Attach the Strava athlete behind an authorization code to the account.

    Three cases (design/specs/auth.md § Rattachement Strava): a new athlete is
    created attached; an existing one with no account (a user from before
    accounts) is attached, and their history comes back with it; one attached to
    *another* account is refused, before anything is written. An account already
    holding a different athlete is refused too — one Strava per account.
    """
    settings = get_settings()
    missing = settings.missing_for_auth()
    if missing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server not configured for Strava; missing: {', '.join(missing)}",
        )

    replay_key = f"{account.id}:{payload.code}"
    replayed = _replay_exchange(replay_key)
    if replayed is not None:
        return replayed

    service = get_token_service()
    try:
        identity, credentials = service.fetch_identity(payload.code)
    except Exception as error:
        # An expired code, or one whose exchange we no longer remember; nothing
        # actionable for the client beyond starting again.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Could not complete Strava authorization: {error}",
        )

    accounts = get_account_repository()
    owner = accounts.account_id_of_athlete(identity.id)
    if owner is not None and owner != account.id:
        raise _fail(status.HTTP_409_CONFLICT, "strava_taken", lang)
    own = accounts.athlete_id_for(account.id)
    if own is not None and own != identity.id:
        raise _fail(status.HTTP_409_CONFLICT, "strava_other", lang)

    athlete = service.store(identity, credentials)
    if not accounts.link_athlete(athlete.id, account.id):
        # Lost a race to another account between the check and the write.
        raise _fail(status.HTTP_409_CONFLICT, "strava_taken", lang)

    response = {"athlete": {"id": athlete.id, "display_name": athlete.display_name}}
    _remember_exchange(replay_key, response)
    return response


@router.delete("/strava")
def disconnect_strava(account: Account = Depends(current_account)) -> dict:
    """Forget the account's Strava tokens. The athlete stays attached, data included,
    so reconnecting the same Strava picks everything back up."""
    athlete_id = get_account_repository().athlete_id_for(account.id)
    if athlete_id is not None:
        get_athlete_repository().delete_credentials(athlete_id)
    return {"ok": True}


@router.get("/session")
def session(request: Request, account: Account = Depends(current_account)) -> dict:
    """Who is signed in and which tier they reach — cheap, for the web app's shell.

    Read on every server render to pick the rail and the page variant, so it is
    the session lookup and nothing else: no activity summaries, unlike ``/me``.
    """
    return {
        "account": {"id": account.id, "email": account.email, "role": account.role},
        "strava_connected": request.state.account_athlete_id is not None,
        "is_coach": account.is_coach,
        "is_master": account.is_master,
        # Opens the Coaching page (design/specs/coaching.md: the predicate, not
        # an environment variable).
        "is_coached": get_coaching_repository().is_coached(account.id),
        "email_verified": account.email_verified,
        "lang": account.lang,
    }


@router.get("/me")
def me(
    request: Request,
    account: Account = Depends(current_account),
    lang: str = Depends(language),
) -> dict:
    """The signed-in account's profile — or, for a coach viewing another athlete,
    that athlete's. An account with no Strava gets the same shape, emptied."""
    athlete = _effective_athlete(request, account)
    if athlete is None:
        return _without_strava(account)
    return _me_payload(request, account, athlete)


@router.patch("/me")
def update_me(
    request: Request,
    payload: ProfileUpdate = Body(...),
    account: Account = Depends(current_account),
) -> dict:
    """Update the self-reported fields.

    The language belongs to the account and works without Strava; every other
    field belongs to the athlete, so needs one.

    Weight is the one with computational consequences: stored power is
    per-kilogram, so a new weight takes effect across the whole history with no
    recomputation — but the cached plot outputs have to go, since their numbers were
    scaled with the old value. Birthdate and height feed no metric, so they leave
    the cache alone.
    """
    touched = payload.model_dump(exclude_unset=True)

    if "lang" in touched:
        if payload.lang not in LANGUAGES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown language: {payload.lang!r}",
            )
        get_account_repository().set_lang(account.id, payload.lang)
        account.lang = payload.lang

    athlete = _effective_athlete(request, account)
    if athlete is None:
        if touched.keys() - {"lang"}:
            raise HTTPException(status.HTTP_409_CONFLICT, detail=STRAVA_NOT_CONNECTED)
        return _without_strava(account)

    athletes = get_athlete_repository()
    if "lang" in touched:
        athletes.set_lang(athlete.id, payload.lang)
        athlete.lang = payload.lang

    if "weight_kg" in touched:
        athletes.set_weight(athlete.id, payload.weight_kg)
        athlete.weight_kg = payload.weight_kg
        invalidate_caches(athlete.id)

    if "birthdate" in touched or "height_cm" in touched:
        # A partial update must not blank the field the client didn't mention.
        birthdate = payload.birthdate if "birthdate" in touched else athlete.birthdate
        height_cm = payload.height_cm if "height_cm" in touched else athlete.height_cm
        athletes.set_body(athlete.id, birthdate, height_cm)
        athlete.birthdate = birthdate
        athlete.height_cm = height_cm

    zone_fields = (
        "hr_zone1_end", "hr_zone2_end", "hr_zone3_end", "hr_zone4_end",
        "hr_max", "vma_pace_s_per_km",
    )
    if touched.keys() & set(zone_fields):
        # A partial update must not blank a zone the client didn't mention.
        values = {
            field: getattr(payload, field) if field in touched else getattr(athlete, field)
            for field in zone_fields
        }
        athletes.set_zones(athlete.id, **values)
        for field, value in values.items():
            setattr(athlete, field, value)

    return _me_payload(request, account, athlete)


# --- Helpers -----------------------------------------------------------------

def _send_verification(account: Account) -> bool:
    """Email a verification link. False when no mail is configured, or it failed —
    the account works regardless; only role promotion waits on it."""
    sender = get_mail_sender()
    if sender is None:
        return False
    token = new_token()
    get_account_repository().create_verification(hash_token(token), account.id, VERIFY_TTL_S)
    link = f"{get_settings().web_app_url.rstrip('/')}/verify/{token}"
    try:
        sender.send(
            account.email,
            translate("ui.auth.verify.mail.subject", account.lang),
            translate("ui.auth.verify.mail.body", account.lang).replace("{link}", link),
        )
    except Exception as error:
        logger.warning("could not send a verification email: %s", type(error).__name__)
        return False
    return True


def _mark_verified(account: Account) -> None:
    """Record the proof, and grant what waited on it: MASTER_EMAIL becomes master."""
    accounts = get_account_repository()
    accounts.mark_verified(account.id)
    account.email_verified = True
    if get_settings().is_master_email(account.email) and account.role != "master":
        accounts.set_role(account.id, "master")
        account.role = "master"

def _open_session(account: Account, request: Request) -> dict:
    """Mint a session for the account; the token leaves here once, never stored."""
    token = new_token()
    get_account_repository().create_session(
        account.id,
        hash_token(token),
        session_ttl_s(),
        request.headers.get("x-client-user-agent", ""),
    )
    return {
        "session_token": token,
        "expires_in_days": get_settings().session_ttl_days,
        "lang": account.lang,
    }


def _effective_athlete(request: Request, account: Account) -> Optional[Athlete]:
    """The athlete this request reads (view-as included), or ``None`` without Strava."""
    try:
        athlete_id = current_athlete_id(request, account)
    except HTTPException as error:
        if error.detail == STRAVA_NOT_CONNECTED:
            return None
        raise
    athlete = get_athlete_repository().get(athlete_id)
    if athlete is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown athlete.")
    return athlete


def _without_strava(account: Account) -> dict:
    """``/auth/me`` before Strava. The Zones card is the one that can be full: its
    VMA and HRmax come from the latest level estimate (design/tagg/access.md)."""
    payload = account_without_strava_payload(account)
    latest = get_level_repository(account.id).latest()
    if latest:
        payload["vma_pace_s_per_km"] = latest["result"].get("vma_pace_s_per_km")
        payload["hr_max"] = latest["result"].get("hr_max")
    payload["level_estimate"] = _estimate_meta(latest)
    return _with_account(payload, account, False)


def _estimate_meta(latest: Optional[dict]) -> Optional[dict]:
    """What the Zones card's "estimated on … · method" line needs."""
    if not latest:
        return None
    return {
        "method": latest["method"],
        "created_at": latest["created_at"],
        "vma_pace_s_per_km": latest["result"].get("vma_pace_s_per_km"),
    }


def _with_account(payload: dict, account: Account, strava_connected: bool) -> dict:
    payload["account"] = {
        "id": account.id,
        "email": account.email,
        "role": account.role,
        "email_verified": account.email_verified,
        # Without a mail provider no link was sent, so the app must not ask for one.
        "can_verify": get_mail_sender() is not None,
    }
    payload["strava_connected"] = strava_connected
    payload["is_coach"] = account.is_coach
    payload["is_master"] = account.is_master
    payload.setdefault("viewing_as", False)
    return payload


def _me_payload(request: Request, account: Account, athlete: Athlete) -> dict:
    activities = get_activity_repository()
    summaries = activities.summaries(athlete.id)
    payload = athlete_payload(
        athlete,
        sync=get_athlete_repository().get_sync_state(athlete.id),
        date_range=activities.date_range(athlete.id),
        activity_count=len(summaries),
        sport_types=sorted({row["sport_type"] for row in summaries}),
    )
    accounts = get_account_repository()
    viewing_as = athlete.id != request.state.real_athlete_id
    owner_id = account.id if not viewing_as else accounts.account_id_of_athlete(athlete.id)
    owner = account if not viewing_as else (accounts.get(owner_id) if owner_id else None)
    # The sign-in email, not the legacy `athletes.email` — except for a coach
    # browsing an athlete from before accounts, who has nothing else.
    payload["email"] = owner.email if owner else athlete.email
    payload["viewing_as"] = viewing_as
    # Whether Strava still answers for this athlete: disconnected keeps the data
    # (and `id`) but drops the tokens.
    connected = get_athlete_repository().get_credentials(athlete.id) is not None
    payload["strava_authorized"] = connected
    # Only the account's own estimate: a coach's would describe the wrong runner.
    payload["level_estimate"] = (
        None if viewing_as else _estimate_meta(get_level_repository(account.id).latest())
    )
    return _with_account(payload, account, True)
