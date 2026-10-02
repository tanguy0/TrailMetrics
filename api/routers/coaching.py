"""Coaching: the request, the coach's decision, and who is coached.

The page opens on ``is_coached`` — a row in ``coaching`` — never an environment
variable (design/specs/coaching.md). Everyone else sees the offer and a request
form; one pending request per account, editable or withdrawable while pending.
A decline closes the request (kept, never deleted) and a new one may follow
after 30 days.

Coaches (``role`` coach or master) see the pending requests and their athletes,
and decide. A new request is announced to them by email when mail is set up.
"""

import logging
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import (
    current_account,
    get_account_repository,
    get_coaching_repository,
    language,
    require_coach,
)
from api.mail import get_mail_sender
from src.domain.coaching import (
    MIN_PROOF_COUNT,
    can_request_again_at,
    looks_like_phone,
    normalize_phone,
)
from src.domain.ports.accounts import Account
from src.infrastructure.postgres.coaching_repository import PendingRequestExists
from src.translations import translate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coaching", tags=["coaching"])


class CoachingRequestBody(BaseModel):
    message: str = Field(default="", max_length=4000)
    contact: Literal["email", "phone"] = "email"
    phone: Optional[str] = Field(default=None, max_length=40)


@router.get("/me")
def my_coaching(account: Account = Depends(current_account)) -> dict:
    """Everything the Coaching page needs to pick its state."""
    coaching = get_coaching_repository()
    latest = coaching.latest_request(account.id)
    again = None
    if latest and latest["status"] == "declined":
        again = can_request_again_at(datetime.fromisoformat(latest["decided_at"]))
    count = coaching.coached_count()
    return {
        "coached": coaching.is_coached(account.id),
        "request": latest,
        "can_request_again_at": again.isoformat() if again else None,
        "email": account.email,
        # Proof figures come from the database, never written in; hidden below
        # MIN_PROOF_COUNT (coaching.md § 4).
        "proof": {"coached_count": count} if count >= MIN_PROOF_COUNT else None,
    }


@router.put("/request")
def put_request(
    body: CoachingRequestBody,
    account: Account = Depends(current_account),
    lang: str = Depends(language),
) -> dict:
    """Send the request, or edit it while it is pending."""
    coaching = get_coaching_repository()
    if coaching.is_coached(account.id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail=translate("ui.coaching.error.coached", lang))

    phone = (body.phone or "").strip() or None
    if body.contact == "phone" and not phone:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=translate("ui.coaching.error.phone_needed", lang)
        )
    if phone and not looks_like_phone(phone):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=translate("ui.coaching.error.phone_invalid", lang)
        )
    e164 = normalize_phone(phone) if phone else None
    message = body.message.strip()

    updated = coaching.update_pending(account.id, message, phone, e164, body.contact)
    if updated is not None:
        return {"request": updated}

    latest = coaching.latest_request(account.id)
    if latest and latest["status"] == "declined":
        again = can_request_again_at(datetime.fromisoformat(latest["decided_at"]))
        if again is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail=translate("ui.coaching.error.too_soon", lang)
            )
    try:
        created = coaching.create_request(account.id, message, phone, e164, body.contact)
    except PendingRequestExists:
        # Two submits racing: the other one created it; edit that one instead.
        created = coaching.update_pending(account.id, message, phone, e164, body.contact)
    _announce(account, created or {})
    return {"request": created}


@router.delete("/request")
def withdraw_request(account: Account = Depends(current_account)) -> dict:
    return {"withdrawn": get_coaching_repository().withdraw(account.id)}


@router.get("/requests")
def coach_board(coach: Account = Depends(require_coach)) -> dict:
    """The coach's "Athletes" card: pending requests, and the athletes coached."""
    coaching = get_coaching_repository()
    return {"pending": coaching.pending_requests(), "coached": coaching.coached_by(coach.id)}


@router.post("/requests/{request_id}/accept")
def accept(request_id: str, coach: Account = Depends(require_coach)) -> dict:
    coaching = get_coaching_repository()
    decided = coaching.decide(request_id, "accepted", coach.id)
    if decided is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No pending request with this id.")
    coaching.link(coach.id, decided["account_id"])
    return {"request": decided}


@router.post("/requests/{request_id}/decline")
def decline(request_id: str, coach: Account = Depends(require_coach)) -> dict:
    decided = get_coaching_repository().decide(request_id, "declined", coach.id)
    if decided is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No pending request with this id.")
    return {"request": decided}


def _announce(account: Account, request: dict) -> None:
    """Email the coaches about a new request — best effort, never a failure."""
    sender = get_mail_sender()
    if sender is None or not request:
        return
    recipients = get_coaching_repository().coach_emails()
    if not recipients:
        return
    contact = request.get("phone") if request.get("contact") == "phone" else account.email
    for email in recipients:
        found = get_account_repository().by_email(email)
        lang = found[0].lang if found else "en"
        try:
            sender.send(
                email,
                translate("ui.coaching.mail.subject", lang).replace("{email}", account.email),
                translate("ui.coaching.mail.body", lang)
                .replace("{email}", account.email)
                .replace("{contact}", contact or account.email)
                .replace("{message}", request.get("message") or "—"),
            )
        except Exception as error:
            logger.warning("could not send a coaching notification: %s", type(error).__name__)
