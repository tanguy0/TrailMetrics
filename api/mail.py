"""Outgoing mail, behind one small interface.

**Off in the current deployment** (no ``MAIL_FROM``): nothing is sent, and every
caller already handles that — no verification link, no coaching notification,
"Forgot your password?" points to ``MASTER_EMAIL``, resets go through
``python -m api.roles reset-link``. Turning it on is configuration only; see
docs/DEPLOYMENT.md § Mail.

Two adapters — Resend (an HTTP API, through the ``httpx`` the API already
carries) and plain SMTP (the standard library) — and a third state, *none*: with
no provider configured, :func:`get_mail_sender` returns ``None`` and callers fall
back to showing the operator's address ("write to …") instead of sending.

Which one is in play is decided by which variables are set, like storage in
:mod:`api.config`: ``MAIL_RESEND_API_KEY`` wins, then ``MAIL_SMTP_HOST``.
"""

import logging
import smtplib
from email.message import EmailMessage
from functools import lru_cache
from typing import Optional, Protocol

import httpx

from api.config import Settings, get_settings

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


class MailSender(Protocol):
    def send(self, to: str, subject: str, text: str) -> None:
        """Send a plain-text message. Raises on failure; callers decide."""


class ResendMailSender:
    def __init__(self, api_key: str, sender: str):
        self.api_key = api_key
        self.sender = sender

    def send(self, to: str, subject: str, text: str) -> None:
        response = httpx.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"from": self.sender, "to": [to], "subject": subject, "text": text},
            timeout=10.0,
        )
        response.raise_for_status()


class SmtpMailSender:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sender = sender

    def send(self, to: str, subject: str, text: str) -> None:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(text)
        with smtplib.SMTP(self.host, self.port, timeout=10) as smtp:
            smtp.starttls()
            if self.user:
                smtp.login(self.user, self.password)
            smtp.send_message(message)


def build_mail_sender(settings: Settings) -> Optional[MailSender]:
    if not settings.mail_from:
        return None
    if settings.mail_resend_api_key:
        return ResendMailSender(settings.mail_resend_api_key, settings.mail_from)
    if settings.mail_smtp_host:
        return SmtpMailSender(
            settings.mail_smtp_host,
            settings.mail_smtp_port,
            settings.mail_smtp_user,
            settings.mail_smtp_password,
            settings.mail_from,
        )
    return None


@lru_cache(maxsize=1)
def get_mail_sender() -> Optional[MailSender]:
    return build_mail_sender(get_settings())
