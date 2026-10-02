"""Postgres store for accounts, their sessions and password resets.

Tokens (session and reset) arrive here already hashed: this module never sees a
value that could be replayed as a cookie or a reset link. Hashing lives in
``api/security.py``, the one place that also mints them.
"""

from typing import Optional, Tuple

from psycopg import errors

from src.domain.ports.accounts import Account, Session
from src.infrastructure.postgres.pool import Database

_ACCOUNT = "select id, email, role, lang, created_at, last_login_at from accounts"


class AccountExists(Exception):
    """The email already belongs to an account."""


class PostgresAccountRepository:
    def __init__(self, db: Database):
        self.db = db

    # --- Accounts ----------------------------------------------------------

    def create(self, email: str, password_hash: str, role: str, lang: str) -> Account:
        try:
            row = self.db.fetch_one(
                "insert into accounts (email, password_hash, role, lang) "
                "values (%s, %s, %s, %s) "
                "returning id, email, role, lang, created_at, last_login_at",
                (email, password_hash, role, lang),
            )
        except errors.UniqueViolation as error:
            raise AccountExists(email) from error
        return _account(row)

    def get(self, account_id: str) -> Optional[Account]:
        row = self.db.fetch_one(f"{_ACCOUNT} where id = %s", (account_id,))
        return _account(row) if row else None

    def by_email(self, email: str) -> Optional[Tuple[Account, str]]:
        """The account and its password hash — the one read that returns the hash."""
        row = self.db.fetch_one(
            "select id, email, role, lang, created_at, last_login_at, password_hash "
            "from accounts where email = %s",
            (email,),
        )
        return (_account(row), row["password_hash"]) if row else None

    def set_password_hash(self, account_id: str, password_hash: str) -> None:
        self.db.execute(
            "update accounts set password_hash = %s where id = %s",
            (password_hash, account_id),
        )

    def touch_login(self, account_id: str) -> None:
        self.db.execute(
            "update accounts set last_login_at = now() where id = %s", (account_id,)
        )

    def set_lang(self, account_id: str, lang: str) -> None:
        self.db.execute("update accounts set lang = %s where id = %s", (lang, account_id))

    # --- Strava attachment -------------------------------------------------

    def athlete_id_for(self, account_id: str) -> Optional[int]:
        row = self.db.fetch_one(
            "select id from athletes where account_id = %s", (account_id,)
        )
        return int(row["id"]) if row else None

    def account_id_of_athlete(self, athlete_id: int) -> Optional[str]:
        row = self.db.fetch_one(
            "select account_id from athletes where id = %s", (athlete_id,)
        )
        return str(row["account_id"]) if row and row["account_id"] else None

    def link_athlete(self, athlete_id: int, account_id: str) -> bool:
        """Attach the athlete to the account, unless another account holds it.

        A single conditional update, so two accounts racing for the same Strava
        athlete cannot both win. The account's saved plans follow the athlete.
        """
        linked = self.db.execute(
            "update athletes set account_id = %s, updated_at = now() "
            "where id = %s and (account_id is null or account_id = %s)",
            (account_id, athlete_id, account_id),
        )
        if not linked:
            return False
        self.db.execute(
            "update race_plans set account_id = %s "
            "where athlete_id = %s and account_id is null",
            (account_id, athlete_id),
        )
        return True

    # --- Sessions ----------------------------------------------------------

    def create_session(
        self, account_id: str, token_hash: bytes, ttl_s: int, user_agent: str = ""
    ) -> None:
        self.db.execute(
            "insert into sessions (account_id, token_hash, expires_at, last_seen_at, user_agent) "
            "values (%s, %s, now() + make_interval(secs => %s), now(), %s)",
            (account_id, token_hash, ttl_s, (user_agent or "")[:300] or None),
        )

    def session_by_token(self, token_hash: bytes) -> Optional[Tuple[Session, Account]]:
        """A live session and its account; an expired one reads as absent.

        The attached Strava athlete comes along in the same query: nearly every
        request needs it, and this read happens on every request.
        """
        row = self.db.fetch_one(
            "select s.id as session_id, s.expires_at, s.last_seen_at, "
            "a.id, a.email, a.role, a.lang, a.created_at, a.last_login_at, "
            "ath.id as athlete_id "
            "from sessions s join accounts a on a.id = s.account_id "
            "left join athletes ath on ath.account_id = a.id "
            "where s.token_hash = %s and s.expires_at > now()",
            (token_hash,),
        )
        if row is None:
            return None
        session = Session(
            id=str(row["session_id"]),
            account_id=str(row["id"]),
            expires_at=row["expires_at"],
            last_seen_at=row["last_seen_at"],
            athlete_id=int(row["athlete_id"]) if row["athlete_id"] is not None else None,
        )
        return session, _account(row)

    def extend_session(self, session_id: str, ttl_s: int) -> None:
        self.db.execute(
            "update sessions set expires_at = now() + make_interval(secs => %s), "
            "last_seen_at = now() where id = %s",
            (ttl_s, session_id),
        )

    def delete_session(self, token_hash: bytes) -> None:
        self.db.execute("delete from sessions where token_hash = %s", (token_hash,))

    def delete_sessions(self, account_id: str) -> None:
        self.db.execute("delete from sessions where account_id = %s", (account_id,))

    # --- Password resets ---------------------------------------------------

    def create_reset(self, token_hash: bytes, account_id: str, ttl_s: int) -> None:
        self.db.execute(
            "insert into password_resets (token_hash, account_id, expires_at) "
            "values (%s, %s, now() + make_interval(secs => %s))",
            (token_hash, account_id, ttl_s),
        )

    def consume_reset(self, token_hash: bytes) -> Optional[str]:
        """Mark a live reset token used and return its account — at most once."""
        row = self.db.fetch_one(
            "update password_resets set used_at = now() "
            "where token_hash = %s and used_at is null and expires_at > now() "
            "returning account_id",
            (token_hash,),
        )
        return str(row["account_id"]) if row else None

    # --- Rate limiting -----------------------------------------------------

    def hit(self, key: str, window_s: int) -> int:
        """Count one attempt against ``key`` in the current fixed window."""
        row = self.db.fetch_one(
            "insert into login_attempts (key, window_start, count) "
            "values (%s, to_timestamp(floor(extract(epoch from now()) / %s) * %s), 1) "
            "on conflict (key, window_start) do update "
            "set count = login_attempts.count + 1 returning count",
            (key, window_s, window_s),
        )
        return int(row["count"])

    def prune_attempts(self) -> None:
        self.db.execute(
            "delete from login_attempts where window_start < now() - interval '1 day'"
        )


def _account(row) -> Account:
    return Account(
        id=str(row["id"]),
        email=str(row["email"]),
        role=row["role"],
        lang=row["lang"] or "en",
        created_at=row.get("created_at"),
        last_login_at=row.get("last_login_at"),
    )
