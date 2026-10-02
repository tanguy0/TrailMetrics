"""Postgres store for coaching requests and the coach → athlete links.

Keyed by **account** on both sides: coaching is a relation between people, not
between Strava athletes. Where Strava is needed — the view-as check, the roster's
"last activity" — the athlete attached to the account is joined in.
"""

from typing import Any, Dict, List, Optional

from psycopg import errors

from src.infrastructure.postgres.pool import Database

_REQUEST = (
    "select id, account_id, message, phone, phone_e164, contact, status, created_at, "
    "decided_at from coaching_requests"
)


class PendingRequestExists(Exception):
    """The account already has a pending request."""


class PostgresCoachingRepository:
    def __init__(self, db: Database):
        self.db = db

    # --- The athlete's side ----------------------------------------------------

    def latest_request(self, account_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            f"{_REQUEST} where account_id = %s order by created_at desc limit 1",
            (account_id,),
        )
        return _request(row) if row else None

    def create_request(
        self, account_id: str, message: str, phone: Optional[str],
        phone_e164: Optional[str], contact: str,
    ) -> Dict[str, Any]:
        try:
            row = self.db.fetch_one(
                "insert into coaching_requests (account_id, message, phone, phone_e164, contact) "
                "values (%s, %s, %s, %s, %s) returning id, account_id, message, phone, "
                "phone_e164, contact, status, created_at, decided_at",
                (account_id, message, phone, phone_e164, contact),
            )
        except errors.UniqueViolation as error:
            raise PendingRequestExists(account_id) from error
        return _request(row)

    def update_pending(
        self, account_id: str, message: str, phone: Optional[str],
        phone_e164: Optional[str], contact: str,
    ) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            "update coaching_requests set message = %s, phone = %s, phone_e164 = %s, "
            "contact = %s where account_id = %s and status = 'pending' "
            "returning id, account_id, message, phone, phone_e164, contact, status, "
            "created_at, decided_at",
            (message, phone, phone_e164, contact, account_id),
        )
        return _request(row) if row else None

    def withdraw(self, account_id: str) -> bool:
        return bool(self.db.execute(
            "update coaching_requests set status = 'withdrawn', decided_at = now() "
            "where account_id = %s and status = 'pending'",
            (account_id,),
        ))

    def is_coached(self, account_id: str) -> bool:
        return self.db.fetch_one(
            "select 1 from coaching where athlete_id = %s limit 1", (account_id,)
        ) is not None

    # --- The coach's side --------------------------------------------------------

    def pending_requests(self) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "select r.id, r.account_id, r.message, r.phone, r.phone_e164, r.contact, "
            "r.status, r.created_at, r.decided_at, a.email, "
            "nullif(trim(coalesce(ath.firstname, '') || ' ' || coalesce(ath.lastname, '')), '') "
            "as display_name "
            "from coaching_requests r join accounts a on a.id = r.account_id "
            "left join athletes ath on ath.account_id = r.account_id "
            "where r.status = 'pending' order by r.created_at",
        )
        return [_request(row) | {"email": str(row["email"]), "display_name": row["display_name"]}
                for row in rows]

    def decide(self, request_id: str, status: str, coach_id: str) -> Optional[Dict[str, Any]]:
        """Accept or decline a pending request — at most once."""
        row = self.db.fetch_one(
            "update coaching_requests set status = %s, decided_at = now(), decided_by = %s "
            "where id = %s and status = 'pending' "
            "returning id, account_id, message, phone, phone_e164, contact, status, "
            "created_at, decided_at",
            (status, coach_id, request_id),
        )
        return _request(row) if row else None

    def link(self, coach_id: str, athlete_account_id: str) -> None:
        self.db.execute(
            "insert into coaching (coach_id, athlete_id) values (%s, %s) "
            "on conflict do nothing",
            (coach_id, athlete_account_id),
        )

    def coached_by(self, coach_id: str) -> List[Dict[str, Any]]:
        """The coach's athletes: who, their Strava athlete, their last activity."""
        rows = self.db.fetch_all(
            "select c.athlete_id as account_id, a.email, c.since, ath.id as athlete_id, "
            "nullif(trim(coalesce(ath.firstname, '') || ' ' || coalesce(ath.lastname, '')), '') "
            "as display_name, ath.profile_url, "
            "(select max(start_date) from activities act where act.athlete_id = ath.id) "
            "as last_activity "
            "from coaching c join accounts a on a.id = c.athlete_id "
            "left join athletes ath on ath.account_id = c.athlete_id "
            "where c.coach_id = %s order by c.since",
            (coach_id,),
        )
        return [{
            "account_id": str(row["account_id"]),
            "email": str(row["email"]),
            "athlete_id": int(row["athlete_id"]) if row["athlete_id"] is not None else None,
            "display_name": row["display_name"] or str(row["email"]),
            "profile_url": row["profile_url"],
            "since": row["since"].isoformat(),
            "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
        } for row in rows]

    def coaches_athlete(self, coach_id: str, strava_athlete_id: int) -> bool:
        """Whether this coach may view this Strava athlete — the view-as check."""
        return self.db.fetch_one(
            "select 1 from coaching c join athletes ath on ath.account_id = c.athlete_id "
            "where c.coach_id = %s and ath.id = %s",
            (coach_id, strava_athlete_id),
        ) is not None

    def coached_count(self) -> int:
        row = self.db.fetch_one("select count(distinct athlete_id) as n from coaching")
        return int(row["n"])

    def coach_emails(self) -> List[str]:
        """Where a new request is announced: every coach whose address is proven."""
        rows = self.db.fetch_all(
            "select email from accounts where role in ('coach', 'master') "
            "and email_verified_at is not null"
        )
        return [str(row["email"]) for row in rows]


def _request(row) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "account_id": str(row["account_id"]),
        "message": row["message"],
        "phone": row["phone"],
        "phone_e164": row["phone_e164"],
        "contact": row["contact"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
        "decided_at": row["decided_at"].isoformat() if row["decided_at"] else None,
    }
