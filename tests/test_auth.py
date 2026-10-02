"""Accounts, sessions and Strava attachment (design/specs/auth.md § Tests à exiger).

    /path/to/venv/bin/python -m unittest discover -s tests -t . -v

The password and token tests are pure. The API tests need a throwaway Postgres —
never a hosted one: they create and delete accounts. They run only when
``TEST_DATABASE_URL`` is set, e.g. against the local container from the README:

    TEST_DATABASE_URL=postgresql://postgres:tm@127.0.0.1:55432/trailmetrics

Only rows they create are deleted (``@test.tagg`` accounts, athletes from
``ATHLETE_BASE`` up), so pointing them at a dev database with data in it is safe.

``test_coach_sees_only_coached_athletes`` lands with the coaching tables.
"""

import unittest
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher

from api import passwords
from api.security import hash_token, new_token
from tests.api_harness import ATHLETE_BASE, DOMAIN, SERVICE, ApiTestCase, requires_database


class PasswordTest(unittest.TestCase):
    def test_hash_and_verify(self):
        stored = passwords.hash_password("correct horse battery")
        self.assertTrue(stored.startswith("$argon2id$"))
        self.assertEqual(passwords.verify(stored, "correct horse battery"), (True, None))
        self.assertEqual(passwords.verify(stored, "wrong horse battery"), (False, None))

    def test_unknown_account_still_spends_a_verification(self):
        self.assertEqual(passwords.verify(None, "whatever it is"), (False, None))

    def test_outdated_parameters_are_rehashed_on_success(self):
        weak = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)
        ok, rehashed = passwords.verify(weak.hash("correct horse battery"), "correct horse battery")
        self.assertTrue(ok)
        self.assertIsNotNone(rehashed)
        self.assertEqual(passwords.verify(rehashed, "correct horse battery"), (True, None))

    def test_policy(self):
        self.assertEqual(passwords.policy_error("short"), "too_short")
        self.assertEqual(passwords.policy_error("x" * 129), "too_long")
        self.assertEqual(passwords.policy_error("1234567890"), "too_common")
        self.assertEqual(passwords.policy_error("QwertyUiop"), "too_common")
        self.assertIsNone(passwords.policy_error("le col du galibier à 6h"))

    def test_overlong_password_is_refused_without_hashing(self):
        stored = passwords.hash_password("correct horse battery")
        self.assertEqual(passwords.verify(stored, "x" * 10_000), (False, None))


class TokenTest(unittest.TestCase):
    def test_tokens_are_random_and_hashed(self):
        first, second = new_token(), new_token()
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(len(first), 43)  # 32 bytes, base64url
        self.assertEqual(hash_token(first), hash_token(first))
        self.assertEqual(len(hash_token(first)), 32)


@requires_database
class AuthApiTest(ApiTestCase):
    # --- registration -----------------------------------------------------------

    def test_register_signs_in_an_account_without_strava(self):
        token = self.token_for("ana")
        body = self.me(token).json()
        self.assertFalse(body["strava_connected"])
        self.assertEqual(body["email"], f"ana@{DOMAIN}")
        self.assertEqual(body["account"]["role"], "athlete")
        self.assertEqual(body["lang"], "fr")

    def test_session_reports_the_tier(self):
        token = self.token_for("ana")
        response = self.client.get("/auth/session", headers=self.bearer(token))
        self.assertEqual(response.json()["strava_connected"], False)
        self.exchange(token, ATHLETE_BASE + 9)
        response = self.client.get("/auth/session", headers=self.bearer(token))
        self.assertEqual(response.json()["strava_connected"], True)

    def test_register_says_when_the_address_is_taken(self):
        self.token_for("ana")
        response = self.client.post(
            "/auth/register",
            json={"email": f"ANA@{DOMAIN}", "password": "un autre mot de passe"},
            headers=SERVICE,
        )
        self.assertEqual(response.status_code, 409)

    def test_register_enforces_the_password_policy(self):
        self.assertEqual(self.register("ana", password="1234567890").status_code, 422)
        self.assertEqual(self.register("ana", password="short").status_code, 422)

    def test_register_needs_the_service_token(self):
        response = self.client.post(
            "/auth/register", json={"email": f"x@{DOMAIN}", "password": "le col du galibier"}
        )
        self.assertEqual(response.status_code, 403)

    def test_master_email_is_not_master_before_proving_it(self):
        token = self.token_for("boss")
        account = self.me(token).json()["account"]
        self.assertEqual(account["role"], "athlete")
        self.assertFalse(account["email_verified"])

    # --- email verification -------------------------------------------------------

    def _with_mailbox(self):
        from unittest import mock

        sent = []

        class FakeSender:
            def send(self, to, subject, text):
                sent.append((to, text))

        return sent, mock.patch("api.routers.auth.get_mail_sender", return_value=FakeSender())

    @staticmethod
    def _token_in(text, path):
        return next(word for word in text.split() if path in word).rsplit("/", 1)[1]

    def test_verification_link_promotes_the_master_email(self):
        sent, mailbox = self._with_mailbox()
        with mailbox:
            token = self.token_for("boss")
        link_token = self._token_in(sent[0][1], "/verify/")
        response = self.client.post("/auth/verify/confirm", json={"token": link_token})
        self.assertEqual(response.status_code, 200, response.text)
        account = self.me(token).json()["account"]
        self.assertEqual(account["role"], "master")
        self.assertTrue(account["email_verified"])
        reused = self.client.post("/auth/verify/confirm", json={"token": link_token})
        self.assertEqual(reused.status_code, 400)

    def test_verification_does_not_promote_anyone_else(self):
        sent, mailbox = self._with_mailbox()
        with mailbox:
            token = self.token_for("ana")
        self.client.post("/auth/verify/confirm", json={"token": self._token_in(sent[0][1], "/verify/")})
        account = self.me(token).json()["account"]
        self.assertEqual(account["role"], "athlete")
        self.assertTrue(account["email_verified"])

    def test_reset_takes_a_squatted_master_email_back(self):
        squatter = self.token_for("boss")  # someone registered the address first
        sent, mailbox = self._with_mailbox()
        with mailbox:
            self.client.post("/auth/reset", json={"email": f"boss@{DOMAIN}"}, headers=SERVICE)
        confirm = {"token": self._token_in(sent[0][1], "/reset/"), "password": "le vrai propriétaire"}
        owner = self.client.post("/auth/reset/confirm", json=confirm, headers=SERVICE).json()
        self.assertEqual(self.me(squatter).status_code, 401)
        self.assertEqual(self.me(owner["session_token"]).json()["account"]["role"], "master")

    def test_resend_is_limited(self):
        token = self.token_for("ana")
        sent, mailbox = self._with_mailbox()
        with mailbox:
            for _ in range(3):
                response = self.client.post("/auth/verify/resend", headers=self.bearer(token))
                self.assertEqual(response.json(), {"sent": True, "verified": False})
            limited = self.client.post("/auth/verify/resend", headers=self.bearer(token))
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(len(sent), 3)

    # --- login ------------------------------------------------------------------

    def test_unknown_email_and_wrong_password_get_the_same_answer(self):
        self.token_for("ana")
        unknown = self.login("nobody")
        wrong = self.login("ana", password="pas le bon mot de passe")
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.json(), wrong.json())

    def test_login_rotates_the_session(self):
        first = self.token_for("ana")
        second = self.login("ana", previous_token=first).json()["session_token"]
        self.assertNotEqual(first, second)
        self.assertEqual(self.me(first).status_code, 401)
        self.assertEqual(self.me(second).status_code, 200)

    def test_too_many_attempts_get_429(self):
        self.token_for("ana")
        for _ in range(10):
            self.assertEqual(self.login("ana", password="pas le bon mot").status_code, 401)
        self.assertEqual(self.login("ana").status_code, 429)

    def test_signup_is_limited_per_ip(self):
        for n in range(5):
            self.assertEqual(self.register(f"u{n}", ip="203.0.113.9").status_code, 200)
        self.assertEqual(self.register("u5", ip="203.0.113.9").status_code, 429)

    # --- sessions ---------------------------------------------------------------

    def test_expired_session_is_refused(self):
        token = self.token_for("ana")
        self.db.execute(
            "update sessions set expires_at = now() - interval '1 second' where token_hash = %s",
            (hash_token(token),),
        )
        self.assertEqual(self.me(token).status_code, 401)

    def test_session_slides_when_used(self):
        token = self.token_for("ana")
        soon = datetime.now(timezone.utc) + timedelta(days=1)
        self.db.execute(
            "update sessions set expires_at = %s, last_seen_at = now() - interval '2 hours' "
            "where token_hash = %s",
            (soon, hash_token(token)),
        )
        self.assertEqual(self.me(token).status_code, 200)
        row = self.db.fetch_one(
            "select expires_at from sessions where token_hash = %s", (hash_token(token),)
        )
        self.assertGreater(row["expires_at"], soon + timedelta(days=28))

    def test_logout_and_logout_all(self):
        token = self.token_for("ana")
        other = self.login("ana").json()["session_token"]
        self.client.post("/auth/logout", headers=self.bearer(token))
        self.assertEqual(self.me(token).status_code, 401)
        self.assertEqual(self.me(other).status_code, 200)
        third = self.login("ana").json()["session_token"]
        self.client.post("/auth/logout-all", headers=self.bearer(third))
        self.assertEqual(self.me(other).status_code, 401)
        self.assertEqual(self.me(third).status_code, 401)

    # --- password reset -----------------------------------------------------------

    def test_reset_without_mail_points_to_the_operator(self):
        self.token_for("ana")
        response = self.client.post("/auth/reset", json={"email": f"ana@{DOMAIN}"}, headers=SERVICE)
        self.assertEqual(response.json(), {"sent": False, "contact": f"boss@{DOMAIN}"})

    def test_reset_flow_revokes_every_session(self):
        from unittest import mock

        sent = []

        class FakeSender:
            def send(self, to, subject, text):
                sent.append((to, text))

        old = self.token_for("ana")
        with mock.patch("api.routers.auth.get_mail_sender", return_value=FakeSender()):
            unknown = self.client.post("/auth/reset", json={"email": f"zed@{DOMAIN}"}, headers=SERVICE)
            known = self.client.post("/auth/reset", json={"email": f"ana@{DOMAIN}"}, headers=SERVICE)
        self.assertEqual(unknown.json(), known.json())
        self.assertEqual(len(sent), 1)
        link = next(word for word in sent[0][1].split() if "/reset/" in word)
        reset_token = link.rsplit("/", 1)[1]

        confirm = {"token": reset_token, "password": "un tout nouveau mot de passe"}
        response = self.client.post("/auth/reset/confirm", json=confirm, headers=SERVICE)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.me(old).status_code, 401)
        self.assertEqual(self.me(response.json()["session_token"]).status_code, 200)
        self.assertEqual(self.login("ana", password="un tout nouveau mot de passe").status_code, 200)
        reused = self.client.post("/auth/reset/confirm", json=confirm, headers=SERVICE)
        self.assertEqual(reused.status_code, 400)

    # --- Strava attachment ------------------------------------------------------------

    def test_strava_new_athlete_is_created_attached(self):
        token = self.token_for("ana")
        self.assertEqual(self.exchange(token, ATHLETE_BASE + 1).status_code, 200)
        body = self.me(token).json()
        self.assertTrue(body["strava_connected"])
        self.assertEqual(body["id"], ATHLETE_BASE + 1)
        self.assertEqual(body["email"], f"ana@{DOMAIN}")

    def test_strava_existing_athlete_without_account_is_attached(self):
        self.db.execute(
            "insert into athletes (id, firstname) values (%s, 'Historique')", (ATHLETE_BASE + 2,)
        )
        token = self.token_for("ana")
        self.assertEqual(self.exchange(token, ATHLETE_BASE + 2).status_code, 200)
        self.assertEqual(self.me(token).json()["id"], ATHLETE_BASE + 2)

    def test_strava_attached_to_another_account_is_refused(self):
        self.exchange(self.token_for("ana"), ATHLETE_BASE + 3)
        intruder = self.token_for("bob")
        response = self.exchange(intruder, ATHLETE_BASE + 3)
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.me(intruder).json()["strava_connected"])

    def test_one_strava_per_account(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 4)
        self.assertEqual(self.exchange(token, ATHLETE_BASE + 5).status_code, 409)

    def test_disconnect_keeps_the_athlete_attached(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 6)
        self.client.delete("/auth/strava", headers=self.bearer(token))
        body = self.me(token).json()
        self.assertTrue(body["strava_connected"])
        self.assertFalse(body["strava_authorized"])

    # --- access ---------------------------------------------------------------------

    def test_athlete_routes_need_strava(self):
        response = self.client.get("/activities", headers=self.bearer(self.token_for("ana")))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "strava_not_connected")

    def test_athlete_cannot_call_coach_routes(self):
        self.assertEqual(
            self.client.get("/coach/athletes", headers=self.bearer(self.token_for("ana"))).status_code,
            403,
        )
        boss = self.token_for("boss")
        self.db.execute("update accounts set role = 'master' where email = %s", (f"boss@{DOMAIN}",))
        self.assertEqual(
            self.client.get("/coach/athletes", headers=self.bearer(boss)).status_code, 200
        )

    def test_view_as_is_ignored_for_an_athlete(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 7)
        self.db.execute("insert into athletes (id, firstname) values (%s, 'Autre')", (ATHLETE_BASE + 8,))
        response = self.client.get(
            "/auth/me",
            headers={**self.bearer(token), "x-view-as-athlete-id": str(ATHLETE_BASE + 8)},
        )
        self.assertEqual(response.json()["id"], ATHLETE_BASE + 7)

    def test_no_session_is_401(self):
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        self.assertEqual(self.me("not-a-token").status_code, 401)


if __name__ == "__main__":
    unittest.main()
