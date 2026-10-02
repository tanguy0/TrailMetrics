"""Accounts, sessions and Strava attachment (design/specs/auth.md § Tests à exiger).

    /path/to/venv/bin/python -m unittest discover -s tests -t . -v

The password and token tests are pure. The API tests need a throwaway Postgres —
never a hosted one: they create and delete accounts. They run only when
``TEST_DATABASE_URL`` is set, e.g. against the local container from the README:

    TEST_DATABASE_URL=postgresql://postgres:tm@127.0.0.1:55432/trailmetrics

Only rows they create are deleted (``@test.tagg`` accounts, athletes from
``ATHLETE_BASE`` up), so pointing them at a dev database with data in it is safe.

``test_coach_sees_only_coached_athletes`` lands with the coaching tables (PR 3).
"""

import os
import unittest
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from cryptography.fernet import Fernet

from api import passwords
from api.security import hash_token, new_token

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
ATHLETE_BASE = 990_000_000
DOMAIN = "test.tagg"
SERVICE = {"x-service-token": "test-service-token", "x-client-ip": "203.0.113.7"}


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


@unittest.skipUnless(TEST_DATABASE_URL, "set TEST_DATABASE_URL to a throwaway Postgres")
class AuthApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import api.config as config
        import api.deps as deps
        import api.mail as mail
        import api.main as main

        settings = config.Settings(
            database_url=TEST_DATABASE_URL,
            service_token=SERVICE["x-service-token"],
            encryption_key=Fernet.generate_key().decode(),
            strava_client_id="1",
            strava_client_secret="secret",
            master_email=f"boss@{DOMAIN}",
            web_app_url="http://web.test",
        )
        config._settings = settings
        main.settings = settings
        for cached in (
            deps.get_database, deps.get_account_repository, deps.get_athlete_repository,
            deps.get_activity_repository, deps.get_token_service, mail.get_mail_sender,
        ):
            cached.cache_clear()
        cls.deps = deps
        cls.main = main
        cls.db = deps.get_database()
        cls.db.apply_schema()
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls._clean()
        cls.deps.get_database.cache_clear()
        cls.db.close()

    @classmethod
    def _clean(cls):
        cls.db.execute("delete from athletes where id >= %s", (ATHLETE_BASE,))
        cls.db.execute("delete from accounts where email like %s", (f"%@{DOMAIN}",))
        cls.db.execute("delete from login_attempts where key like %s", (f"%{DOMAIN}%",))
        cls.db.execute("delete from login_attempts where key like %s", ("%203.0.113.%",))

    def setUp(self):
        self._clean()
        self.main._hits.clear()

    # --- helpers --------------------------------------------------------------

    def register(self, local, password="le col du galibier à 6h", ip="203.0.113.7"):
        return self.client.post(
            "/auth/register",
            json={"email": f"{local}@{DOMAIN}", "password": password, "lang": "fr"},
            headers={**SERVICE, "x-client-ip": ip},
        )

    def login(self, local, password="le col du galibier à 6h", **extra):
        return self.client.post(
            "/auth/login",
            json={"email": f"{local}@{DOMAIN}", "password": password, **extra},
            headers=SERVICE,
        )

    def token_for(self, local):
        response = self.register(local)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session_token"]

    def me(self, token):
        return self.client.get("/auth/me", headers={"authorization": f"Bearer {token}"})

    def bearer(self, token):
        return {"authorization": f"Bearer {token}"}

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

    def test_master_email_gets_the_master_role(self):
        token = self.token_for("boss")
        self.assertEqual(self.me(token).json()["account"]["role"], "master")

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

    def _fake_strava(self, athlete_id):
        from unittest import mock

        from src.domain.ports.storage import Athlete, StravaCredentials
        from src.infrastructure.strava.token_service import StravaTokenService

        class FakeService(StravaTokenService):
            def fetch_identity(self, code):
                return (
                    Athlete(id=athlete_id, firstname="Kilian", lastname="Test"),
                    StravaCredentials(
                        access_token="a", refresh_token="r",
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=6),
                    ),
                )

        service = FakeService("1", "secret", self.deps.get_athlete_repository())
        return mock.patch("api.routers.auth.get_token_service", return_value=service)

    def exchange(self, token, athlete_id, code="code"):
        with self._fake_strava(athlete_id):
            return self.client.post(
                "/auth/strava/exchange",
                json={"code": f"{code}-{athlete_id}-{token[:6]}"},
                headers={**SERVICE, **self.bearer(token)},
            )

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
        self.assertEqual(
            self.client.get("/coach/athletes", headers=self.bearer(self.token_for("boss"))).status_code,
            200,
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
