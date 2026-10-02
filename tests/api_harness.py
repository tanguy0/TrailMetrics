"""Shared harness for the API tests that need a database.

They run only when ``TEST_DATABASE_URL`` points at a throwaway Postgres — never a
hosted one: they create and delete accounts. Only rows they create are deleted
(``@test.tagg`` accounts, athletes from ``ATHLETE_BASE`` up).
"""

import os
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet

from api.security import hash_token  # noqa: F401  (re-exported for tests)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
ATHLETE_BASE = 990_000_000
DOMAIN = "test.tagg"
SERVICE = {"x-service-token": "test-service-token", "x-client-ip": "203.0.113.7"}

requires_database = unittest.skipUnless(
    TEST_DATABASE_URL, "set TEST_DATABASE_URL to a throwaway Postgres"
)


class ApiTestCase(unittest.TestCase):
    """The API on a throwaway database, with helpers to sign up and attach Strava.
    Subclasses decorate themselves with ``requires_database``."""

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
        cls.db.execute("delete from login_attempts where key like %s", ("verify-account:%",))

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

