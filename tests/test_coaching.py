"""Coaching (design/specs/coaching.md): requests, decisions, who sees whom.

    TEST_DATABASE_URL=… /path/to/venv/bin/python -m unittest discover -s tests -t . -v
"""

import unittest

from src.domain.coaching import looks_like_phone, normalize_phone
from tests.api_harness import ATHLETE_BASE, DOMAIN, ApiTestCase, requires_database


class PhoneTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_phone("06 12 34 56 78"), "+33612345678")
        self.assertEqual(normalize_phone("+44 20 7946 0958"), "+442079460958")
        self.assertEqual(normalize_phone("0044 20 7946 0958"), "+442079460958")
        self.assertIsNone(normalize_phone("12345"))

    def test_looks_like_phone(self):
        self.assertTrue(looks_like_phone("06.12.34.56.78"))
        self.assertFalse(looks_like_phone("call me"))


@requires_database
class CoachingApiTest(ApiTestCase):
    def coach_token(self, local="coach"):
        token = self.token_for(local)
        self.db.execute("update accounts set role = 'coach' where email = %s", (f"{local}@{DOMAIN}",))
        return token

    def ask(self, token, **body):
        return self.client.put("/coaching/request", json={"message": "Marathon en avril", **body},
                               headers=self.bearer(token))

    def test_request_edit_withdraw(self):
        token = self.token_for("ana")
        first = self.ask(token).json()["request"]
        edited = self.ask(token, message="Trail en juin", contact="phone", phone="06 12 34 56 78").json()["request"]
        self.assertEqual(first["id"], edited["id"])
        self.assertEqual(edited["phone_e164"], "+33612345678")
        self.client.delete("/coaching/request", headers=self.bearer(token))
        state = self.client.get("/coaching/me", headers=self.bearer(token)).json()
        self.assertEqual(state["request"]["status"], "withdrawn")
        self.assertFalse(state["coached"])

    def test_phone_contact_needs_a_number(self):
        self.assertEqual(self.ask(self.token_for("ana"), contact="phone").status_code, 422)

    def test_accept_opens_coaching(self):
        athlete = self.token_for("ana")
        request_id = self.ask(athlete).json()["request"]["id"]
        coach = self.coach_token()
        board = self.client.get("/coaching/requests", headers=self.bearer(coach)).json()
        self.assertIn(request_id, [r["id"] for r in board["pending"]])
        self.client.post(f"/coaching/requests/{request_id}/accept", headers=self.bearer(coach))
        self.assertTrue(self.client.get("/coaching/me", headers=self.bearer(athlete)).json()["coached"])
        self.assertTrue(self.client.get("/auth/session", headers=self.bearer(athlete)).json()["is_coached"])
        board = self.client.get("/coaching/requests", headers=self.bearer(coach)).json()
        self.assertEqual([a["email"] for a in board["coached"]], [f"ana@{DOMAIN}"])

    def test_decline_then_wait(self):
        athlete = self.token_for("ana")
        request_id = self.ask(athlete).json()["request"]["id"]
        coach = self.coach_token()
        self.client.post(f"/coaching/requests/{request_id}/decline", headers=self.bearer(coach))
        state = self.client.get("/coaching/me", headers=self.bearer(athlete)).json()
        self.assertEqual(state["request"]["status"], "declined")
        self.assertIsNotNone(state["can_request_again_at"])
        self.assertEqual(self.ask(athlete).status_code, 409)

    def test_athlete_cannot_decide(self):
        athlete = self.token_for("ana")
        request_id = self.ask(athlete).json()["request"]["id"]
        other = self.token_for("bob")
        for path in ("/coaching/requests", f"/coaching/requests/{request_id}/accept"):
            method = self.client.get if path.endswith("requests") else self.client.post
            self.assertEqual(method(path, headers=self.bearer(other)).status_code, 403)

    def test_coach_sees_only_coached_athletes(self):
        coached = self.token_for("ana")
        self.exchange(coached, ATHLETE_BASE + 30)
        stranger = self.token_for("bob")
        self.exchange(stranger, ATHLETE_BASE + 31)
        coach = self.coach_token()
        self.exchange(coach, ATHLETE_BASE + 32)
        request_id = self.ask(coached).json()["request"]["id"]
        self.client.post(f"/coaching/requests/{request_id}/accept", headers=self.bearer(coach))

        def viewing(athlete_id):
            return self.client.get(
                "/auth/me", headers={**self.bearer(coach), "x-view-as-athlete-id": str(athlete_id)}
            ).json()["id"]

        self.assertEqual(viewing(ATHLETE_BASE + 30), ATHLETE_BASE + 30)
        self.assertEqual(viewing(ATHLETE_BASE + 31), ATHLETE_BASE + 32)  # ignored: not coached
        roster = self.client.get("/coach/athletes", headers=self.bearer(coach)).json()["athletes"]
        self.assertEqual([a["id"] for a in roster], [ATHLETE_BASE + 30])

    def test_proof_hidden_below_three(self):
        body = self.client.get("/coaching/me", headers=self.bearer(self.token_for("ana"))).json()
        coached = self.db.fetch_one("select count(distinct athlete_id) as n from coaching")["n"]
        if coached < 3:
            self.assertIsNone(body["proof"])
