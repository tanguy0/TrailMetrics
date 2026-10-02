"""The Tools tab's API: level estimates, saved plans by account, analysis templates.

    TEST_DATABASE_URL=… /path/to/venv/bin/python -m unittest discover -s tests -t . -v
"""

from tests.api_harness import ATHLETE_BASE, ApiTestCase, requires_database

RECORDS = {"method": "records", "inputs": {"records": [
    {"distance_m": 5000, "seconds": 1200}, {"distance_m": 10000, "seconds": 2460},
]}}


@requires_database
class ToolsApiTest(ApiTestCase):
    def test_a_visitor_can_estimate_but_nothing_is_saved(self):
        response = self.client.post("/tools/level/estimate", json=RECORDS)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body["saved_at"])
        self.assertEqual(len(body["zones"]), 5)
        self.assertGreater(body["vma_kmh"], 14)

    def test_invalid_inputs_say_why(self):
        response = self.client.post(
            "/tools/level/estimate?lang=fr",
            json={"method": "critical_speed", "inputs": {"d3_m": 1000, "d12_m": 1500}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("cohérentes", response.json()["detail"])

    def test_an_account_saves_and_home_reads_it_without_strava(self):
        token = self.token_for("ana")
        body = self.client.post(
            "/tools/level/estimate", json={**RECORDS, "hr_max": 188}, headers=self.bearer(token)
        ).json()
        self.assertIsNotNone(body["saved_at"])
        me = self.me(token).json()
        self.assertEqual(me["vma_pace_s_per_km"], body["vma_pace_s_per_km"])
        self.assertEqual(me["hr_max"], 188)
        self.assertEqual(me["level_estimate"]["method"], "records")
        latest = self.client.get("/tools/level/latest", headers=self.bearer(token)).json()
        self.assertEqual(latest["estimate"]["result"]["vdot"], body["vdot"])

    def test_the_estimate_becomes_the_athletes_vma(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 20)
        body = self.client.post(
            "/tools/level/estimate", json=RECORDS, headers=self.bearer(token)
        ).json()
        self.assertEqual(self.me(token).json()["vma_pace_s_per_km"], body["vma_pace_s_per_km"])

    def test_zone_definitions_are_served(self):
        body = self.client.get("/tools/zones").json()
        self.assertEqual([z["key"] for z in body["vma_pace"]][:2], ["z2", "endurance"])

    def test_saved_plans_need_an_account_not_strava(self):
        token = self.token_for("ana")
        response = self.client.get("/race-plans", headers=self.bearer(token))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"plans": []})
        self.assertEqual(self.client.get("/race-plans").status_code, 401)

    def test_slope_and_durability_need_strava(self):
        token = self.token_for("ana")
        for path in ("/tools/gap/summary", "/tools/durability/summary"):
            self.assertEqual(self.client.get(path, headers=self.bearer(token)).status_code, 409)

    def test_templates_replace_seeded_defaults(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 21)
        self.assertEqual(self.client.get("/pages", headers=self.bearer(token)).json(), {"pages": []})
        templates = self.client.get("/pages/templates", headers=self.bearer(token)).json()["templates"]
        self.assertIn("durability", [t["key"] for t in templates])
        created = self.client.post(
            "/pages/from-template", json={"key": "durability"}, headers=self.bearer(token)
        ).json()
        self.assertIsNone(created.get("builtin_key"))
        deleted = self.client.delete(f"/pages/{created['id']}", headers=self.bearer(token))
        self.assertEqual(deleted.status_code, 204)
