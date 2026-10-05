"""The Tools tab's API: level estimates, saved plans by account, analysis templates.

    TEST_DATABASE_URL=… /path/to/venv/bin/python -m unittest discover -s tests -t . -v
"""

from tests.api_harness import ATHLETE_BASE, ApiTestCase, requires_database

def _gpx(points=60) -> bytes:
    """A 6 km out-and-up line: enough for the planner, small enough to inline."""
    rows = "".join(
        f'<trkpt lat="45.{i:04d}" lon="6.0000"><ele>{1000 + 5 * i}</ele></trkpt>'
        for i in range(points)
    )
    return (
        '<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><trkseg>{rows}</trkseg></trk></gpx>"
    ).encode()


PLAN_PARAMS = '{"target_time_s": 3600}'

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

    def test_a_saved_plan_is_planned_from_its_stored_gpx(self):
        # Regression: planning by ``plan_id`` used to 500 (``account`` undefined).
        token = self.token_for("ana")
        saved = self.client.post(
            "/race-plans",
            data={"meta": f'{{"title": "Galibier", "params": {PLAN_PARAMS}}}'},
            files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")},
            headers=self.bearer(token),
        )
        self.assertEqual(saved.status_code, 201, saved.text)
        planned = self.client.post(
            "/race-plan",
            data={"params": PLAN_PARAMS, "plan_id": saved.json()["id"]},
            headers=self.bearer(token),
        )
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertFalse(planned.json()["signed_in"])
        visitor = self.client.post(
            "/race-plan", data={"params": PLAN_PARAMS, "plan_id": saved.json()["id"]}
        )
        self.assertEqual(visitor.status_code, 400)

    def test_saved_plans_list_a_thumbnail_and_backfill_old_ones(self):
        token = self.token_for("ana")
        saved = self.client.post(
            "/race-plans",
            data={"meta": f'{{"title": "Galibier", "params": {PLAN_PARAMS}}}'},
            files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")},
            headers=self.bearer(token),
        ).json()
        preview = saved["preview"]
        self.assertEqual(len(preview["route"]), 60)
        self.assertLessEqual(len(preview["profile"]), 120)
        self.assertGreater(preview["profile"][-1][1], preview["profile"][0][1])
        # A plan saved before thumbnails existed gets one from the list.
        self.db.execute("update race_plans set preview = null where id = %s", (saved["id"],))
        listed = self.client.get("/race-plans", headers=self.bearer(token)).json()["plans"]
        self.assertEqual(listed[0]["preview"], preview)
        row = self.db.fetch_one("select preview from race_plans where id = %s", (saved["id"],))
        self.assertIsNotNone(row["preview"])

    def test_a_saved_plan_is_planned_with_strava_attached(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 22)
        saved = self.client.post(
            "/race-plans",
            data={"meta": f'{{"title": "Galibier", "params": {PLAN_PARAMS}}}'},
            files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")},
            headers=self.bearer(token),
        ).json()
        planned = self.client.post(
            "/race-plan",
            data={"params": PLAN_PARAMS, "plan_id": saved["id"]},
            headers=self.bearer(token),
        )
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertTrue(planned.json()["signed_in"])

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
