"""The Tools tab's API: level estimates, saved plans by account, analysis templates.

    TEST_DATABASE_URL=… /path/to/venv/bin/python -m unittest discover -s tests -t . -v
"""

import json
from unittest import mock

from src.domain.durability.capability import ReferenceSpeed
from src.domain.durability.config import DEFAULT_CONFIG as DURABILITY_CONFIG
from src.domain.durability.personalization import PERSONALIZED, AthleteDurabilityModel
from src.domain.durability.segments import IDENTIFIABLE, DurabilitySegment
from src.domain.gap.reference_curves import balanced_runner
from tests.api_harness import ATHLETE_BASE, DOMAIN, ApiTestCase, requires_database

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


def _durability_model() -> AthleteDurabilityModel:
    """A personalized model, as a fit on a few long runs would give."""
    population = DURABILITY_CONFIG.population
    segments = [
        DurabilitySegment(
            activity_id=1, start_date=None, elapsed_s=600.0 * k, gap_speed_m_per_s=3.2,
            heartrate_bpm=150.0 + k, intensity=0.8, exposures={"duration": 0.15 * k},
            observed_log_cost=0.003 * k,
        )
        for k in range(1, 13)
    ]
    return AthleteDurabilityModel(
        coefficients=population.with_values({"duration": 0.03}),
        population=population,
        confidence=PERSONALIZED,
        reference=ReferenceSpeed(4.0, "best_efforts"),
        personal_weight={name: 0.7 for name in IDENTIFIABLE},
        posterior_sd={name: 0.005 for name in IDENTIFIABLE},
        n_activities=1,
        n_segments=len(segments),
        segments=segments,
    )


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

    def test_estimating_never_saves(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 19)
        body = self.client.post(
            "/tools/level/estimate", json=RECORDS, headers=self.bearer(token)
        ).json()
        self.assertIsNone(body["saved_at"])
        latest = self.client.get("/tools/level/latest", headers=self.bearer(token)).json()
        self.assertIsNone(latest["estimate"])

    def test_saving_needs_strava(self):
        token = self.token_for("ana")
        response = self.client.post("/tools/level/save", json=RECORDS, headers=self.bearer(token))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.post("/tools/level/save", json=RECORDS).status_code, 401)

    def test_the_saved_estimate_becomes_the_athletes_vma(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 20)
        body = self.client.post(
            "/tools/level/save", json={**RECORDS, "hr_max": 188}, headers=self.bearer(token)
        ).json()
        self.assertIsNotNone(body["saved_at"])
        me = self.me(token).json()
        self.assertEqual(me["vma_pace_s_per_km"], body["vma_pace_s_per_km"])
        self.assertEqual(me["hr_max"], 188)
        latest = self.client.get("/tools/level/latest", headers=self.bearer(token)).json()
        self.assertEqual(latest["estimate"]["result"]["vdot"], body["vdot"])

    def test_paces_set_by_hand_until_the_next_saved_estimate(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 26)
        z2 = {"fast_s_per_km": 330, "slow_s_per_km": 360}
        me = self.client.patch(
            "/auth/me", json={"pace_overrides": {"z2": z2}}, headers=self.bearer(token)
        ).json()
        self.assertEqual(me["pace_overrides"], {"z2": z2})
        self.assertEqual(self.me(token).json()["pace_overrides"], {"z2": z2})
        for bad in ({"z9": z2}, {"z2": {"fast_s_per_km": 400, "slow_s_per_km": 300}}):
            response = self.client.patch(
                "/auth/me", json={"pace_overrides": bad}, headers=self.bearer(token)
            )
            self.assertEqual(response.status_code, 422)
        self.client.post("/tools/level/save", json=RECORDS, headers=self.bearer(token))
        self.assertEqual(self.me(token).json()["pace_overrides"], {})

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

    def _coached(self, local, athlete_id):
        token = self.token_for(local)
        self.exchange(token, athlete_id)
        request_id = self.client.put(
            "/coaching/request", json={"message": "Trail en juin"}, headers=self.bearer(token)
        ).json()["request"]["id"]
        coach = self.token_for("coach")
        self.db.execute("update accounts set role = 'coach' where email = %s", (f"coach@{DOMAIN}",))
        self.client.post(f"/coaching/requests/{request_id}/accept", headers=self.bearer(coach))
        return token

    def _save(self, token, plan_id=None, **meta):
        body = {"title": "UTMB", "params": json.loads(PLAN_PARAMS), **meta}
        if plan_id:
            return self.client.patch(
                f"/race-plans/{plan_id}", data={"meta": json.dumps(body)}, headers=self.bearer(token)
            ).json()
        return self.client.post(
            "/race-plans", data={"meta": json.dumps(body)},
            files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")},
            headers=self.bearer(token),
        ).json()

    def _goals(self, athlete_id):
        return self.db.fetch_all(
            "select id, date, title, importance from planned_items "
            "where athlete_id = %s and kind = 'goal'", (athlete_id,),
        )

    def test_a_dated_objective_goes_on_a_coached_athletes_diary(self):
        athlete = ATHLETE_BASE + 23
        token = self._coached("ana", athlete)
        saved = self._save(token, event_date="2027-08-27", importance="primary")
        self.assertEqual(saved["event_date"], "2027-08-27")
        [goal] = self._goals(athlete)
        self.assertEqual((str(goal["date"]), goal["title"], goal["importance"]),
                         ("2027-08-27", "UTMB", "primary"))
        # Edits follow the plan; clearing the date leaves the goal alone.
        self._save(token, saved["id"], title="UTMB 2027", event_date="2027-08-28",
                   importance="secondary")
        [goal] = self._goals(athlete)
        self.assertEqual((str(goal["date"]), goal["title"], goal["importance"]),
                         ("2027-08-28", "UTMB 2027", "secondary"))
        self._save(token, saved["id"], importance="secondary")
        self.assertEqual(len(self._goals(athlete)), 1)
        self.client.delete(f"/race-plans/{saved['id']}", headers=self.bearer(token))
        self.assertEqual(self._goals(athlete), [])

    def test_no_goal_without_coaching_or_without_both_fields(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 24)
        saved = self._save(token, event_date="2027-08-27", importance="primary")
        self.assertEqual(saved["importance"], "primary")
        self.assertEqual(self._goals(ATHLETE_BASE + 24), [])
        coached = self._coached("bob", ATHLETE_BASE + 25)
        self._save(coached, event_date="2027-08-27")
        self.assertEqual(self._goals(ATHLETE_BASE + 25), [])

    def test_the_gap_profile_without_runs_says_insufficient_data(self):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + 27)
        body = self.client.get("/tools/gap/summary", headers=self.bearer(token)).json()
        self.assertFalse(body["available"])
        self.assertEqual(
            [(t["key"], t["level"]) for t in body["terrains"]],
            [("steep_downhill", "insufficient"), ("downhill", "insufficient"),
             ("uphill", "insufficient"), ("steep_uphill", "insufficient")],
        )

    def test_gap_and_durability_profiles_need_strava(self):
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


@requires_database
class StoredResultsTest(ApiTestCase):
    """Profiles and saved plans are kept as last computed; Recompute refits."""

    def _athlete(self, offset):
        token = self.token_for("ana")
        self.exchange(token, ATHLETE_BASE + offset)
        return self.bearer(token)

    def test_the_gap_profile_is_kept_until_recomputed(self):
        headers = self._athlete(30)
        fit = mock.Mock(return_value=(balanced_runner(), None))
        with mock.patch("api.athlete_models.fit_personal_curve", fit), \
                mock.patch("api.athlete_models.running_activity_ids", return_value=(1, 2)) as ids:
            first = self.client.get("/tools/gap/summary", headers=headers).json()
            self.assertTrue(first["available"])
            self.assertIsNotNone(first["computed_at"])
            # A new run does not refit: it is counted, and the profile stays.
            ids.return_value = (1, 2, 3)
            again = self.client.get("/tools/gap/summary", headers=headers).json()
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(again["new_runs"], 1)
            self.assertEqual(again["terrains"], first["terrains"])
            self.assertEqual(again["chart"], first["chart"])
            fresh = self.client.post("/tools/gap/recompute", headers=headers).json()
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(fresh["new_runs"], 0)

    def test_a_fit_with_nothing_personal_is_retried_once_runs_come_in(self):
        headers = self._athlete(31)
        fit = mock.Mock(return_value=(None, "race_plan.reason.not_enough_data"))
        with mock.patch("api.athlete_models.fit_personal_curve", fit), \
                mock.patch("api.athlete_models.running_activity_ids", return_value=(1,)) as ids:
            self.assertFalse(self.client.get("/tools/gap/summary", headers=headers).json()["available"])
            self.client.get("/tools/gap/summary", headers=headers)
            self.assertEqual(fit.call_count, 1)
            ids.return_value = (1, 2)
            self.client.get("/tools/gap/summary", headers=headers)
            self.assertEqual(fit.call_count, 2)

    def test_the_durability_profile_is_kept_until_recomputed(self):
        headers = self._athlete(32)
        fit = mock.Mock(return_value=_durability_model())
        with mock.patch("api.athlete_models.fit_athlete_durability", fit):
            first = self.client.get("/tools/durability/summary", headers=headers).json()
            self.assertTrue(first["available"])
            self.assertIsNotNone(first["chart"])
            # Read back from storage: the same levels and the same chart.
            again = self.client.get("/tools/durability/summary", headers=headers).json()
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(again["qualities"], first["qualities"])
            self.assertEqual(again["chart"], first["chart"])
            self.client.post("/tools/durability/recompute", headers=headers)
            self.assertEqual(fit.call_count, 2)

    def _create(self, headers):
        return self.client.post(
            "/race-plans",
            data={"meta": f'{{"title": "Galibier", "params": {PLAN_PARAMS}}}'},
            files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")},
            headers=headers,
        ).json()

    def test_a_saved_plan_opens_on_its_stored_result(self):
        headers = self._athlete(33)
        saved = self._create(headers)
        self.assertIsNotNone(saved["computed_at"])
        self.assertTrue(saved["result"]["signed_in"])
        with mock.patch("api.routers.race_plan.PlanRace.execute") as execute:
            opened = self.client.get(f"/race-plans/{saved['id']}", headers=headers).json()
            execute.assert_not_called()
        self.assertEqual(opened["result"]["summary"], saved["result"]["summary"])
        self.assertEqual(opened["computed_at"], saved["computed_at"])
        # The list stays light: no results in it.
        [listed] = self.client.get("/race-plans", headers=headers).json()["plans"]
        self.assertNotIn("result", listed)
        # Another language plans it once more, in that language, and keeps that.
        french = self.client.get(f"/race-plans/{saved['id']}?lang=fr", headers=headers).json()
        self.assertNotEqual(french["result"]["outputs"], saved["result"]["outputs"])
        with mock.patch("api.routers.race_plan.PlanRace.execute") as execute:
            self.client.get(f"/race-plans/{saved['id']}?lang=fr", headers=headers)
            execute.assert_not_called()

    def test_a_plan_saved_before_results_were_kept_is_planned_once(self):
        headers = self._athlete(34)
        saved = self._create(headers)
        self.db.execute("update race_plans set result = null, computed_at = null "
                        "where id = %s", (saved["id"],))
        opened = self.client.get(f"/race-plans/{saved['id']}", headers=headers).json()
        self.assertIsNotNone(opened["result"])
        row = self.db.fetch_one("select result from race_plans where id = %s", (saved["id"],))
        self.assertIsNotNone(row["result"])

    def test_saving_reuses_the_models_and_recompute_refits_them(self):
        headers = self._athlete(35)
        fit = mock.Mock(return_value=(balanced_runner(), None))
        meta = {"meta": f'{{"title": "Galibier", "params": {PLAN_PARAMS}}}'}
        with mock.patch("api.athlete_models.fit_personal_curve", fit):
            saved = self._create(headers)
            self.assertTrue(saved["result"]["personalized"])
            self.client.patch(f"/race-plans/{saved['id']}", data=meta, headers=headers)
            self.assertEqual(fit.call_count, 1)
            refit = self.client.patch(
                f"/race-plans/{saved['id']}", data={**meta, "refit": "true"}, headers=headers
            ).json()
            self.assertEqual(fit.call_count, 2)
            self.assertIsNotNone(refit["result"])
            # A plan not yet saved recomputes the same way, and stores nothing.
            self.client.post(
                "/race-plan", data={"params": PLAN_PARAMS, "refit": "true"},
                files={"gpx": ("course.gpx", _gpx(), "application/gpx+xml")}, headers=headers,
            )
            self.assertEqual(fit.call_count, 3)
