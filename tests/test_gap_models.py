"""The app offers one GAP model: the efficiency one, shown as "GAP".

The auto-learning model stays in the code but must not surface anywhere a user
can pick or see it. And nobody picks a GAP curve: a race plan is paced on the
athlete's own; the references are comparisons only.
"""

import unittest

from src.domain.plots import gap_curve
from src.usecases.plan_race import (
    BALANCED_RUNNER,
    PERSONAL_EFFICIENCY,
    PlanRace,
    PlanRaceInput,
)
from src.translations import translate


class GapModelsOfferedTest(unittest.TestCase):
    def test_the_panel_form_offers_no_model_choice(self):
        keys = {spec.key for spec in gap_curve.PARAMS}
        self.assertNotIn("models", keys)
        self.assertFalse(keys & {spec.key for spec in gap_curve.AUTO_LEARNING_PARAMS})
        self.assertEqual(gap_curve.APP_MODELS, (gap_curve.EFFICIENCY,))

    def test_the_race_plan_takes_no_curve_choice(self):
        self.assertNotIn("curve", PlanRaceInput.__dataclass_fields__)
        self.assertEqual(translate("race_plan.curve.personal_efficiency", "fr"), "Ma courbe GAP")
        self.assertEqual(translate("gap.models.efficiency", "en"), "GAP")

    def test_a_plan_is_paced_on_the_athletes_own_curve(self):
        asked = []

        def personal(key):
            asked.append(key)
            return None, "race_plan.reason.no_runs"

        try:
            PlanRace(personal_curve=personal).execute(
                PlanRaceInput(gpx=b"<gpx/>", target_time_s=3600)
            )
        except Exception:
            pass  # the GPX is empty; only which curve was asked for matters here

        self.assertEqual(asked, [PERSONAL_EFFICIENCY])

    def test_without_a_personal_curve_the_balanced_runner_stands_in(self):
        notes = []
        _, key = PlanRace()._curve("fr", notes)
        self.assertEqual(key, BALANCED_RUNNER)
        self.assertEqual(notes, [])

        _, key = PlanRace(personal_curve=lambda k: (None, None))._curve("fr", notes)
        self.assertEqual(key, BALANCED_RUNNER)
        self.assertEqual(len(notes), 1)


if __name__ == "__main__":
    unittest.main()
