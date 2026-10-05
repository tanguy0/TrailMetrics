"""The app offers one GAP model: the efficiency one, shown as "GAP".

The auto-learning model stays in the code but must not surface anywhere a user
can pick or see it.
"""

import unittest

from src.domain.plots import gap_curve
from src.usecases.plan_race import (
    PERSONAL_AUTO,
    PERSONAL_EFFICIENCY,
    PlanRace,
    PlanRaceInput,
    curve_options,
)
from src.translations import translate


class GapModelsOfferedTest(unittest.TestCase):
    def test_the_panel_form_offers_no_model_choice(self):
        keys = {spec.key for spec in gap_curve.PARAMS}
        self.assertNotIn("models", keys)
        self.assertFalse(keys & {spec.key for spec in gap_curve.AUTO_LEARNING_PARAMS})
        self.assertEqual(gap_curve.APP_MODELS, (gap_curve.EFFICIENCY,))

    def test_the_race_plan_offers_one_personal_curve(self):
        options = curve_options(True, "fr")
        self.assertNotIn(PERSONAL_AUTO, [o["key"] for o in options])
        self.assertEqual(translate("race_plan.curve.personal_efficiency", "fr"), "Ma courbe GAP")
        self.assertEqual(translate("gap.models.efficiency", "en"), "GAP")

    def test_a_plan_saved_on_the_auto_curve_uses_the_gap_curve(self):
        asked = []

        def personal(key):
            asked.append(key)
            return None, "race_plan.reason.no_runs"

        try:
            PlanRace(personal_curve=personal).execute(
                PlanRaceInput(gpx=b"<gpx/>", target_time_s=3600, curve=PERSONAL_AUTO)
            )
        except Exception:
            pass  # the GPX is empty; only which curve was asked for matters here

        self.assertEqual(asked, [PERSONAL_EFFICIENCY])


if __name__ == "__main__":
    unittest.main()
