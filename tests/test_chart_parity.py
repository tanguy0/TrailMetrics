"""Both engines decide the same family plan for every figure (charts.md § v1.2).

The Python renderer reads ``src/domain/charts/families.py``; the browser reads
its twin ``web/lib/chartFamily.ts``. This runs the TypeScript under Node's type
stripping (Node ≥ 22.6) on the shared fixtures and compares the two plans field
by field — family, area, end labels, legend, widths, opacities, main series and
baseline. Skipped where Node is not installed.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from src.domain.charts.families import plan
from src.domain.charts.ir import PlotOutput
from tests.chart_fixtures import FIXTURES

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "js" / "chart_family_plans.mjs"


def _python_plan(chart) -> dict:
    decided = plan(chart)
    keyed = lambda mapping: {str(k): v for k, v in mapping.items()}  # noqa: E731 — JSON keys
    return {
        "family": decided.family,
        "area": decided.area,
        "endLabels": decided.end_labels,
        "hiddenLegend": decided.hidden_legend,
        "widths": keyed(decided.widths),
        "opacities": keyed(decided.opacities),
        "markerSizes": keyed(decided.marker_sizes),
        "main": decided.main,
        "baseline": decided.baseline,
    }


@unittest.skipUnless(shutil.which("node"), "Node is needed to run the browser twin")
class ChartFamilyParityTest(unittest.TestCase):
    def test_both_engines_plan_every_fixture_alike(self) -> None:
        charts = [PlotOutput(charts=[chart]).to_dict()["charts"][0] for _, chart, _ in FIXTURES]
        run = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", str(RUNNER)],
            input=json.dumps(charts), capture_output=True, text=True, cwd=ROOT, check=True,
        )
        browser = json.loads(run.stdout)
        self.assertEqual(len(browser), len(FIXTURES))
        for (name, chart, _), ts in zip(FIXTURES, browser):
            with self.subTest(chart=name):
                py = _python_plan(chart)
                py_base, ts_base = py.pop("baseline"), ts.pop("baseline")
                if py_base is None or ts_base is None:
                    self.assertIs(py_base, ts_base)
                else:
                    self.assertAlmostEqual(py_base, ts_base)
                self.assertEqual(py, {k: ts[k] for k in py})


if __name__ == "__main__":
    unittest.main()
