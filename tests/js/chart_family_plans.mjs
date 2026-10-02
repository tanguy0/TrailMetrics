// Reads chart IR (JSON array) on stdin, prints each chart's family plan as the
// browser decides it — web/lib/chartFamily.ts, run under Node's type stripping.
// Driven by tests/test_chart_parity.py.
import { planFor } from "../../web/lib/chartFamily.ts";
import { curvePalette, tokens } from "../../web/lib/theme.ts";

const palette = { reference: tokens["chart-ref"], series1: tokens["chart-you-1"], cycle: curvePalette };
let input = "";
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const charts = JSON.parse(input);
  process.stdout.write(JSON.stringify(charts.map((chart) => planFor(chart, palette))));
});
