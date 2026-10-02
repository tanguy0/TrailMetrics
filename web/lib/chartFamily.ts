/**
 * Which family a figure belongs to, and what that means for each trace — the
 * browser twin of `src/domain/charts/families.py` (design/tagg/charts.md § v1.2).
 *
 * Kept free of imports other than types, and handed its two identifying colours,
 * so `tests/test_chart_parity.py` can run it under plain Node and check that
 * both engines decide the same plan for the same figure.
 */

import type { ChartData, ChartFamily, Trace } from "./types";

export const FAMILIES: readonly ChartFamily[] = [
  "tracking", "comparison", "function", "oscillation", "composition", "scatter",
];

/** Mirrors `ZERO_REACH`: only the fallback now — the declared aggregation decides first. */
export const ZERO_REACH = 0.5;

// Stroke weights of charts.md § v1.2, by family. Mirror families.py.
const TRACKING_WIDTH = 2.2;
const FUNCTION_WIDTH = 2.2;
const OSCILLATION_WIDTH = 2.0;
const CURRENT_WIDTH = 2.4;
const OTHER_WIDTH = 1.5;
const OTHER_OPACITY = 0.7;
const CROWD = 3;
const CROWD_OPACITY = 0.55;
const SCATTER_OPACITY = 0.6;
const SCATTER_MARKER_SIZE = 8; // r 4
const SCATTER_TREND_WIDTH = 2.0;

/** The two colours that say what a trace *is*, plus the fallback cycle. */
export interface FamilyPalette {
  reference: string;
  series1: string;
  cycle: readonly string[];
}

export interface Plan {
  family: ChartFamily;
  area: number | null;
  endLabels: number[];
  hiddenLegend: number[];
  widths: Record<number, number>;
  opacities: Record<number, number>;
  markerSizes: Record<number, number>;
  main: number | null;
  baseline: number | null;
}

const hasPoints = (trace: Trace) => trace.y.some((v) => v != null);
const isLine = (trace: Trace) => trace.kind === "line" || trace.kind === "step";
const finite = (values: (number | null)[] | null | undefined): number[] =>
  (values ?? []).filter((v): v is number => v != null && Number.isFinite(v));

function colorOf(chart: ChartData, index: number, palette: FamilyPalette): string {
  return chart.traces[index].color || palette.cycle[index % palette.cycle.length];
}

const isReference = (color: string, palette: FamilyPalette) =>
  color.toLowerCase() === palette.reference.toLowerCase();

function onLeft(chart: ChartData, trace: Trace): boolean {
  return trace.axis !== "y2" || !chart.y2_axis;
}

/** The athlete's own series: not a reference, not a backdrop, with data. */
function athlete(chart: ChartData, palette: FamilyPalette): number[] {
  return chart.traces
    .map((trace, index) => ({ trace, index }))
    .filter(({ trace, index }) =>
      hasPoints(trace) && !trace.background && !isReference(colorOf(chart, index, palette), palette))
    .map(({ index }) => index);
}

/** Mirrors `classify`: declared, else read off the figure's shape. */
export function classify(chart: ChartData, palette: FamilyPalette): ChartFamily {
  if (chart.family && FAMILIES.includes(chart.family)) return chart.family;
  const own = athlete(chart, palette).map((i) => chart.traces[i]);
  if (own.some((t) => t.stack_group || t.kind === "area" || t.kind === "bar")) return "composition";
  if (own.some((t) => t.kind === "scatter")) return "scatter";
  if (chart.x_axis.kind !== "date") return "function";
  const left = own.filter((t) => isLine(t) && onLeft(chart, t));
  if (left.length >= 2) return "comparison";
  if (!left.length) return "function";
  const values = finite(left[0].y);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || Math.abs(hi) || 1;
  if (chart.y_axis.kind !== "linear" || (lo < 0 && hi > 0) || lo > ZERO_REACH * span) {
    return "oscillation";
  }
  return "tracking";
}

function lastX(trace: Trace): string {
  for (let i = trace.y.length - 1; i >= 0; i--) {
    if (trace.y[i] != null) return String(trace.x[i]);
  }
  return "";
}

/** Mirrors `_current`: declared, else the latest period, else series 1. */
function current(chart: ChartData, lines: number[], palette: FamilyPalette): number | null {
  const declared = lines.find((i) => chart.traces[i].end_label === true);
  if (declared !== undefined) return declared;
  if (chart.x_axis.kind === "date" && lines.length) {
    // ISO strings order chronologically; ties go to the first series.
    return lines.reduce((best, i) => (lastX(chart.traces[i]) > lastX(chart.traces[best]) ? i : best));
  }
  const first = lines.find((i) => colorOf(chart, i, palette).toLowerCase() === palette.series1.toLowerCase());
  return first ?? lines[0] ?? null;
}

function canFill(chart: ChartData, trace: Trace): boolean {
  return (
    !chart.y2_axis &&
    chart.y_axis.kind === "linear" &&
    !chart.y_axis.reversed &&
    isLine(trace) &&
    !trace.band_upper &&
    !trace.stack_group
  );
}

function baselineOf(chart: ChartData, lines: number[]): number | null {
  if (chart.baseline != null) return chart.baseline;
  const values = lines.flatMap((i) => finite(chart.traces[i].y));
  if (!values.length || (chart.y_axis.kind !== "linear" && chart.y_axis.kind !== "duration")) return null;
  if (Math.min(...values) < 0 && Math.max(...values) > 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Mirrors `plan`. */
export function planFor(chart: ChartData, palette: FamilyPalette): Plan {
  const family = classify(chart, palette);
  const declared = Boolean(chart.family && FAMILIES.includes(chart.family));
  const plan: Plan = {
    family, area: null, endLabels: [], hiddenLegend: [],
    widths: {}, opacities: {}, markerSizes: {}, main: null, baseline: null,
  };
  const own = athlete(chart, palette);
  const lines = own.filter((i) => isLine(chart.traces[i]) && onLeft(chart, chart.traces[i]));
  const dual = Boolean(chart.y2_axis);

  let labelled: number[] = [];
  if (family === "tracking") {
    const candidates = lines.filter((i) => chart.traces[i].area !== false);
    if (candidates.length && canFill(chart, chart.traces[candidates[0]])) plan.area = candidates[0];
    labelled = lines;
    if (!declared) lines.forEach((i) => (plan.widths[i] = TRACKING_WIDTH));
  } else if (family === "comparison") {
    const now = current(chart, lines, palette);
    const others = lines.filter((i) => i !== now);
    if (now !== null) {
      plan.widths[now] = CURRENT_WIDTH;
      labelled = [now];
    }
    const fade = others.length > CROWD ? CROWD_OPACITY : OTHER_OPACITY;
    others.forEach((i) => {
      plan.widths[i] = OTHER_WIDTH;
      plan.opacities[i] = fade;
    });
    plan.main = now;
  } else if (family === "function") {
    if (!declared) lines.forEach((i) => (plan.widths[i] = FUNCTION_WIDTH));
  } else if (family === "oscillation") {
    labelled = lines;
    if (!declared) lines.forEach((i) => (plan.widths[i] = OSCILLATION_WIDTH));
    plan.baseline = baselineOf(chart, lines);
  } else if (family === "scatter") {
    own.forEach((i) => {
      if (chart.traces[i].kind === "scatter") {
        plan.opacities[i] = SCATTER_OPACITY;
        plan.markerSizes[i] = SCATTER_MARKER_SIZE;
      } else if (isLine(chart.traces[i])) {
        plan.widths[i] = SCATTER_TREND_WIDTH;
      }
    });
  }

  // A declared area (the fatigue) outranks the family's own pick.
  const declaredArea = lines.find((i) => chart.traces[i].area === true && canFill(chart, chart.traces[i]));
  if (declaredArea !== undefined) plan.area = declaredArea;

  if (family !== "comparison") {
    labelled = labelled.filter((i) => chart.traces[i].end_label !== false);
    labelled.push(...lines.filter((i) => chart.traces[i].end_label === true && !labelled.includes(i)));
  }
  plan.endLabels = dual ? [] : [...labelled].sort((a, b) => a - b);

  if ((family === "tracking" || family === "oscillation") && lines.length === 1 && plan.endLabels.length) {
    plan.hiddenLegend = [...plan.endLabels];
  }
  if (plan.main === null) plan.main = plan.area ?? lines[0] ?? null;
  return plan;
}
