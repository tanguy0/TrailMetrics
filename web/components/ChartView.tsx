"use client";

/**
 * Chart IR → Plotly figure.
 *
 * The browser twin of `src/domain/charts/plotly.py`: the same IR, the same theme,
 * the same axis quirks. A plot definition describes data once and gets a figure in
 * both places, which is why adding a plot type needs no frontend work at all.
 *
 * Plotly is loaded lazily on first render — it is a large bundle, and a page of
 * tables shouldn't pay for it.
 */

import { useEffect, useRef, useState } from "react";

import { Callout } from "@/components/Callout";
import { durationToEpoch, formatHms, toCsv, downloadCsv } from "@/lib/format";
import { planFor, type FamilyPalette, type Plan } from "@/lib/chartFamily";
import { AREA_ALPHA_TOP, curvePalette, dashByCode, isReference, rgba, theme, tokens } from "@/lib/theme";
import type { Axis, ChartData, Trace } from "@/lib/types";

// Resolved once per session; `plotly.js-dist-min` has no types of its own.
type Plotly = typeof import("plotly.js-dist-min").default;
let plotlyPromise: Promise<Plotly> | null = null;
function loadPlotly(): Promise<Plotly> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m) => m.default ?? m);
  return plotlyPromise;
}

const BAND_ALPHA = 0.16;

// Mirrors `_REF_WIDTH` in src/domain/charts/plotly.py.
const REF_WIDTH = 1.5;

// Mirrors `MARGIN` in src/domain/plotting_common.py: no title in the figure, the
// card carries it.
const MARGIN = { l: 44, r: 16, t: 16, b: 32 };

const AXIS_FONT = { family: theme.fontMono, size: 11, color: theme.chartAxis };

// --- v1.1 (charts.md § Plus de caractère); each mirrors src/domain/charts/plotly.py.
const BAR_RADIUS = 4;
const END_MARKER_SIZE = 8; // r 4
const END_LABEL_SHIFT = 9;
const END_LABEL_FONT_SIZE = 11;
const END_LABEL_MARGIN_R = 52;
const BAR_LABEL_FONT_SIZE = 11;
const MARKER_DASH = "2px,4px";
const RACE_DOT_PX = 4;

/** Mirrors `SPIKES` in src/domain/plotting_common.py. */
const SPIKES = {
  showspikes: true,
  spikecolor: theme.lineStrong,
  spikedash: "2px,4px",
  spikethickness: 1,
  spikemode: "across",
  spikesnap: "cursor",
};
const Y_NTICKS = 5;

const hasPoints = (trace: Trace) => trace.y.some((v) => v != null);

/** What identifies a trace for the family plan (lib/chartFamily.ts). */
const PALETTE: FamilyPalette = {
  reference: tokens["chart-ref"],
  series1: tokens["chart-you-1"],
  cycle: curvePalette,
};

// Mirror `_RANGE_PAD`, `_BACKGROUND_ALPHA`, `_BASELINE_WIDTH` in src/domain/charts/plotly.py.
const RANGE_PAD = 0.05;
const BACKGROUND_ALPHA = 0.35;
const BASELINE_WIDTH = 1;
// Mirror `_BACKGROUND_PAD_*` / `_BACKGROUND_AXIS`: an axis of backdrops only
// frames their relief; the hidden third axis takes one when both are taken.
const BACKGROUND_PAD_LOW = 0.15;
const BACKGROUND_PAD_HIGH = 0.3;
const BACKGROUND_AXIS = "y3";
const HIDDEN_AXIS: Axis = {
  title: "", kind: "linear", reversed: false, tick_format: null, suffix: null,
  range: null, dtick: null, color: null,
};

/** Mirrors `background_range`. */
function backgroundRange(values: (number | null)[]): number[] | null {
  const data = finite(values);
  if (!data.length) return null;
  const lo = Math.min(...data);
  const hi = Math.max(...data);
  const span = hi - lo || 1;
  return [lo - BACKGROUND_PAD_LOW * span, hi + BACKGROUND_PAD_HIGH * span];
}

/** Mirrors `_frame_backgrounds`: the range of each axis holding backdrops only. */
function backgroundFrames(chart: ChartData): Partial<Record<"y" | "y2" | "y3", number[]>> {
  const axisOf = (t: Trace) => (t.axis === "y2" && chart.y2_axis ? "y2" : t.axis === BACKGROUND_AXIS ? "y3" : "y");
  const frames: Partial<Record<"y" | "y2" | "y3", number[]>> = {};
  const explicit = { y: chart.y_axis.range, y2: chart.y2_axis?.range ?? null, y3: null };
  (["y", "y2", "y3"] as const).forEach((name) => {
    const onAxis = chart.traces.filter((t) => axisOf(t) === name);
    if (!onAxis.length || !onAxis.every((t) => t.background) || explicit[name]) return;
    const framed = backgroundRange(onAxis.flatMap((t) => t.y));
    if (framed) frames[name] = framed;
  });
  return frames;
}

const finite = (values: (number | null)[] | null | undefined): number[] =>
  (values ?? []).filter((v): v is number => v != null && Number.isFinite(v));

/**
 * Mirrors `area_y_range`: the left axis's range when the figure carries an
 * area, as `[lo, hi, dataMax]`. An area is a quantity that accumulates, so it
 * starts at zero; set explicitly so the gradient fades over what is visible.
 */
function areaYRange(chart: ChartData): [number, number, number] | null {
  const primary = chart.traces.filter((t) => (t.axis !== "y2" || !chart.y2_axis) && !t.background);
  const values = primary.flatMap((t) => [...finite(t.y), ...finite(t.band_upper), ...finite(t.band_lower)]);
  if (!values.length) return null;
  const dataMax = Math.max(...values);
  if (chart.y_axis.range) return [chart.y_axis.range[0], chart.y_axis.range[1], dataMax];
  const lo = Math.min(Math.min(...values), 0);
  const hi = Math.max(dataMax, 0);
  const pad = RANGE_PAD * (hi - lo || 1);
  return [lo === 0 ? lo : lo - pad, hi === 0 ? hi : hi + pad, dataMax];
}

/** The family plan and the area's range — what every part of the figure reads. */
function roles(chart: ChartData): { plan: Plan; areaRange: [number, number, number] | null } {
  const plan = planFor(chart, PALETTE);
  return { plan, areaRange: plan.area !== null ? areaYRange(chart) : null };
}

/** Mirrors `resolve_hover_mode`: unified unless the chart is a scatter. */
function hoverMode(chart: ChartData): string {
  if (chart.hover_mode !== "auto" && chart.hover_mode !== "closest") return chart.hover_mode;
  const plotted = chart.traces.filter(hasPoints);
  return plotted.length && plotted.every((t) => t.kind === "scatter") ? "closest" : "x unified";
}

/** Mirrors `end_label_text`: `4:21`, `68`, `1.42`. */
function endLabelText(value: number, axis: Axis): string {
  if (axis.kind === "duration") return formatHms(value);
  const magnitude = Math.abs(value);
  const decimals = magnitude >= 10 ? 0 : magnitude >= 1 ? 1 : 2;
  return (
    value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) +
    (axis.suffix ?? "")
  );
}

function lastPoint(trace: Trace): [number | string | null, number] | null {
  for (let i = trace.y.length - 1; i >= 0; i--) {
    const y = trace.y[i];
    if (y != null && !Number.isNaN(y)) return [trace.x[i], y];
  }
  return null;
}

/**
 * The y-axis title moves into the card's sub-line (charts.md § v1.1: no axis
 * title when the card carries the unit). A dual-axis chart keeps both titles —
 * tinted to their series, they are what says which line each scale measures.
 */
function unitInCard(chart: ChartData): string | null {
  return !chart.y2_axis && chart.y_axis.title ? chart.y_axis.title : null;
}

/** Mirrors `axis_style` in src/domain/plotting_common.py. */
function axisStyle(grid: boolean): Record<string, unknown> {
  return {
    showgrid: grid,
    gridcolor: theme.chartGrid,
    gridwidth: 1,
    zeroline: false,
    showline: !grid,
    linecolor: theme.line,
    ticks: "",
    tickfont: AXIS_FONT,
    automargin: true,
  };
}

// Where the badge row sits, as a share of the plot's height (1 = the very top).
// Inside the frame: above it, the row would fight the title and legend for the
// same strip of margin. Mirrors `_BADGE_ROW_Y` in src/domain/charts/plotly.py.
const BADGE_ROW_Y = 0.98;
const BADGE_FONT_SIZE = 9;
// Tight: a 30-week window leaves each badge ~20px of x to sit in.
const BADGE_PADDING = 1;
// Pixels a badge needs before its full wording fits rather than its `short` form.
// Mirrors `_MIN_FULL_BADGE_PX` in src/domain/charts/plotly.py — which has to
// assume a width, where this side can measure the container it was given.
const MIN_FULL_BADGE_PX = 62;

/** Map IR values onto what Plotly needs for this axis kind. */
function encode(values: (number | string | null)[], axis: Axis): unknown[] {
  if (axis.kind === "duration") {
    return values.map((v) => durationToEpoch(v == null ? null : Number(v)));
  }
  return values;
}

function axisLayout(axis: Axis, grid: boolean): Record<string, unknown> {
  const layout: Record<string, unknown> = {
    ...axisStyle(grid),
    title: { text: axis.title, font: AXIS_FONT },
  };
  if (axis.kind === "duration") {
    layout.type = "date";
    layout.tickformat = axis.tick_format || "%M:%S";
  } else if (axis.kind === "date") {
    layout.type = "date";
    if (axis.tick_format) layout.tickformat = axis.tick_format;
  } else if (axis.kind === "category") {
    layout.type = "category";
  } else if (axis.tick_format) {
    layout.tickformat = axis.tick_format;
  }

  // `reversed` and an explicit range are mutually exclusive in Plotly.
  if (axis.reversed) layout.autorange = "reversed";
  else if (axis.range) layout.range = axis.range;

  if (axis.suffix) layout.ticksuffix = axis.suffix;
  if (axis.dtick != null) layout.dtick = axis.dtick;
  if (axis.tick_values?.length && axis.tick_labels?.length) {
    layout.tickmode = "array";
    layout.tickvals = axis.tick_values;
    layout.ticktext = axis.tick_labels;
  }
  // Tints the axis to its series, so a dual-axis chart says which line it measures.
  if (axis.color) {
    layout.title = { text: axis.title, font: { ...AXIS_FONT, color: axis.color } };
    layout.tickfont = { ...AXIS_FONT, color: axis.color };
  }
  return layout;
}

function toPlotlyTraces(chart: ChartData): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  const { plan, areaRange } = roles(chart);
  // A backdrop joins the legend only when it is the figure's only series.
  const lonely = !chart.traces.some((t) => hasPoints(t) && !t.background);

  // Backdrops first, then references, so both sit underneath the athlete's lines;
  // `legendrank` keeps the legend in the chart's own order (as in the Python renderer).
  const ordered = chart.traces
    .map((trace, index) => ({
      trace,
      index,
      color: trace.color || curvePalette[index % curvePalette.length],
    }))
    .sort(
      (a, b) =>
        Number(Boolean(b.trace.background)) - Number(Boolean(a.trace.background)) ||
        Number(isReference(b.color)) - Number(isReference(a.color)),
    );

  ordered.forEach(({ trace, index, color }) => {
    // A trace's values are encoded against the axis it is actually measured on.
    const onSecondary = trace.axis === "y2" && Boolean(chart.y2_axis);
    const onHidden = trace.axis === BACKGROUND_AXIS;
    const yAxis = onSecondary ? chart.y2_axis! : onHidden ? HIDDEN_AXIS : chart.y_axis;
    const x = encode(trace.x, chart.x_axis);
    const y = encode(trace.y, yAxis);

    if (trace.background) {
      // A flat line-strong fill, no stroke — never "the area" (charts.md § v1.2).
      out.push({
        x, y, type: "scatter", mode: "lines", name: trace.name,
        line: { width: 0, color: theme.lineStrong },
        fill: "tozeroy",
        fillcolor: rgba(theme.lineStrong, BACKGROUND_ALPHA),
        showlegend: lonely && trace.show_legend,
        legendgroup: trace.legend_group || trace.name,
        ...(trace.hover_text ? { customdata: trace.hover_text } : {}),
        ...(trace.hover_template ? { hovertemplate: trace.hover_template } : {}),
        ...(onSecondary ? { yaxis: "y2" } : onHidden ? { yaxis: BACKGROUND_AXIS } : {}),
      });
      return;
    }

    // The ±band goes first so the line draws on top of its own ribbon.
    if (trace.band_upper && trace.band_lower) {
      out.push({
        x: [...encode(trace.x, chart.x_axis), ...encode([...trace.x].reverse(), chart.x_axis)],
        y: [
          ...encode(trace.band_upper, yAxis),
          ...encode([...trace.band_lower].reverse(), yAxis),
        ],
        type: "scatter",
        fill: "toself",
        fillcolor: rgba(color, trace.band_opacity ?? BAND_ALPHA),
        line: { width: 0 },
        hoverinfo: "skip",
        showlegend: false,
        legendgroup: trace.legend_group || trace.name,
        name: trace.name,
        ...(onSecondary ? { yaxis: "y2" } : {}),
      });
    }

    const common: Record<string, unknown> = {
      x,
      y,
      name: trace.name,
      legendgroup: trace.legend_group || trace.name,
      showlegend: trace.show_legend && !plan.hiddenLegend.includes(index),
      legendrank: index + 1,
      opacity: plan.opacities[index] ?? trace.opacity,
      ...(onSecondary ? { yaxis: "y2" } : {}),
    };
    if (trace.hover_text) common.customdata = trace.hover_text;
    if (trace.hover_template) common.hovertemplate = trace.hover_template;
    else if (index === plan.main && yAxis.kind === "linear") {
      // The main series leads the unified hover, its value bold in sun-ink.
      common.hovertemplate =
        `%{fullData.name} : <b><span style='color:${theme.sunInk}'>%{y}</span></b><extra></extra>`;
    }

    if (trace.kind === "bar") {
      const marker: Record<string, unknown> = {
        color: trace.point_colors ?? color,
        cornerradius: BAR_RADIUS,
      };
      if (trace.point_opacity) marker.opacity = trace.point_opacity;
      out.push({
        ...common,
        type: "bar",
        marker,
        ...(trace.point_widths ? { width: trace.point_widths } : {}),
        ...(trace.bar_base != null ? basedBars(trace, yAxis) : {}),
        ...(trace.point_text
          ? {
              text: trace.point_text,
              textposition: "outside",
              cliponaxis: false,
              textfont: { family: theme.fontMono, size: BAR_LABEL_FONT_SIZE, color: theme.inkMuted },
            }
          : {}),
      });
      return;
    }

    const planned = plan.widths[index] ?? trace.width;
    const width = isReference(color) ? Math.min(planned, REF_WIDTH) : planned;
    const line: Record<string, unknown> = { color, width };
    const dash = dashByCode[trace.dash] ?? "solid";
    if (dash !== "solid") line.dash = dash;
    if (trace.kind === "step") line.shape = "hv";

    const scatter: Record<string, unknown> = { ...common, type: "scatter", line };
    const size = plan.markerSizes[index] ?? trace.marker_size;
    const markerColor = trace.point_colors ?? color;
    if (trace.kind === "scatter") {
      scatter.mode = "markers";
      scatter.marker = { color: markerColor, size };
    } else {
      scatter.mode = trace.markers ? "lines+markers" : "lines";
      if (trace.markers) scatter.marker = { color: markerColor, size };
    }
    if (trace.kind === "area") {
      scatter.stackgroup = trace.stack_group || "area";
      scatter.fillcolor = rgba(color, trace.stack_group ? 0.35 : 0.2);
      scatter.line = { color, width: 0.35 };
    } else if (index === plan.area && areaRange) {
      // The figure's one area (tracking, or the declared fatigue): AREA_ALPHA_TOP
      // at the data's top, fading to nothing at the bottom of the visible axis.
      scatter.fill = "tozeroy";
      scatter.fillgradient = {
        type: "vertical",
        start: areaRange[0],
        stop: areaRange[2],
        colorscale: [
          [0, rgba(color, 0)],
          [1, rgba(color, AREA_ALPHA_TOP)],
        ],
      };
    }
    out.push(scatter);

    // The end-of-line dot; its value label is an annotation (see endAnnotations).
    const last = plan.endLabels.includes(index) ? lastPoint(trace) : null;
    if (last) {
      out.push({
        x: encode([last[0]], chart.x_axis),
        y: encode([last[1]], yAxis),
        type: "scatter",
        mode: "markers",
        marker: { color, size: END_MARKER_SIZE },
        hoverinfo: "skip",
        showlegend: false,
        cliponaxis: false,
        legendgroup: trace.legend_group || trace.name,
      });
    }
  });

  return out;
}

/** Mirrors `_based_bars`: bars from `bar_base`, their `y` read as a length (ms on a duration axis). */
function basedBars(trace: Trace, axis: Axis): Record<string, unknown> {
  const base = trace.bar_base!;
  const scale = axis.kind === "duration" ? 1000 : 1;
  return {
    y: trace.y.map((v) => (v == null ? null : (v - base) * scale)),
    base: encode([base], axis)[0],
  };
}

/** Today as a dotted sun line; a race or aid station as a terra dot; a boundary as a thin rule. */
function markerShapes(chart: ChartData): Record<string, unknown>[] {
  return (chart.markers ?? []).flatMap((marker): Record<string, unknown>[] => {
    const x = encode([marker.x], chart.x_axis)[0];
    if (marker.kind === "today") {
      return [{
        type: "line", xref: "x", yref: "y domain", x0: x, x1: x, y0: 0, y1: 1,
        line: { color: theme.todayMarker, width: 1, dash: MARKER_DASH },
      }];
    }
    if (marker.kind === "boundary") {
      return [{
        type: "line", xref: "x", yref: "y domain", x0: x, x1: x, y0: 0, y1: 1,
        line: { color: theme.line, width: 1 }, layer: "below",
      }];
    }
    if (marker.kind === "race" || marker.kind === "aid") {
      return [{
        type: "circle", xref: "x", yref: "y domain",
        xsizemode: "pixel", ysizemode: "pixel", xanchor: x, yanchor: 0,
        x0: -RACE_DOT_PX, x1: RACE_DOT_PX, y0: 0, y1: 2 * RACE_DOT_PX,
        fillcolor: theme.raceMarker, line: { width: 0 },
      }];
    }
    return [];
  });
}

/** The markers' mono labels: above the plot for today, above the dot for a race. */
function markerAnnotations(chart: ChartData): Record<string, unknown>[] {
  return (chart.markers ?? []).filter((marker) => marker.kind !== "boundary").map((marker) => ({
    x: encode([marker.x], chart.x_axis)[0],
    xref: "x",
    y: marker.kind === "today" ? 1 : 0,
    yref: "y domain",
    yanchor: "bottom",
    yshift: marker.kind === "today" ? 0 : 2 * RACE_DOT_PX + 2,
    text: marker.label,
    showarrow: false,
    font: {
      family: theme.fontMono,
      size: END_LABEL_FONT_SIZE,
      color: marker.kind === "today" ? theme.sunInk : theme.raceMarker,
    },
  }));
}

/** An oscillation's reference level, a line-strong rule across the plot. */
function baselineShapes(chart: ChartData, plan: Plan): Record<string, unknown>[] {
  if (plan.baseline === null) return [];
  const y = encode([plan.baseline], chart.y_axis)[0];
  return [{
    type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: y, y1: y,
    line: { color: theme.lineStrong, width: BASELINE_WIDTH }, layer: "below",
  }];
}

/** Each labelled line's last value, right of its end dot, in its colour. */
function endAnnotations(chart: ChartData, plan: Plan): Record<string, unknown>[] {
  return plan.endLabels.flatMap((index) => {
    const trace = chart.traces[index];
    const last = lastPoint(trace);
    if (!last) return [];
    const yAxis = trace.axis === "y2" && chart.y2_axis ? chart.y2_axis : chart.y_axis;
    const color = trace.color || curvePalette[index % curvePalette.length];
    return [{
      x: encode([last[0]], chart.x_axis)[0],
      y: encode([last[1]], yAxis)[0],
      xref: "x",
      yref: "y",
      text: endLabelText(last[1], yAxis),
      showarrow: false,
      xanchor: "left",
      xshift: END_LABEL_SHIFT,
      font: { family: theme.fontMono, size: END_LABEL_FONT_SIZE, color },
    }];
  });
}

/** Bands as full-height rectangles behind the traces. */
function toShapes(chart: ChartData): Record<string, unknown>[] {
  return (chart.bands ?? []).map((band) => {
    const [x0, x1] = encode([band.x0, band.x1], chart.x_axis);
    return {
      type: "rect",
      xref: "x",
      yref: "y domain",
      x0,
      x1,
      y0: 0,
      y1: 1,
      fillcolor: rgba(band.color, band.opacity),
      line: { width: 0 },
      layer: "below",
    };
  });
}

/**
 * Badges as a row of bordered annotations just inside the top of the plot.
 *
 * `width` is the figure's measured width: with too little room per badge the row
 * falls back to each badge's `short` form, since Plotly draws every annotation
 * whether or not they overlap.
 */
function toAnnotations(chart: ChartData, width: number): Record<string, unknown>[] {
  const badges = chart.badges ?? [];
  const room = width / Math.max(badges.length, 1);
  return badges.map((badge) => ({
    x: encode([badge.x], chart.x_axis)[0],
    xref: "x",
    y: BADGE_ROW_Y,
    yref: "y domain",
    yanchor: "top",
    text: badge.short && room < MIN_FULL_BADGE_PX ? badge.short : badge.text,
    showarrow: false,
    font: { family: theme.fontMono, color: badge.color, size: BADGE_FONT_SIZE },
    bgcolor: badge.fill ?? undefined,
    bordercolor: badge.color,
    borderwidth: 1,
    borderpad: BADGE_PADDING,
  }));
}

function layoutFor(chart: ChartData, width: number): Record<string, unknown> {
  const stacked = chart.traces.some((t) => t.stack_group);
  const hasBars = chart.traces.some((t) => t.kind === "bar");
  const { plan, areaRange } = roles(chart);
  const frames = backgroundFrames(chart);
  const shapes = [...toShapes(chart), ...baselineShapes(chart, plan), ...markerShapes(chart)];
  const annotations = [
    ...(chart.badges?.length ? toAnnotations(chart, width) : []),
    ...endAnnotations(chart, plan),
    ...markerAnnotations(chart),
  ];
  // The card carries the unit, so the figure drops the y-axis title.
  const yAxis = unitInCard(chart) ? { ...chart.y_axis, title: "" } : chart.y_axis;
  return {
    paper_bgcolor: theme.bgChart,
    plot_bgcolor: theme.bgChart,
    font: { family: theme.fontSans, color: theme.ink, size: 12 },
    // Horizontal, above the plot, no frame — clear of a right-hand axis's ticks.
    legend: {
      orientation: "h",
      x: 0,
      xanchor: "left",
      y: 1.02,
      yanchor: "bottom",
      bgcolor: "rgba(0,0,0,0)",
      borderwidth: 0,
      itemwidth: 30,
      font: { family: theme.fontSans, color: theme.inkMuted, size: 12 },
    },
    margin: plan.endLabels.length ? { ...MARGIN, r: END_LABEL_MARGIN_R } : MARGIN,
    height: chart.height,
    hovermode: hoverMode(chart),
    hoverlabel: {
      bgcolor: theme.bgSurface,
      bordercolor: theme.line,
      font: { family: theme.fontSans, color: theme.ink, size: 12 },
    },
    xaxis: { ...axisLayout(chart.x_axis, false), ...SPIKES },
    yaxis: {
      ...axisLayout(yAxis, true),
      nticks: Y_NTICKS,
      // The area's range is set, not left to the fill (see areaYRange).
      ...(areaRange ? { range: [areaRange[0], areaRange[1]] } : {}),
      ...(frames.y ? { range: frames.y } : {}),
    },
    ...(frames.y3 ? { yaxis3: { overlaying: "y", visible: false, range: frames.y3 } } : {}),
    ...(chart.y2_axis
      ? {
          yaxis2: {
            // One set of gridlines only: two at different intervals make a mesh
            // that is harder to read than either scale alone. Only x draws a line.
            ...axisLayout(chart.y2_axis, false),
            showline: false,
            overlaying: "y",
            side: "right",
            ...(frames.y2 ? { range: frames.y2 } : {}),
          },
        }
      : {}),
    ...(hasBars ? { barmode: stacked ? "stack" : "group" } : {}),
    ...(hasBars && chart.bargap != null ? { bargap: chart.bargap } : {}),
    ...(shapes.length ? { shapes } : {}),
    ...(annotations.length ? { annotations } : {}),
  };
}

/**
 * The hovered section takes a forest-tint slab (charts.md § v1.2 — Plan de
 * course), between the boundary rules either side of the cursor. Browser only:
 * it is interaction, which the exported figure has no use for.
 */
function tintHoveredSection(Plotly: Plotly, element: HTMLElement, chart: ChartData): void {
  const cuts = (chart.markers ?? [])
    .filter((m) => m.kind === "boundary")
    .map((m) => Number(m.x))
    .sort((a, b) => a - b);
  if (!cuts.length) return;
  const xs = chart.traces.flatMap((t) => t.x.map(Number)).filter(Number.isFinite);
  const edges = [Math.min(...xs), ...cuts, Math.max(...xs)];
  const plotted = element as HTMLElement & {
    on?: (event: string, handler: (event: { points?: { x: unknown }[] }) => void) => void;
    removeAllListeners?: (event: string) => void;
    layout?: { shapes?: unknown[] };
  };
  if (!plotted.on) return;
  // A redraw (resize, new chart) binds again; drop the last binding first.
  plotted.removeAllListeners?.("plotly_hover");
  plotted.removeAllListeners?.("plotly_unhover");
  const base = () => (plotted.layout?.shapes ?? []).filter((s) => (s as { name?: string }).name !== "hovered");
  plotted.on("plotly_hover", (event) => {
    const x = Number(event.points?.[0]?.x);
    const i = edges.findIndex((edge, k) => k < edges.length - 1 && x >= edge && x <= edges[k + 1]);
    if (i < 0) return;
    Plotly.relayout(element, {
      shapes: [...base(), {
        name: "hovered", type: "rect", xref: "x", yref: "y domain",
        x0: edges[i], x1: edges[i + 1], y0: 0, y1: 1,
        fillcolor: theme.forestTint, line: { width: 0 }, layer: "below",
      }],
    });
  });
  plotted.on("plotly_unhover", () => Plotly.relayout(element, { shapes: base() }));
}

const CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

export function ChartView({ chart }: { chart: ChartData }) {
  const node = useRef<HTMLDivElement>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    const element = node.current;
    if (!element) return;

    const draw = () =>
      loadPlotly()
        .then(async (Plotly) => {
          if (disposed) return;
          await Plotly.react(
            element,
            toPlotlyTraces(chart),
            layoutFor(chart, element.clientWidth),
            CONFIG,
          );
          tintHoveredSection(Plotly, element, chart);
        })
        .catch((error: Error) => !disposed && setFailure(error.message));

    draw();

    // Plotly's own `responsive` handles the resize; what it cannot do is revisit a
    // decision that depended on the width — whether the badge row fits its full
    // wording. Only observed when there is a badge row to re-decide, and only
    // redrawn when the answer actually flips.
    let observer: ResizeObserver | undefined;
    if (chart.badges?.length && typeof ResizeObserver !== "undefined") {
      let fitted = element.clientWidth / chart.badges.length >= MIN_FULL_BADGE_PX;
      observer = new ResizeObserver(() => {
        const fits = element.clientWidth / chart.badges.length >= MIN_FULL_BADGE_PX;
        if (fits === fitted) return;
        fitted = fits;
        draw();
      });
      observer.observe(element);
    }

    return () => {
      disposed = true;
      observer?.disconnect();
      // Plotly attaches listeners and a WebGL context; purge on unmount so a page
      // of many panels doesn't leak them.
      loadPlotly().then((Plotly) => element && Plotly.purge(element)).catch(() => {});
    };
  }, [chart]);

  if (failure) {
    return <Callout tone="terra">Could not draw the chart: {failure}</Callout>;
  }

  return (
    <figure className="chart">
      {/* The title lives in the card, not in the figure (charts.md § Chrome). */}
      {chart.title && (
        <figcaption className="tm-plot__title chart__title">{chart.title}</figcaption>
      )}
      {unitInCard(chart) && <p className="tm-plot__sub chart__unit">{unitInCard(chart)}</p>}
      <div ref={node} className="chart__canvas" />
      {chart.caption && <p className="tm-plot__sub chart__caption">{chart.caption}</p>}
      <button
        type="button"
        className="tm-btn tm-btn--secondary tm-btn--sm"
        onClick={() => downloadChartCsv(chart)}
      >
        Download data (CSV)
      </button>
    </figure>
  );
}

/** Long format: one row per (series, x, y). Any chart can leave as data. */
function downloadChartCsv(chart: ChartData): void {
  const rows: unknown[][] = [];
  for (const trace of chart.traces) {
    trace.y.forEach((y, index) => {
      if (y == null) return;
      rows.push([trace.name, trace.x[index], y]);
    });
  }
  const name = chart.title.replace(/[^\w-]+/g, "_").toLowerCase() || "chart";
  downloadCsv(name, toCsv(["series", "x", "y"], rows));
}

export type { Trace };
