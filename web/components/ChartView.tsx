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

import { durationToEpoch, toCsv, downloadCsv } from "@/lib/format";
import { curvePalette, dashByCode, rgba, theme } from "@/lib/theme";
import type { Axis, ChartData, Trace } from "@/lib/types";

// Resolved once per session; `plotly.js-dist-min` has no types of its own.
let plotlyPromise: Promise<any> | null = null;
function loadPlotly(): Promise<any> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m: any) => m.default ?? m);
  return plotlyPromise;
}

const BAND_ALPHA = 0.16;

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

function axisLayout(axis: Axis): Record<string, unknown> {
  const layout: Record<string, unknown> = {
    title: { text: axis.title },
    gridcolor: theme.grid,
    linecolor: theme.spine,
    zeroline: false,
    color: theme.text,
    automargin: true,
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
  // Tints the axis to its series, so a dual-axis chart says which line it measures.
  if (axis.color) {
    layout.title = { text: axis.title, font: { color: axis.color } };
    layout.tickfont = { color: axis.color };
  }
  return layout;
}

function toPlotlyTraces(chart: ChartData): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];

  chart.traces.forEach((trace, index) => {
    const color = trace.color || curvePalette[index % curvePalette.length];
    // A trace's values are encoded against the axis it is actually measured on.
    const onSecondary = trace.axis === "y2" && Boolean(chart.y2_axis);
    const yAxis = onSecondary ? chart.y2_axis! : chart.y_axis;
    const x = encode(trace.x, chart.x_axis);
    const y = encode(trace.y, yAxis);

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
        fillcolor: rgba(color, BAND_ALPHA),
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
      showlegend: trace.show_legend,
      opacity: trace.opacity,
      ...(onSecondary ? { yaxis: "y2" } : {}),
    };
    if (trace.hover_text) common.customdata = trace.hover_text;
    if (trace.hover_template) common.hovertemplate = trace.hover_template;

    if (trace.kind === "bar") {
      out.push({ ...common, type: "bar", marker: { color } });
      return;
    }

    const line: Record<string, unknown> = { color, width: trace.width };
    const dash = dashByCode[trace.dash] ?? "solid";
    if (dash !== "solid") line.dash = dash;
    if (trace.kind === "step") line.shape = "hv";

    const scatter: Record<string, unknown> = { ...common, type: "scatter", line };
    if (trace.kind === "scatter") {
      scatter.mode = "markers";
      scatter.marker = { color, size: trace.marker_size };
    } else {
      scatter.mode = trace.markers ? "lines+markers" : "lines";
      if (trace.markers) scatter.marker = { color, size: trace.marker_size };
    }
    if (trace.kind === "area") {
      scatter.stackgroup = trace.stack_group || "area";
      scatter.fillcolor = rgba(color, trace.stack_group ? 0.35 : 0.2);
      scatter.line = { color, width: 0.35 };
    }
    out.push(scatter);
  });

  return out;
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
    font: { color: badge.color, size: BADGE_FONT_SIZE },
    bgcolor: badge.fill ?? undefined,
    bordercolor: badge.color,
    borderwidth: 1,
    borderpad: BADGE_PADDING,
  }));
}

function layoutFor(chart: ChartData, width: number): Record<string, unknown> {
  const stacked = chart.traces.some((t) => t.stack_group);
  const hasBars = chart.traces.some((t) => t.kind === "bar");
  return {
    title: { text: chart.title, font: { color: theme.text, size: 16 } },
    paper_bgcolor: theme.figureFace,
    plot_bgcolor: theme.axesFace,
    font: { color: theme.text, size: 12 },
    legend: {
      bgcolor: theme.axesFace,
      bordercolor: theme.spine,
      borderwidth: 1,
      font: { color: theme.text },
    },
    // A right-hand axis needs room for its own ticks and title.
    margin: { l: 64, r: chart.y2_axis ? 64 : 20, t: 48, b: 48 },
    height: chart.height,
    // On a dual-axis chart the shared x-value is the only thing the two series
    // genuinely have in common, so read them together rather than one at a time.
    hovermode: chart.y2_axis ? "x unified" : chart.hover_mode || "closest",
    hoverlabel: { bgcolor: theme.axesFace, font: { color: theme.text } },
    xaxis: axisLayout(chart.x_axis),
    yaxis: axisLayout(chart.y_axis),
    ...(chart.y2_axis
      ? {
          yaxis2: {
            ...axisLayout(chart.y2_axis),
            overlaying: "y",
            side: "right",
            // One set of gridlines only: two at different intervals make a mesh
            // that is harder to read than either scale alone.
            showgrid: false,
          },
        }
      : {}),
    ...(hasBars ? { barmode: stacked ? "stack" : "group" } : {}),
    ...(chart.bands?.length ? { shapes: toShapes(chart) } : {}),
    ...(chart.badges?.length ? { annotations: toAnnotations(chart, width) } : {}),
  };
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
        .then((Plotly) => {
          if (disposed) return;
          return Plotly.react(
            element,
            toPlotlyTraces(chart),
            layoutFor(chart, element.clientWidth),
            CONFIG,
          );
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
    return <p className="note note--error">Could not draw the chart: {failure}</p>;
  }

  return (
    <figure className="chart">
      <div ref={node} className="chart__canvas" />
      {chart.caption && <figcaption className="chart__caption">{chart.caption}</figcaption>}
      <button
        type="button"
        className="button button--ghost button--small"
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
