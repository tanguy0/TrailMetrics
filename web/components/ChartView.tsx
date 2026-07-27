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
  return layout;
}

function toPlotlyTraces(chart: ChartData): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];

  chart.traces.forEach((trace, index) => {
    const color = trace.color || curvePalette[index % curvePalette.length];
    const x = encode(trace.x, chart.x_axis);
    const y = encode(trace.y, chart.y_axis);

    // The ±band goes first so the line draws on top of its own ribbon.
    if (trace.band_upper && trace.band_lower) {
      out.push({
        x: [...encode(trace.x, chart.x_axis), ...encode([...trace.x].reverse(), chart.x_axis)],
        y: [
          ...encode(trace.band_upper, chart.y_axis),
          ...encode([...trace.band_lower].reverse(), chart.y_axis),
        ],
        type: "scatter",
        fill: "toself",
        fillcolor: rgba(color, BAND_ALPHA),
        line: { width: 0 },
        hoverinfo: "skip",
        showlegend: false,
        legendgroup: trace.legend_group || trace.name,
        name: trace.name,
      });
    }

    const common: Record<string, unknown> = {
      x,
      y,
      name: trace.name,
      legendgroup: trace.legend_group || trace.name,
      showlegend: trace.show_legend,
      opacity: trace.opacity,
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
      scatter.line = { color, width: 0.5 };
    }
    out.push(scatter);
  });

  return out;
}

function layoutFor(chart: ChartData): Record<string, unknown> {
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
    margin: { l: 64, r: 20, t: 48, b: 48 },
    height: chart.height,
    hovermode: chart.hover_mode || "closest",
    hoverlabel: { bgcolor: theme.axesFace, font: { color: theme.text } },
    xaxis: axisLayout(chart.x_axis),
    yaxis: axisLayout(chart.y_axis),
    ...(hasBars ? { barmode: stacked ? "stack" : "group" } : {}),
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

    loadPlotly()
      .then((Plotly) => {
        if (disposed) return;
        return Plotly.react(element, toPlotlyTraces(chart), layoutFor(chart), CONFIG);
      })
      .catch((error: Error) => !disposed && setFailure(error.message));

    return () => {
      disposed = true;
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
