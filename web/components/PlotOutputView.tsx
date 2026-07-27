"use client";

/**
 * One plot's result: its charts, its tables, and its notes.
 *
 * The notes matter as much as the charts. The backend uses them to explain a
 * partial or empty result — an HR band with no samples, a model that could not be
 * calibrated, activities skipped for lacking per-second data — so a plot that shows
 * less than expected always says why instead of looking broken.
 */

import { ChartView } from "./ChartView";
import { TableView } from "./TableView";
import type { PlotResult } from "@/lib/types";

export function PlotOutputView({
  result,
  onCompute,
}: {
  result: PlotResult;
  onCompute?: (plotId: string) => void;
}) {
  if (result.error) {
    return (
      <p className="note note--error">
        <strong>{result.plot_type}</strong> failed: {result.error}
      </p>
    );
  }

  if (result.pending) {
    return (
      <div className="pending">
        <p className="muted">
          This plot fits a model, so it runs on demand rather than on every edit.
        </p>
        <button
          type="button"
          className="button"
          onClick={() => onCompute?.(result.plot_id)}
          disabled={!onCompute}
        >
          Compute
        </button>
      </div>
    );
  }

  const { charts, tables, notes } = result.output;
  const empty = !charts.some((c) => c.traces.length) && !tables.some((t) => t.rows.length);

  return (
    <div className="plot-output">
      {charts.map((chart, index) => (
        <ChartView key={index} chart={chart} />
      ))}
      {tables.map((table, index) => (
        <TableView key={index} table={table} />
      ))}
      {empty && !notes.length && <p className="note">No data for this selection.</p>}
      {notes.map((note, index) => (
        <p key={index} className="note">
          {note}
        </p>
      ))}
    </div>
  );
}
