"use client";

/**
 * One plot's result: its charts, tables, prose, images and notes.
 *
 * The notes matter as much as the charts. The backend uses them to explain a
 * partial or empty result — an HR band with no samples, a model that could not be
 * calibrated, activities skipped for lacking per-second data — so a plot that shows
 * less than expected always says why instead of looking broken.
 */

import { ChartView } from "./ChartView";
import { ImageBlockView, TextBlockView } from "./ContentBlocks";
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

  // `texts`/`images` are absent from an output cached before they existed, so read
  // them defensively rather than trusting the type.
  const { charts, tables, notes } = result.output;
  const texts = result.output.texts ?? [];
  const images = result.output.images ?? [];
  const empty =
    !charts.some((c) => c.traces.length) &&
    !tables.some((t) => t.rows.length) &&
    !texts.some((t) => t.text.trim()) &&
    !images.some((i) => i.src.trim());

  return (
    <div className="plot-output">
      {texts.map((text, index) => (
        <TextBlockView key={index} block={text} />
      ))}
      {images.map((image, index) => (
        <ImageBlockView key={index} block={image} />
      ))}
      {charts.map((chart, index) => (
        <ChartView key={index} chart={chart} />
      ))}
      {tables.map((table, index) => (
        <TableView key={index} table={table} />
      ))}
      {/* A content block that is simply still empty says nothing: its own editor is
          right above it, and "no data for this selection" would be a lie. */}
      {empty && !notes.length && !texts.length && !images.length && (
        <p className="note">No data for this selection.</p>
      )}
      {notes.map((note, index) => (
        <p key={index} className="note">
          {note}
        </p>
      ))}
    </div>
  );
}
