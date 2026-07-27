/**
 * Value formatting, matching the Python side.
 *
 * Durations arrive as plain seconds and paces as seconds-per-kilometre, so the
 * clock formatting has to happen here. Keeping the IR numeric rather than
 * pre-formatted is what lets the same payload feed a chart axis, a hover label, a
 * table cell and a CSV export.
 */

import type { CellFormat } from "./types";

export function formatHms(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}

export function formatPace(secondsPerKm: number | null | undefined): string {
  if (secondsPerKm == null || !Number.isFinite(secondsPerKm)) return "—";
  const total = Math.round(secondsPerKm);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}/km`;
}

export function formatNumber(value: number, decimals: number): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatDate(value: string | number | Date | null): string {
  if (value == null) return "—";
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toISOString().slice(0, 10);
}

/** Render one table cell according to its column's declared format. */
export function formatCell(value: unknown, format: CellFormat): string {
  if (value == null || value === "") return "—";
  switch (format.kind) {
    case "duration":
      return formatHms(Number(value));
    case "pace":
      return formatPace(Number(value));
    case "date":
      return formatDate(value as string);
    case "integer":
      return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "—";
    case "percent": {
      const n = Number(value);
      return Number.isFinite(n) ? `${formatNumber(n, format.decimals)} %` : "—";
    }
    case "number": {
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      const text = formatNumber(n, format.decimals);
      return format.suffix ? `${text} ${format.suffix}` : text;
    }
    default:
      return String(value);
  }
}

/**
 * Seconds → an epoch timestamp (ms), so a duration can ride on a time axis and
 * tick as `m:ss` instead of a raw number. The same trick the Python renderer uses.
 */
export function durationToEpoch(seconds: number | null): number | null {
  if (seconds == null || !Number.isFinite(seconds)) return null;
  return seconds * 1000;
}

export function formatDistanceKm(metres: number): string {
  return `${(metres / 1000).toFixed(2)} km`;
}

/** Quote a CSV field only when it needs it. */
function csvField(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(headers: string[], rows: unknown[][]): string {
  return [headers, ...rows].map((row) => row.map(csvField).join(",")).join("\n");
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.endsWith(".csv") ? filename : `${filename}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
