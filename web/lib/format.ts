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

/**
 * A duration as `5h24` (hours present) or `27min` (no hours) — no seconds.
 * For a total where second-level precision is noise, not signal (a week's
 * summed moving time); a single activity's duration or a PR still wants
 * `formatHms`'s seconds.
 */
export function formatHoursMinutes(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const totalMinutes = Math.round(seconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${hours}h${String(minutes).padStart(2, "0")}` : `${minutes}min`;
}

export function formatPace(secondsPerKm: number | null | undefined): string {
  if (secondsPerKm == null || !Number.isFinite(secondsPerKm)) return "—";
  const total = Math.round(secondsPerKm);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}/km`;
}

export function formatSpeed(kmh: number | null | undefined): string {
  if (kmh == null || !Number.isFinite(kmh)) return "—";
  return `${kmh.toFixed(1)} km/h`;
}

/** A pace as `M:SS`, for an editable field — no `/km` suffix to re-parse out. */
export function formatPaceInput(secondsPerKm: number | null | undefined): string {
  if (secondsPerKm == null || !Number.isFinite(secondsPerKm)) return "";
  const total = Math.round(secondsPerKm);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** Parses `M:SS` or `MM:SS` back into seconds; `null` for anything else. */
export function parsePaceInput(text: string): number | null {
  const match = text.trim().match(/^(\d+):([0-5]\d)$/);
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]);
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

/** A distance in km: one decimal below 100 km, none at or above — a three- or
 *  four-digit distance doesn't need a decimal to be readable. */
export function formatDistanceAdaptive(km: number): string {
  return formatNumber(km, km >= 100 ? 0 : 1);
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
