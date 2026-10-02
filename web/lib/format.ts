/**
 * Value formatting, matching the Python side.
 *
 * Durations arrive as plain seconds and paces as seconds-per-kilometre, so the
 * clock formatting has to happen here. Keeping the IR numeric rather than
 * pre-formatted is what lets the same payload feed a chart axis, a hover label, a
 * table cell and a CSV export.
 */

import type { CellFormat } from "./types";

/**
 * A duration as a clock: `1:12` under the hour, `4:12:08` above, never a
 * leading zero (design/tagg/density.md). Past a day, a total reads in hours —
 * `1 062 h` — unless `exact`: a race target of 30 hours is still `30:00:00`,
 * and an input needs the clock back.
 */
export function formatHms(
  seconds: number | null | undefined,
  { exact = false }: { exact?: boolean } = {},
): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  if (!exact && total >= 86_400) return `${formatNumber(total / 3600, 0)} h`;
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

export type DateStyle = "short" | "relative" | "long";

/**
 * The page's locale when the caller has none to hand — `<html lang>`, which the
 * root layout sets from the session. Callers with strings pass `t("locale")`.
 */
function defaultLocale(): string | undefined {
  return typeof document !== "undefined" ? document.documentElement.lang || undefined : undefined;
}

/** A `YYYY-MM-DD` is a calendar day, read in local time — not UTC midnight. */
function toDate(value: string | number | Date): Date {
  if (value instanceof Date) return value;
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [y, m, d] = value.split("-").map(Number);
    return new Date(y, m - 1, d);
  }
  return new Date(value);
}

/**
 * A date as a person reads it (design/tagg/density.md — never ISO in the UI):
 *
 * - `short` — tiles, cells, pills: `30 sept.`, with `25` added when the year is
 *   not the current one;
 * - `relative` — hero, meta, sync state: `il y a 12 min`, `hier`, `lundi`, then
 *   `short` past a week;
 * - `long` — page titles, PDF: `mardi 30 septembre 2026`.
 */
export function formatDate(
  value: string | number | Date | null | undefined,
  style: DateStyle = "short",
  locale: string | undefined = defaultLocale(),
): string {
  if (value == null || value === "") return "—";
  const date = toDate(value);
  if (Number.isNaN(date.getTime())) return "—";
  if (style === "long") {
    return date.toLocaleDateString(locale, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  }
  if (style === "relative") {
    const relative = relativeDate(date, locale);
    if (relative) return relative;
  }
  const dayMonth = date.toLocaleDateString(locale, { day: "numeric", month: "short" });
  return date.getFullYear() === new Date().getFullYear()
    ? dayMonth
    : `${dayMonth} ${String(date.getFullYear()).slice(-2)}`;
}

/** `il y a 12 min`, `il y a 3 h`, `hier`, a weekday — or null past a week. */
function relativeDate(date: Date, locale: string | undefined): string | null {
  const now = new Date();
  const minutes = Math.round((now.getTime() - date.getTime()) / 60_000);
  if (minutes < 0) return null;
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto", style: "short" });
  if (minutes < 60) return rtf.format(-minutes, "minute");
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(now) - startOf(date)) / 86_400_000);
  if (days === 0) return rtf.format(-Math.round(minutes / 60), "hour");
  if (days === 1) return rtf.format(-1, "day");
  if (days < 7) return date.toLocaleDateString(locale, { weekday: "long" });
  return null;
}

/** A date range: `29 sept. – 5 oct.` (thin spaces around an en dash). */
export function formatDateRange(
  start: string | Date,
  end: string | Date,
  locale: string | undefined = defaultLocale(),
): string {
  return `${formatDate(start, "short", locale)}\u2009–\u2009${formatDate(end, "short", locale)}`;
}

/**
 * A pace interval, fastest first, no spaces: `6:17–6:48` (density.md). The
 * `/km` belongs in the label then — "Allure (/km)" — not in the value.
 */
export function formatPaceRange(fastSecondsPerKm: number, slowSecondsPerKm: number): string {
  return `${formatPaceInput(fastSecondsPerKm)}–${formatPaceInput(slowSecondsPerKm)}`;
}

/**
 * A tile's number size by its length, unit excluded (density.md): the default
 * up to 5 characters, `--lg` from 6, `--sm` past 9.
 */
export function kpiNumClass(value: string): string {
  const length = value.length;
  return length > 9 ? "tm-kpi__num tm-kpi__num--sm" : length > 5 ? "tm-kpi__num tm-kpi__num--lg" : "tm-kpi__num";
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
      return formatDate(value as string, "short");
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
