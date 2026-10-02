/**
 * TAGG colour theme — the TypeScript mirror of `design/tagg/tokens.json`
 * (couleurs) and of `src/domain/gap/theme.py`. CSS reads the same values from
 * `app/tokens.css`; `tests/test_theme_tokens.py` keeps the three in sync.
 *
 * Only the values the *renderer* needs live here. Trace colours themselves come
 * down in the chart IR, decided server-side, so a series keeps the same colour in
 * the web app, in an exported figure and in a notebook.
 */

/** Every colour token, by its `tokens.json` name, aliases resolved. */
export const tokens = {
  "bg-page": "#eee9de",
  "bg-surface": "#ffffff",
  "bg-surface-alt": "#f8f6f1",
  "bg-rail": "#1f4b2c",
  "bg-chart": "#ffffff",
  "bg-hero": "#1f4b2c",
  "on-hero-muted": "rgba(255,255,255,0.68)",
  "on-hero-line": "rgba(255,255,255,0.25)",
  "on-hero-fill": "rgba(255,255,255,0.10)",
  "bg-tile": "#f8f6f1",
  hl: "#fdf4e3",
  line: "#e8e2d6",
  "line-strong": "#d5cdbe",
  ink: "#241f19",
  "ink-muted": "#6b6157",
  "ink-faint": "#7d7366",
  "on-forest": "#ffffff",
  "on-rail-muted": "rgba(255,255,255,0.72)",
  forest: "#2e6f40",
  "forest-hover": "#26603a",
  "forest-tint": "#eaefe7",
  terra: "#c65d3b",
  "terra-ink": "#a84a2c",
  "terra-tint": "#f9ede6",
  sun: "#e8a33d",
  "sun-ink": "#9a6516",
  "sun-tint": "#fdf4e3",
  moss: "#5e9c4e",
  "moss-ink": "#3f7a32",
  "moss-tint": "#eff3e8",
  danger: "#8e2c18",
  "danger-tint": "#f4e8e2",
  "sport-run": "#2e6f40",
  "sport-bike": "#3a6ea5",
  "sport-hike": "#b8781f",
  "sport-swim": "#7a4e9e",
  "sport-other": "#7d7366",
  "chart-grid": "#ece7dd",
  "chart-axis": "#7d7366",
  "chart-ref": "#a69a87",
  "chart-you-1": "#2e6f40",
  "chart-you-2": "#c65d3b",
  "chart-you-3": "#e8a33d",
  "chart-you-4": "#3a6ea5",
  "chart-you-5": "#7a4e9e",
  strava: "#fc4c02",
} as const;

/** The roles a renderer asks for (Plotly chrome, map markers). */
export const theme = {
  forest: tokens.forest,
  terra: tokens.terra,
  sun: tokens.sun,
  moss: tokens.moss,
  danger: tokens.danger,

  bgChart: tokens["bg-chart"],
  bgSurface: tokens["bg-surface"],
  chartGrid: tokens["chart-grid"],
  chartAxis: tokens["chart-axis"],
  chartRef: tokens["chart-ref"],
  line: tokens.line,
  ink: tokens.ink,
  inkMuted: tokens["ink-muted"],
  sunInk: tokens["sun-ink"],
  forestTint: tokens["forest-tint"],
  terraTint: tokens["terra-tint"],
  sunTint: tokens["sun-tint"],
  lineStrong: tokens["line-strong"],

  // v1.1 — slope bars read the sign of the gradient, not the series order;
  // fitness / fatigue / form; time markers. Matches theme.py.
  slopeDown: tokens.moss,
  slopeFlat: tokens.forest,
  slopeUp: tokens.terra,
  fitness: tokens["chart-you-1"],
  fatigue: tokens["chart-you-2"],
  form: tokens.sun,
  todayMarker: tokens.sun,
  raceMarker: tokens.terra,

  // tokens.json § type.families — Plotly takes a plain family list.
  fontSans: "Manrope, Helvetica Neue, Arial, sans-serif",
  fontMono: "DM Mono, ui-monospace, SF Mono, Menlo, monospace",
};

/** Top of the vertical fill under series 1 (charts.md § v1.1); matches AREA_ALPHA_TOP. */
export const AREA_ALPHA_TOP = 0.22;

/** Fallback cycle for traces with no explicit colour; matches CURVE_CYCLE. */
export const curvePalette = [
  tokens["chart-you-1"],
  tokens["chart-you-2"],
  tokens["chart-you-3"],
  tokens["chart-you-4"],
  tokens["chart-you-5"],
  tokens["chart-ref"],
];

/**
 * matplotlib-style line codes → Plotly dashes; matches DASH_BY_LINESTYLE. The two
 * dashed styles are the reference patterns of charts.md: "--" is 5-4, ":" is 2-4.
 */
export const dashByCode: Record<string, string> = {
  "-": "solid",
  "--": "5px,4px",
  "-.": "dashdot",
  ":": "2px,4px",
};

/** A reference series is drawn in the reference grey (charts.md § Séries). */
export function isReference(color: string): boolean {
  return color.toLowerCase() === tokens["chart-ref"];
}

/** `#RRGGBB` → `rgba(...)`, for the translucent ±band ribbons. */
export function rgba(color: string, alpha: number): string {
  const hex = color.replace("#", "");
  if (hex.length !== 6) return color;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
