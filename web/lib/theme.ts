/**
 * Trail / Earthy palette — the TypeScript mirror of `src/domain/gap/theme.py`.
 *
 * Only the values the *renderer* needs live here. Trace colours themselves come
 * down in the chart IR, decided server-side, so a series keeps the same colour in
 * the web app, in an exported figure and in a notebook.
 */

export const theme = {
  primary: "#2E6F40", // forest green
  terracotta: "#C65D3B",
  sunrise: "#E8A33D",
  moss: "#5E9C4E",

  figureFace: "#FBF8F3", // warm off-white
  axesFace: "#FFFDF9",
  grid: "#CFC3AE",
  text: "#241F19",
  spine: "#B8AC97",
  muted: "#6B6157",
  danger: "#8E2C18",
};

/** Fallback cycle for traces with no explicit colour; matches CURVE_PALETTE. */
export const curvePalette = [
  "#2E6F40", "#C65D3B", "#E8A33D", "#3A6EA5", "#7A4E9E",
  "#5E9C4E", "#A6843E", "#14532B", "#B5651D", "#6B4226",
  "#2A7E8C", "#9E4E6E",
];

/** matplotlib-style line codes → Plotly dash names. */
export const dashByCode: Record<string, string> = {
  "-": "solid",
  "--": "dash",
  "-.": "dashdot",
  ":": "dot",
};

/** `#RRGGBB` → `rgba(...)`, for the translucent ±band ribbons. */
export function rgba(color: string, alpha: number): string {
  const hex = color.replace("#", "");
  if (hex.length !== 6) return color;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
