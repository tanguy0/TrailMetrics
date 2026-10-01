/**
 * Which `tm-chip` tone a value reads as (design/tagg/components/Chip.md).
 *
 * The colour is decided once here so the same trend or rating reads the same on
 * Home, on the training calendar and in a session's detail.
 */

export type ChipTone = "neutral" | "forest" | "terra" | "sun" | "moss" | "danger";

/** A metric's direction: up is positive, flat is a signal, down is "you, unfavourably". */
export const TREND_TONE: Record<"increasing" | "stable" | "decreasing", ChipTone> = {
  increasing: "moss",
  stable: "sun",
  decreasing: "terra",
};

/** `tm-chip` plus its tone modifier (neutral has none). */
export function chipClass(tone: ChipTone, extra = ""): string {
  return ["tm-chip", tone === "neutral" ? "" : `tm-chip--${tone}`, extra].filter(Boolean).join(" ");
}

/** RPE 1–10: easy is positive, 5–7 a signal, 8 and up an alert (SessionCard.md). */
export function rpeTone(rpe: number): ChipTone {
  if (rpe <= 4) return "moss";
  if (rpe <= 7) return "sun";
  return "danger";
}

/** How a session felt: the reverse direction of RPE — "fort" is the good one. */
export const FEELING_TONE: Record<"faible" | "ok" | "fort", ChipTone> = {
  fort: "moss",
  ok: "sun",
  faible: "danger",
};
