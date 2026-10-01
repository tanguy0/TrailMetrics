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
