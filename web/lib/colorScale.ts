/**
 * The app's one progressive accent scale, gold to red.
 *
 * `globals.css` defines the same scale as six fixed steps (`--scale-1`
 * through `--scale-6`, and the `.scale-1`..`.scale-6` utility classes that
 * set `--section-accent`/`-rgb`/`-tint` from them) for pages with a known,
 * small number of sections, like Home's cards. This mirrors the exact same
 * two endpoints and formula for the case a fixed set of classes can't cover:
 * a page with a *variable* number of sections, like an Analysis page's
 * panels — one CSS class per possible count doesn't scale, so this
 * interpolates exactly instead.
 *
 * Keep the endpoints and the tint formula in sync with `globals.css`'s
 * `:root` block if either changes.
 */

import type { CSSProperties } from "react";

const GOLD: [number, number, number] = [232, 163, 61]; // --sunrise
const RED: [number, number, number] = [142, 44, 24]; // --danger
const SURFACE: [number, number, number] = [255, 253, 249]; // --surface

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function toHex(rgb: [number, number, number]): string {
  return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export interface ScaleStep {
  accent: string;
  accentRgb: string;
  tint: string;
}

/** One step of the gold-to-red scale at position `t`, 0 (gold) to 1 (red). */
export function scaleStepAt(t: number): ScaleStep {
  const clamped = Math.min(1, Math.max(0, t));
  const rgb: [number, number, number] = [
    lerp(GOLD[0], RED[0], clamped),
    lerp(GOLD[1], RED[1], clamped),
    lerp(GOLD[2], RED[2], clamped),
  ];
  // Same "10% accent over --surface" formula as every `-tint` token in
  // globals.css.
  const tintRgb: [number, number, number] = [
    lerp(SURFACE[0], rgb[0], 0.1),
    lerp(SURFACE[1], rgb[1], 0.1),
    lerp(SURFACE[2], rgb[2], 0.1),
  ];
  return {
    accent: toHex(rgb),
    accentRgb: rgb.join(", "),
    tint: toHex(tintRgb),
  };
}

/**
 * The `index`-th of `count` evenly-spaced steps — the usual case: a page's
 * `index`-th panel out of `count` panels total. A single panel sits at
 * `t=0` (gold), not the midpoint, matching how one Home section on its own
 * would still be `--scale-1`.
 */
export function scaleStep(index: number, count: number): ScaleStep {
  return scaleStepAt(count > 1 ? index / (count - 1) : 0);
}

/**
 * `scaleStep` as inline CSS custom properties, ready to spread into a React
 * `style` prop — the child elements that read `--section-accent`/`-rgb`/
 * `-tint` (`.panel__title`, `.plot-card`, ...) never need to know they came
 * from JS rather than a `.scale-N` class.
 */
export function scaleStepStyle(index: number, count: number): CSSProperties {
  const step = scaleStep(index, count);
  return {
    "--section-accent": step.accent,
    "--section-accent-rgb": step.accentRgb,
    "--section-accent-tint": step.tint,
  } as CSSProperties;
}
