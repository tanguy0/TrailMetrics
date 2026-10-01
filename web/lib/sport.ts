/**
 * Sport-type constants shared across the app.
 *
 * Mirrors `src/domain/dataset/sport.py` — kept manually in sync rather than
 * generated, since it's a short, rarely-changed list. Four families: running
 * (what the app was built for), cycling (added alongside it), hiking and
 * swimming. Home's volume totals and PR ladder stay running-only, and the
 * Analysis section's panels can't mix families (see that Python module's
 * docstring for why: GAP and modelled power are running biomechanics, not
 * comparable to a ride, a hike or a swim) — but Home's "latest activity" has
 * no such comparability problem and shows any sport.
 */

export const RUNNING_SPORT_TYPES = ["Run", "TrailRun", "VirtualRun"];
export const CYCLING_SPORT_TYPES = [
  "Ride", "MountainBikeRide", "GravelRide", "VirtualRide",
];
export const HIKING_SPORT_TYPES = ["Hike", "Walk"];
export const SWIMMING_SPORT_TYPES = ["Swim"];

/** The `data-sport` a sport type reads as wherever one is shown — the
 * calendar's session card (its left border), the session detail's sport tag and
 * a training week's totals — so a run, trail, road or virtual, looks the same
 * everywhere. One key per family, not per exact sport type: the families are the
 * unit a rider or a hiker thinks in. Each key has its fixed `--sport-*` token. */
export type SportKey = "run" | "bike" | "hike" | "swim" | "other";

export function sportKey(sportType: string): SportKey {
  if (RUNNING_SPORT_TYPES.includes(sportType)) return "run";
  if (CYCLING_SPORT_TYPES.includes(sportType)) return "bike";
  if (HIKING_SPORT_TYPES.includes(sportType)) return "hike";
  if (SWIMMING_SPORT_TYPES.includes(sportType)) return "swim";
  return "other";
}

/** The stroke icon that redoes the sport beside its colour (SessionCard.md). */
export const SPORT_ICON = {
  run: "run",
  bike: "bike",
  hike: "mountain",
  swim: "swim",
  other: "activity",
} as const satisfies Record<SportKey, string>;
