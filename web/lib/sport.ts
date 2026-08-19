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

/** Colour a sport type reads as, wherever one is shown: the calendar's
 * completed-session chip, the session detail's sport tag, and a training
 * week's totals share this so a run — trail, road, or virtual — looks the
 * same everywhere it appears. One tone per family, not per exact sport
 * type: the four families are the unit a rider or a hiker thinks in, not
 * the distinction between a trail and a road run. */
export type SportTone = "running" | "hiking" | "cycling" | "swimming" | "neutral";

export function sportTone(sportType: string): SportTone {
  if (RUNNING_SPORT_TYPES.includes(sportType)) return "running";
  if (CYCLING_SPORT_TYPES.includes(sportType)) return "cycling";
  if (HIKING_SPORT_TYPES.includes(sportType)) return "hiking";
  if (SWIMMING_SPORT_TYPES.includes(sportType)) return "swimming";
  return "neutral";
}
