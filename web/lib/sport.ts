/**
 * Sport-type constants shared across the app.
 *
 * Mirrors `src/domain/dataset/sport.py` — kept manually in sync rather than
 * generated, since it's a short, rarely-changed list. Two families: running
 * (what the app was built for) and cycling (added alongside it). Home stays
 * running-only; the Analysis section's panels can't mix the two (see that
 * Python module's docstring for why: GAP and modelled power are running
 * biomechanics, not comparable to a ride).
 */

export const RUNNING_SPORT_TYPES = ["Run", "TrailRun", "VirtualRun"];
export const CYCLING_SPORT_TYPES = [
  "Ride", "MountainBikeRide", "GravelRide", "VirtualRide",
];

/** Colour a sport type reads as, wherever one is shown: the calendar's
 * completed-session chip and the session detail's sport tag share this so a
 * trail run — or a ride — looks the same everywhere it appears. */
export type SportTone = "trail" | "run" | "cycling" | "neutral";

export function sportTone(sportType: string): SportTone {
  if (sportType === "TrailRun") return "trail";
  if (sportType === "Run") return "run";
  if (CYCLING_SPORT_TYPES.includes(sportType)) return "cycling";
  return "neutral";
}
