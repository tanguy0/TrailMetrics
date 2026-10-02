/**
 * The Coaching tab (formerly Training).
 *
 * A visitor gets the teaser (an account is what opens the offer and its form);
 * any account gets `CoachingScreen`, which picks between the offer, the diary
 * and the coach's board from `/coaching/me`. Server-side only to decide the
 * visitor case before anything renders, as elsewhere.
 */

import { CoachingScreen } from "@/components/CoachingScreen";
import { Teaser } from "@/components/Teaser";
import { getViewer } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function CoachingPage() {
  const strings = await loadStrings();
  const viewer = await getViewer();
  if (!viewer) return <Teaser page="training" t={translator(strings)} />;
  return <CoachingScreen strings={strings} isCoach={viewer.isCoach} hasStrava={viewer.tier === "strava"} />;
}
