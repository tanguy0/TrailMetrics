/**
 * The Race plan tab ("Plan de course") — public, like the blog.
 *
 * An athlete with Strava lands on their saved plans; anyone else lands straight
 * on the planner (reference curves only). Saved plans are still keyed by the
 * Strava athlete — they move to the account with the Tools pages. Server
 * component so the session decides before anything renders, as elsewhere.
 */

import type { Metadata } from "next";

import { RacePlanList } from "@/components/RacePlanList";
import { RacePlanScreen } from "@/components/RacePlanScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "Plan de course — TAGG" };

export default async function RacePlanPage() {
  const strings = await loadStrings();
  if ((await getViewer())?.tier === "strava") return <RacePlanList strings={strings} />;
  return <RacePlanScreen strings={strings} signedIn={false} planId={null} />;
}
