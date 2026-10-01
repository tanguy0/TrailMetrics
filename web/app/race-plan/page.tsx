/**
 * The Race plan tab ("Plan de course") — public, like the blog.
 *
 * A signed-in athlete lands on their saved plans; a visitor, who has nothing to
 * save plans to, lands straight on the planner (reference curves only). Server
 * component so the session decides before anything renders, as elsewhere.
 */

import type { Metadata } from "next";

import { RacePlanList } from "@/components/RacePlanList";
import { RacePlanScreen } from "@/components/RacePlanScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "Plan de course — TAGG" };

export default async function RacePlanPage() {
  const strings = await loadStrings();
  if (await readSession()) return <RacePlanList strings={strings} />;
  return <RacePlanScreen strings={strings} signedIn={false} planId={null} />;
}
