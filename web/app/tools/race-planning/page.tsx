/**
 * Tools → Race Planning — open to everyone, like the blog.
 *
 * An account lands on its saved plans; a visitor, who has nowhere to save them,
 * lands straight on the planner. Personal curves additionally need Strava, which
 * the planner itself handles (reference curves otherwise).
 */

import type { Metadata } from "next";

import { RacePlanList } from "@/components/RacePlanList";
import { RacePlanScreen } from "@/components/RacePlanScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "Planification de course — TAGG" };

export default async function RacePlanningPage() {
  const strings = await loadStrings();
  if (await getViewer()) return <RacePlanList strings={strings} />;
  return <RacePlanScreen strings={strings} signedIn={false} planId={null} />;
}
