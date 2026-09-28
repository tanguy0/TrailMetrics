/** One saved race plan. Saved plans belong to an athlete, so this needs a session. */

import { redirect } from "next/navigation";

import { RacePlanScreen } from "@/components/RacePlanScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function SavedRacePlanPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  if (!(await readSession())) redirect("/race-plan");
  const { id } = await params;
  return <RacePlanScreen strings={await loadStrings()} signedIn planId={id} />;
}
