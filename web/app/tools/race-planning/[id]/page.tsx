/** One saved race plan. Saved plans belong to an account, so this needs one. */

import { redirect } from "next/navigation";

import { RacePlanScreen } from "@/components/RacePlanScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function SavedRacePlanPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  if (!(await getViewer())) redirect("/tools/race-planning");
  const { id } = await params;
  return <RacePlanScreen strings={await loadStrings()} signedIn planId={id} />;
}
