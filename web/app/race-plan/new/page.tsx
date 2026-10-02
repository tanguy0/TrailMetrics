/** A new, not-yet-saved race plan. Open to everyone; saving needs a Strava athlete for now. */

import { RacePlanScreen } from "@/components/RacePlanScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function NewRacePlanPage() {
  const signedIn = (await getViewer())?.tier === "strava";
  return <RacePlanScreen strings={await loadStrings()} signedIn={signedIn} planId={null} />;
}
