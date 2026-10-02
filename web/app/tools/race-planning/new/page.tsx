/** A new, not-yet-saved race plan. Open to everyone; saving needs an account. */

import { RacePlanScreen } from "@/components/RacePlanScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function NewRacePlanPage() {
  const signedIn = Boolean(await getViewer());
  return <RacePlanScreen strings={await loadStrings()} signedIn={signedIn} planId={null} />;
}
