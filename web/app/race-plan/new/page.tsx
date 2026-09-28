/** A new, not-yet-saved race plan. Open to visitors too — they just can't save it. */

import { RacePlanScreen } from "@/components/RacePlanScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function NewRacePlanPage() {
  const signedIn = Boolean(await readSession());
  return <RacePlanScreen strings={await loadStrings()} signedIn={signedIn} planId={null} />;
}
