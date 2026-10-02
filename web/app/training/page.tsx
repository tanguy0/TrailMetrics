/**
 * The Training tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards — the
 * same shell `home/page.tsx` and `pages/page.tsx` use.
 */

import { TrainingScreen } from "@/components/TrainingScreen";
import { Teaser } from "@/components/Teaser";
import { getViewer } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function TrainingPage() {
  const strings = await loadStrings();
  // A visitor, or an account without Strava, stays here: the page's empty
  // structure and what the next tier would put in it (design/tagg/access.md).
  const viewer = await getViewer();
  if (viewer?.tier !== "strava") {
    return <Teaser page="training" tier={viewer ? "account" : "visitor"} t={translator(strings)} />;
  }
  return <TrainingScreen strings={strings} />;
}
