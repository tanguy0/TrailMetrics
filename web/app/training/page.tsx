/**
 * The Training tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards — the
 * same shell `home/page.tsx` and `pages/page.tsx` use.
 */

import { TrainingScreen } from "@/components/TrainingScreen";
import { Teaser } from "@/components/Teaser";
import { readSession } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function TrainingPage() {
  const strings = await loadStrings();
  // A visitor stays here: the page's empty structure, and what Strava would put
  // in it (design/tagg/visitor.md) — no redirect to the landing.
  if (!(await readSession())) return <Teaser page="training" t={translator(strings)} />;
  return <TrainingScreen strings={strings} />;
}
