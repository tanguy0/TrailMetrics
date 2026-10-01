/**
 * The Home tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards.
 */

import { HomeScreen } from "@/components/HomeScreen";
import { Teaser } from "@/components/Teaser";
import { readSession } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function HomePage() {
  const strings = await loadStrings();
  // A visitor stays here: the page's empty structure, and what Strava would put
  // in it (design/tagg/visitor.md) — no redirect to the landing.
  if (!(await readSession())) return <Teaser page="home" t={translator(strings)} />;
  return <HomeScreen strings={strings} />;
}
