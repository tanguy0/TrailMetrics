/**
 * The Home tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards.
 */

import { HomeScreen } from "@/components/HomeScreen";
import { Teaser } from "@/components/Teaser";
import { getViewer } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const strings = await loadStrings();
  // A visitor stays here: the page's empty structure, and what an account would
  // put in it (design/tagg/visitor.md) — no redirect to the landing. An account
  // without Strava gets the real page, emptied (HomeScreen's degraded mode).
  if (!(await getViewer())) return <Teaser page="home" t={translator(strings)} />;
  // `error` is the Strava callback's way back (a refused attachment, a declined
  // consent), already in the reader's language.
  return <HomeScreen strings={strings} notice={(await searchParams).error ?? null} />;
}
