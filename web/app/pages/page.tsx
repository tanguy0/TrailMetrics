/**
 * The Analysis tab: what an analysis is, then every one this athlete has.
 *
 * Deliberately in that order. An analysis is an unusual object — a document you
 * assemble from panels rather than a fixed report — so the explanation comes before
 * the grid, and the button to add one comes last, after there is any reason to press
 * it. The three analyses everyone starts with sit in that grid, not off in an
 * "examples" section: they are ordinary analyses that happen to ship with the app.
 *
 * Importing from Strava lives on Home, next to the data it describes.
 */

import { AnalysisScreen } from "@/components/AnalysisScreen";
import { Teaser } from "@/components/Teaser";
import { readSession } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export default async function AnalysisPage() {
  const strings = await loadStrings();
  // A visitor stays here: the page's empty structure, and what Strava would put
  // in it (design/tagg/visitor.md) — no redirect to the landing.
  if (!(await readSession())) return <Teaser page="analysis" t={translator(strings)} />;
  return <AnalysisScreen strings={strings} />;
}
