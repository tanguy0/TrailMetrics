/**
 * The My Pages tab: what a page is, the examples, then your own pages.
 *
 * Deliberately in that order. A page is an unusual object — a document you
 * assemble from panels rather than a fixed report — so the explanation and the
 * three worked examples come before the empty state, and the button to make one
 * comes last, after there is any reason to press it.
 *
 * Importing from Strava lives on Home, next to the data it describes.
 */

import { redirect } from "next/navigation";

import { PagesScreen } from "@/components/PagesScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function MyPages() {
  if (!(await readSession())) redirect("/");
  return <PagesScreen strings={await loadStrings()} />;
}
