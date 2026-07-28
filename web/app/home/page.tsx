/**
 * The Home tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards.
 */

import { redirect } from "next/navigation";

import { HomeScreen } from "@/components/HomeScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function HomePage() {
  if (!(await readSession())) redirect("/");
  return <HomeScreen strings={await loadStrings()} />;
}
