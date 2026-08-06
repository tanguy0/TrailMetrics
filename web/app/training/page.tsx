/**
 * The Training tab.
 *
 * A server component so the session decides before anything renders, and so the
 * translated strings are in the first paint rather than fetched afterwards — the
 * same shell `home/page.tsx` and `pages/page.tsx` use.
 */

import { redirect } from "next/navigation";

import { TrainingScreen } from "@/components/TrainingScreen";
import { readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function TrainingPage() {
  if (!(await readSession())) redirect("/");
  return <TrainingScreen strings={await loadStrings()} />;
}
