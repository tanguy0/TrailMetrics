/**
 * First-run step: the email address Strava does not give us.
 *
 * A server component so the decision is made before anything paints: no session
 * means back to the sign-in page, and an athlete who already answered is sent
 * straight on rather than being asked twice. That check is server-side on purpose —
 * the same question asked client-side would flash this screen at everyone who
 * navigates here by accident.
 */

import { redirect } from "next/navigation";

import { WelcomeScreen } from "@/components/WelcomeScreen";
import { apiBaseUrl, lang, readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";
import type { Athlete } from "@/lib/types";

/** Relative paths only, so `?next=` can't be used to bounce someone off-site. */
function safeNext(raw: string | undefined): string {
  return raw && raw.startsWith("/") && !raw.startsWith("//") ? raw : "/home";
}

export default async function WelcomePage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>;
}) {
  const session = await readSession();
  if (!session) redirect("/");

  const next = safeNext((await searchParams).next);
  const athlete = await fetchAthlete(session);
  // Already answered — nothing to ask. A failed fetch falls through and asks, which
  // is the harmless direction to be wrong in.
  if (athlete && !athlete.needs_email) redirect(next);

  return <WelcomeScreen strings={await loadStrings()} next={next} />;
}

async function fetchAthlete(session: string): Promise<Athlete | null> {
  try {
    const response = await fetch(
      `${apiBaseUrl()}/auth/me?lang=${encodeURIComponent(lang())}`,
      {
        headers: { authorization: `Bearer ${session}` },
        cache: "no-store",
      },
    );
    return response.ok ? ((await response.json()) as Athlete) : null;
  } catch {
    return null;
  }
}
