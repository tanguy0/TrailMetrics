/**
 * Step 2 of connecting Strava: Strava redirects the browser here with a code.
 *
 * The code is exchanged **server-to-server** against the compute API, with the
 * shared service token *and* the account's session: the API attaches the Strava
 * athlete to that account (or refuses, if another account holds it). No cookie
 * is set here any more — the session already exists. The authorization code and
 * the Strava tokens never touch client-side JavaScript.
 */

import { NextRequest, NextResponse } from "next/server";

import {
  LANG_COOKIE,
  SESSION_COOKIE,
  apiBaseUrl,
  appUrl,
  safeNext,
  serviceHeaders,
} from "@/lib/session";

function back(destination: string, error?: string) {
  const url = new URL(destination, appUrl());
  if (error) url.searchParams.set("error", error.slice(0, 300));
  return NextResponse.redirect(url.toString(), 302);
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const code = params.get("code");
  const denied = params.get("error");
  // `state` carries where to go next; only relative paths are honoured.
  const next = safeNext(params.get("state"));

  const session = request.cookies.get(SESSION_COOKIE)?.value;
  if (!session) return back(`/login?next=${encodeURIComponent(next)}`);
  if (denied) return back(next, `Strava authorization was declined (${denied}).`);
  if (!code) return back(next, "Strava did not return an authorization code.");

  const lang = request.cookies.get(LANG_COOKIE)?.value || process.env.NEXT_PUBLIC_LANG || "en";
  try {
    const response = await fetch(
      `${apiBaseUrl()}/auth/strava/exchange?lang=${encodeURIComponent(lang)}`,
      {
        method: "POST",
        headers: { ...serviceHeaders(request), authorization: `Bearer ${session}` },
        body: JSON.stringify({ code }),
        cache: "no-store",
      },
    );
    if (response.status === 401) return back(`/login?next=${encodeURIComponent(next)}`);
    if (!response.ok) {
      const payload = (await response.json().catch(() => ({}))) as { detail?: unknown };
      const detail = typeof payload.detail === "string" ? payload.detail : response.statusText;
      return back(next, detail);
    }
  } catch (error) {
    return back(next, `Cannot reach the compute API: ${(error as Error).message}`);
  }
  return back(next);
}
