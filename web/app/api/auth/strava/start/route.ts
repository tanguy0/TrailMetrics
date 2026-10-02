/**
 * Step 1 of connecting Strava: send the user to Strava's consent screen.
 *
 * Strava attaches to an account (design/specs/auth.md), so a visitor is sent to
 * create one first, with this very route as where to go next — they land on
 * Strava's screen right after signing up.
 *
 * The API builds the URL (it owns the client id and the scopes); this route only
 * decides where Strava should come back to — which must be *this* app, so the
 * callback can read the session cookie.
 */

import { NextRequest, NextResponse } from "next/server";

import { SESSION_COOKIE, apiBaseUrl, appUrl, safeNext } from "@/lib/session";

export async function GET(request: NextRequest) {
  // Where to land once connected; kept relative so it can't be an open redirect.
  // Home by default: it is where the import and the profile are.
  const next = safeNext(request.nextUrl.searchParams.get("next"));

  if (!request.cookies.get(SESSION_COOKIE)?.value) {
    const back = `/api/auth/strava/start?next=${encodeURIComponent(next)}`;
    return NextResponse.redirect(
      new URL(`/register?next=${encodeURIComponent(back)}`, appUrl()).toString(),
      302,
    );
  }

  const redirectUri = `${appUrl()}/api/auth/strava/callback`;
  try {
    const response = await fetch(`${apiBaseUrl()}/auth/strava/url`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ redirect_uri: redirectUri, state: next }),
      cache: "no-store",
    });
    if (!response.ok) {
      const detail = await response.text();
      return Response.json(
        { detail: `Could not start Strava login: ${detail}` },
        { status: response.status },
      );
    }
    const { url } = (await response.json()) as { url: string };
    return Response.redirect(url, 302);
  } catch (error) {
    return Response.json(
      { detail: `Cannot reach the compute API: ${(error as Error).message}` },
      { status: 502 },
    );
  }
}
