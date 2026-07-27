/**
 * Step 1 of the Strava login: send the user to Strava's consent screen.
 *
 * The API builds the URL (it owns the client id and the scopes); this route only
 * decides where Strava should come back to — which must be *this* app, so the
 * session cookie it eventually sets is first-party.
 */

import { NextRequest } from "next/server";

import { apiBaseUrl, appUrl } from "@/lib/session";

export async function GET(request: NextRequest) {
  const redirectUri = `${appUrl()}/api/auth/strava/callback`;
  // Where to land after login; kept relative so it can't be used as an open redirect.
  const next = request.nextUrl.searchParams.get("next") || "/pages";
  const state = next.startsWith("/") ? next : "/pages";

  try {
    const response = await fetch(`${apiBaseUrl()}/auth/strava/url`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ redirect_uri: redirectUri, state }),
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
