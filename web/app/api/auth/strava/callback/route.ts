/**
 * Step 2 of the Strava login: Strava redirects the browser here with a code.
 *
 * The code is exchanged **server-to-server** against the compute API, using the
 * shared service token, and the session it returns goes into a first-party
 * `httpOnly` cookie. The authorization code and the Strava tokens never touch
 * client-side JavaScript.
 */

import { NextRequest, NextResponse } from "next/server";

import {
  SESSION_COOKIE,
  apiBaseUrl,
  appUrl,
  sessionCookieOptions,
  serviceToken,
} from "@/lib/session";

function failure(message: string) {
  const url = new URL("/", appUrl());
  url.searchParams.set("error", message.slice(0, 300));
  return NextResponse.redirect(url.toString(), 302);
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const code = params.get("code");
  const denied = params.get("error");

  if (denied) return failure(`Strava authorization was declined (${denied}).`);
  if (!code) return failure("Strava did not return an authorization code.");

  let payload: {
    session_token: string;
    expires_in_days: number;
    // Strava returns no email address, so a brand-new athlete has to be asked for
    // one. Reported by the exchange so this redirect can be decided here, before the
    // app renders anything.
    needs_email?: boolean;
  };
  try {
    const response = await fetch(`${apiBaseUrl()}/auth/strava/exchange`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-service-token": serviceToken(),
      },
      body: JSON.stringify({ code }),
      cache: "no-store",
    });
    if (!response.ok) {
      const detail = await response.text();
      return failure(`Could not complete sign-in: ${detail}`);
    }
    payload = await response.json();
  } catch (error) {
    return failure(`Cannot reach the compute API: ${(error as Error).message}`);
  }

  // `state` carries where to go next; only relative paths are honoured, so a
  // crafted redirect can't bounce the user off-site.
  const state = params.get("state") || "/home";
  const next = state.startsWith("/") && !state.startsWith("//") ? state : "/home";

  // A first-time athlete is asked for their email before anything else, carrying
  // their intended destination along so they land where they were headed.
  const destination = payload.needs_email
    ? `/welcome?next=${encodeURIComponent(next)}`
    : next;

  // `NextResponse.redirect` rather than `Response.redirect`: the latter returns a
  // response whose headers are *immutable*, so setting the session cookie on it
  // throws `TypeError: immutable` and the whole login 500s.
  const redirect = NextResponse.redirect(
    new URL(destination, appUrl()).toString(),
    302,
  );
  redirect.cookies.set(
    SESSION_COOKIE,
    payload.session_token,
    sessionCookieOptions(payload.expires_in_days ?? 30),
  );
  return redirect;
}
