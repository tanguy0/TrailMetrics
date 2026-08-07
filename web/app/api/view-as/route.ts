/**
 * Sets or clears the coach's "viewing as" cookie.
 *
 * Not itself a security boundary: the API re-checks on every proxied request
 * whether the real, signed-in athlete is actually a coach (see api/deps.py). This
 * route only carries the chosen athlete id along so it survives navigation.
 */

import { NextResponse } from "next/server";

import { VIEW_AS_COOKIE, viewAsCookieOptions } from "@/lib/session";

export async function POST(request: Request) {
  const { athleteId } = await request.json();
  const response = NextResponse.json({ ok: true });
  if (athleteId == null) {
    response.cookies.set(VIEW_AS_COOKIE, "", { ...viewAsCookieOptions(), maxAge: 0 });
  } else {
    response.cookies.set(VIEW_AS_COOKIE, String(athleteId), viewAsCookieOptions());
  }
  return response;
}
