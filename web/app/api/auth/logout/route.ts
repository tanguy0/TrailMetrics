/**
 * Sign out by clearing the session cookie.
 *
 * Sessions are stateless signed tokens, so there is nothing to revoke server-side.
 * Dropping the cookie is the whole operation.
 */

import { NextResponse } from "next/server";

import { appUrl, SESSION_COOKIE, sessionCookieOptions } from "@/lib/session";

function clearAndRedirect() {
  // `NextResponse.redirect` rather than `Response.redirect`: the latter returns a
  // response whose headers are *immutable*, so attaching a cookie to it throws.
  const response = NextResponse.redirect(appUrl(), 302);
  response.cookies.set(SESSION_COOKIE, "", { ...sessionCookieOptions(0), maxAge: 0 });
  return response;
}

export async function POST() {
  return clearAndRedirect();
}

export async function GET() {
  return clearAndRedirect();
}
