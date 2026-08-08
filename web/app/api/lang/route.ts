/**
 * Sets the UI-language cookie for this browser.
 *
 * The durable, cross-device preference is the athlete's `lang` column, written
 * through the normal `PATCH /auth/me` (see `updateProfile`) — this route only
 * carries the choice into the cookie `lib/session.ts`'s `lang()` reads
 * synchronously, so the server-rendered shell (root layout, `loadStrings()`)
 * doesn't need a DB round trip on every request. Mirrors `/api/view-as`.
 */

import { NextResponse } from "next/server";

import { LANG_COOKIE, langCookieOptions } from "@/lib/session";

const KNOWN_LANGS = new Set(["en", "fr"]);

export async function POST(request: Request) {
  const { lang } = await request.json();
  if (typeof lang !== "string" || !KNOWN_LANGS.has(lang)) {
    return NextResponse.json({ detail: "Unknown language." }, { status: 422 });
  }
  const response = NextResponse.json({ ok: true });
  response.cookies.set(LANG_COOKIE, lang, langCookieOptions());
  return response;
}
