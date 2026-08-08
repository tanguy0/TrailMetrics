/**
 * Server-side session handling.
 *
 * The session token is minted by the Python API and stored here in a **first-party**
 * `httpOnly` cookie. That is the whole reason the OAuth callback lands on this app
 * rather than on the API: a cookie set by another domain would be third-party and
 * is increasingly blocked outright, and it would force credentialed CORS.
 *
 * The browser never receives a Strava token — only this opaque session.
 */

import { cookies } from "next/headers";

export const SESSION_COOKIE = "tm_session";
// Which athlete a coach account is currently browsing as. Not itself a security
// boundary — the API re-checks on every request whether the real, signed-in
// athlete (from `tm_session`) is actually a coach — so it doesn't need to be
// signed, just carried along. See api/deps.py's `current_athlete_id`.
export const VIEW_AS_COOKIE = "tm_view_as";
// The chosen UI language, mirroring the athlete's `lang` column so server
// components (the root layout, `loadStrings()`) can pick the right strings
// synchronously, without a DB round trip on every request. The database row is
// still the durable, cross-device source of truth — see `/api/lang`, which
// writes both — this cookie is only the fast path for *this* browser.
export const LANG_COOKIE = "tm_lang";

export function apiBaseUrl(): string {
  const url = process.env.TRAILMETRICS_API_URL;
  if (!url) throw new Error("TRAILMETRICS_API_URL is not set");
  return url.replace(/\/$/, "");
}

export function serviceToken(): string {
  const token = process.env.TRAILMETRICS_SERVICE_TOKEN;
  if (!token) throw new Error("TRAILMETRICS_SERVICE_TOKEN is not set");
  return token;
}

export function appUrl(): string {
  return (process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000").replace(/\/$/, "");
}

/** Not exported: `LANG_COOKIE`'s value is only ever "en" or "fr" (set by
 *  `/api/lang`), but be defensive against a stale or hand-edited cookie. */
const KNOWN_LANGS = new Set(["en", "fr"]);

export async function lang(): Promise<string> {
  const store = await cookies();
  const cookie = store.get(LANG_COOKIE)?.value;
  if (cookie && KNOWN_LANGS.has(cookie)) return cookie;
  return process.env.NEXT_PUBLIC_LANG || "en";
}

export async function readSession(): Promise<string | null> {
  const store = await cookies();
  return store.get(SESSION_COOKIE)?.value ?? null;
}

export function sessionCookieOptions(maxAgeDays: number) {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    // `lax` rather than `strict`: the OAuth redirect back from Strava is a
    // cross-site navigation, and `strict` would drop the cookie on arrival.
    sameSite: "lax" as const,
    path: "/",
    maxAge: maxAgeDays * 24 * 60 * 60,
  };
}

/** No `maxAge`: a browser-session cookie, so a forgotten "viewing as" doesn't
 *  outlive the tab. */
export function viewAsCookieOptions() {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax" as const,
    path: "/",
  };
}

/** A year: long-lived like the choice it remembers, not tied to the session. */
export function langCookieOptions() {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax" as const,
    path: "/",
    maxAge: 365 * 24 * 60 * 60,
  };
}
