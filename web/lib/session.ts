/**
 * Server-side session handling.
 *
 * The session token is minted by the Python API (at sign-in, sign-up or password
 * reset) and stored here in a **first-party** `httpOnly` cookie. That is why the
 * auth forms post to this app's own routes rather than to the API: a cookie set
 * by another domain would be third-party and is increasingly blocked outright,
 * and it would force credentialed CORS.
 *
 * The browser never receives a Strava token — only this opaque session, which
 * the API can revoke (design/specs/auth.md § Sessions).
 */

import { cookies } from "next/headers";
import type { NextRequest } from "next/server";
import { cache } from "react";

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

/**
 * Who is signed in, and which access tier that reaches (design/tagg/access.md):
 * `account` (email + password) or `strava` (a Strava athlete attached). `null`
 * is a visitor — no cookie, or one the API no longer honours.
 *
 * Asked of the API once per server render (`cache`), because a cookie alone
 * cannot say whether the session is still alive or whether Strava is attached.
 */
export interface Viewer {
  email: string;
  role: "athlete" | "coach" | "master";
  tier: "account" | "strava";
  isCoach: boolean;
  isMaster: boolean;
}

export const getViewer = cache(async (): Promise<Viewer | null> => {
  const session = await readSession();
  if (!session) return null;
  try {
    const response = await fetch(`${apiBaseUrl()}/auth/session`, {
      headers: { authorization: `Bearer ${session}` },
      cache: "no-store",
    });
    if (!response.ok) return null;
    const body = (await response.json()) as {
      account: { email: string; role: Viewer["role"] };
      strava_connected: boolean;
      is_coach: boolean;
      is_master: boolean;
    };
    return {
      email: body.account.email,
      role: body.account.role,
      tier: body.strava_connected ? "strava" : "account",
      isCoach: body.is_coach,
      isMaster: body.is_master,
    };
  } catch {
    return null;
  }
});

/**
 * The headers a server-to-server auth call carries: the shared service token,
 * and the browser's own IP and user agent — the API sees only this server's
 * address, so its per-IP limits depend on being told the real one.
 */
export function serviceHeaders(request: NextRequest): Record<string, string> {
  const forwarded = request.headers.get("x-forwarded-for");
  const ip = forwarded?.split(",")[0].trim() || request.headers.get("x-real-ip") || "";
  return {
    "content-type": "application/json",
    "x-service-token": serviceToken(),
    "x-client-ip": ip,
    "x-client-user-agent": (request.headers.get("user-agent") ?? "").slice(0, 300),
  };
}

/**
 * CSRF check for every mutating request this app accepts (auth routes, proxy).
 *
 * `SameSite=Lax` already keeps the session cookie off cross-site POSTs; this is
 * the second lock the spec asks for. The forms here post with `fetch`, which
 * always sends `Origin` on a POST — even under this app's `no-referrer` policy,
 * which only blanks it on navigations — so a missing `Origin` falls back to
 * `Sec-Fetch-Site` and anything else is refused.
 */
export function isSameOrigin(request: NextRequest): boolean {
  const origin = request.headers.get("origin");
  if (origin) return origin === request.nextUrl.origin || origin === appUrl();
  const site = request.headers.get("sec-fetch-site");
  return site === "same-origin" || site === "none";
}

/** A relative path to land on after sign-in; anything else becomes `/home`. */
export function safeNext(next: string | null | undefined): string {
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/home";
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
