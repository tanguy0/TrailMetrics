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

export function lang(): string {
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
