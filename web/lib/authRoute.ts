/**
 * The shared body of this app's auth routes (`/api/auth/*`).
 *
 * Each one checks the request is same-origin, forwards the form to the API with
 * the service token, and — when the API answers with a session — puts it in the
 * first-party `httpOnly` cookie. The token never reaches client-side JavaScript:
 * the browser gets back everything *but* it.
 */

import { NextRequest, NextResponse } from "next/server";

import {
  LANG_COOKIE,
  SESSION_COOKIE,
  apiBaseUrl,
  isSameOrigin,
  langCookieOptions,
  serviceHeaders,
  sessionCookieOptions,
} from "@/lib/session";

const KNOWN_LANGS = new Set(["en", "fr"]);

export function forbidden() {
  return NextResponse.json({ detail: "Cross-origin request refused." }, { status: 403 });
}

export async function readJson(request: NextRequest): Promise<Record<string, unknown>> {
  try {
    const body = await request.json();
    return body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/** POST `body` to the API's `path`; set the session cookie if one comes back. */
export async function forwardAuth(
  request: NextRequest,
  path: string,
  body: Record<string, unknown>,
): Promise<NextResponse> {
  if (!isSameOrigin(request)) return forbidden();
  const lang = request.cookies.get(LANG_COOKIE)?.value || process.env.NEXT_PUBLIC_LANG || "en";

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}?lang=${encodeURIComponent(lang)}`, {
      method: "POST",
      headers: serviceHeaders(request),
      body: JSON.stringify(body),
      cache: "no-store",
    });
  } catch (error) {
    return NextResponse.json(
      { detail: `Cannot reach the compute API: ${(error as Error).message}` },
      { status: 502 },
    );
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (!response.ok) {
    // The API's auth errors arrive already translated; a validation error from
    // FastAPI (a list) is not meant for the reader, so it is left to the form.
    const detail = typeof payload.detail === "string" ? payload.detail : null;
    return NextResponse.json({ detail }, { status: response.status });
  }

  const { session_token: token, expires_in_days: days, ...rest } = payload;
  const out = NextResponse.json({ ok: true, ...rest });
  if (typeof token === "string") {
    out.cookies.set(SESSION_COOKIE, token, sessionCookieOptions(Number(days) || 30));
    // The account's language follows it onto this browser.
    if (typeof rest.lang === "string" && KNOWN_LANGS.has(rest.lang)) {
      out.cookies.set(LANG_COOKIE, rest.lang, langCookieOptions());
    }
  }
  return out;
}

/** Revoke via the API (`/auth/logout` or `/auth/logout-all`), then drop the cookies. */
export async function signOut(request: NextRequest, path: string): Promise<NextResponse> {
  if (!isSameOrigin(request)) return forbidden();
  const session = request.cookies.get(SESSION_COOKIE)?.value;
  if (session) {
    try {
      await fetch(`${apiBaseUrl()}${path}`, {
        method: "POST",
        headers: { authorization: `Bearer ${session}` },
        cache: "no-store",
      });
    } catch {
      // The cookie goes regardless: signing out must work with the API down.
    }
  }
  const out = NextResponse.json({ ok: true });
  out.cookies.set(SESSION_COOKIE, "", { ...sessionCookieOptions(0), maxAge: 0 });
  out.cookies.set("tm_view_as", "", { ...sessionCookieOptions(0), maxAge: 0 });
  return out;
}
