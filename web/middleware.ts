import { NextRequest, NextResponse } from "next/server";

/**
 * Nonce-based Content-Security-Policy for script-src.
 *
 * This has to live in middleware, not next.config.mjs's static `headers()|
 * (see there for the rest of the security headers): a nonce must be fresh
 * per request, and middleware is the only place that runs on every request.
 *
 * The App Router always injects its own inline `<script>` tags to stream RSC
 * hydration data — there is no way to turn that off. A `script-src 'self'` with
 * no nonce and no `unsafe-inline` blocks those unconditionally, which doesn't
 * error loudly — the page just never hydrates, so every client-side data fetch
 * silently never runs. Next reads the nonce back out of this response's own
 * `Content-Security-Policy` header and stamps it onto every script it injects,
 * so nothing else has to change.
 */
export function middleware(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const isDev = process.env.NODE_ENV !== "production";

  const csp = [
    "default-src 'self'",
    // `strict-dynamic` lets the nonce'd bootstrap script load whatever chunks it
    // needs without this policy having to enumerate them. `unsafe-eval` only in
    // dev, for hot reload's `eval` — see next.config.mjs for the full story.
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${isDev ? " 'unsafe-eval'" : ""}`,
    // Inline `style={{...}}` attributes have no nonce mechanism in the CSP spec
    // — only <style> *elements* can be nonce'd — so `unsafe-inline` is the only
    // way to allow the handful of components that set one directly.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "object-src 'none'",
  ].join("; ");

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", csp);
  return response;
}

export const config = {
  // Skip static assets and image optimization — no point minting a nonce for a
  // chunk request that emits no HTML.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
