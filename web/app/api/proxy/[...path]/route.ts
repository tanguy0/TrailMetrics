/**
 * Proxy from this app's origin to the Python compute API.
 *
 * Every browser call goes through here, which buys three things:
 *
 *  - the session cookie stays first-party (no third-party-cookie problems, no CORS);
 *  - the API's URL and the service token never reach the client;
 *  - the session cookie is exchanged for an `Authorization` header server-side, so
 *    the token itself is never readable by JavaScript.
 */

import { NextRequest } from "next/server";

import { SESSION_COOKIE, VIEW_AS_COOKIE, apiBaseUrl } from "@/lib/session";

// Renders can take a while (a GAP fit is a real model fit), so allow well past the
// default. Vercel caps this by plan; the API's own timeouts are the real bound.
export const maxDuration = 120;

const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "transfer-encoding", "upgrade",
  "proxy-authenticate", "proxy-authorization", "te", "trailer",
  "content-encoding", "content-length",
]);

async function forward(request: NextRequest, path: string[]): Promise<Response> {
  const target = new URL(`${apiBaseUrl()}/${path.join("/")}`);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  headers.set("accept", request.headers.get("accept") ?? "application/json");

  const session = request.cookies.get(SESSION_COOKIE)?.value;
  if (session) headers.set("authorization", `Bearer ${session}`);

  // Only takes effect server-side if the *signed-in* athlete is a coach — see
  // api/deps.py's `current_athlete_id`. Forwarding it unconditionally is safe.
  const viewAs = request.cookies.get(VIEW_AS_COOKIE)?.value;
  if (viewAs) headers.set("x-view-as-athlete-id", viewAs);

  const hasBody = !["GET", "HEAD"].includes(request.method);
  let response: Response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers,
      // `arrayBuffer` rather than `text`: an image upload is multipart with binary
      // parts, and decoding it as text corrupts the bytes. JSON survives this
      // unchanged, so there is no reason to branch on the content type.
      body: hasBody ? await request.arrayBuffer() : undefined,
      // The proxy is the caching boundary; let the API decide freshness.
      cache: "no-store",
    });
  } catch (error) {
    // A cold or unreachable compute service is the most common local failure, so
    // say which one it is instead of surfacing an opaque 500.
    return Response.json(
      { detail: `Cannot reach the compute API: ${(error as Error).message}` },
      { status: 502 },
    );
  }

  const out = new Headers();
  response.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) out.set(key, value);
  });
  return new Response(response.body, { status: response.status, headers: out });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: Context) {
  return forward(request, (await context.params).path);
}
export async function POST(request: NextRequest, context: Context) {
  return forward(request, (await context.params).path);
}
export async function PUT(request: NextRequest, context: Context) {
  return forward(request, (await context.params).path);
}
export async function PATCH(request: NextRequest, context: Context) {
  return forward(request, (await context.params).path);
}
export async function DELETE(request: NextRequest, context: Context) {
  return forward(request, (await context.params).path);
}
