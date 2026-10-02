/** Sign in: email + password → session cookie. The previous session, if any, is revoked. */

import { NextRequest } from "next/server";

import { forwardAuth, readJson } from "@/lib/authRoute";
import { SESSION_COOKIE } from "@/lib/session";

export async function POST(request: NextRequest) {
  const { email, password } = await readJson(request);
  return forwardAuth(request, "/auth/login", {
    email,
    password,
    previous_token: request.cookies.get(SESSION_COOKIE)?.value ?? "",
  });
}
