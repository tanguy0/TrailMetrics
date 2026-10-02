/** Set a new password from a reset link; every other session is revoked, this one opens. */

import { NextRequest } from "next/server";

import { forwardAuth, readJson } from "@/lib/authRoute";

export async function POST(request: NextRequest) {
  const { token, password } = await readJson(request);
  return forwardAuth(request, "/auth/reset/confirm", { token, password });
}
