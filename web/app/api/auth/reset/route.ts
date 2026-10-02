/** Ask for a reset link. The answer does not say whether the address has an account. */

import { NextRequest } from "next/server";

import { forwardAuth, readJson } from "@/lib/authRoute";

export async function POST(request: NextRequest) {
  const { email } = await readJson(request);
  return forwardAuth(request, "/auth/reset", { email });
}
