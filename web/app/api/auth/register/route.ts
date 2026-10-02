/** Create an account and sign it in, in the language this browser already uses. */

import { NextRequest } from "next/server";

import { forwardAuth, readJson } from "@/lib/authRoute";
import { LANG_COOKIE } from "@/lib/session";

export async function POST(request: NextRequest) {
  const { email, password } = await readJson(request);
  return forwardAuth(request, "/auth/register", {
    email,
    password,
    lang: request.cookies.get(LANG_COOKIE)?.value || process.env.NEXT_PUBLIC_LANG || "en",
  });
}
