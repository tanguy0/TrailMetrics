/**
 * Sign out: the API deletes the session, this drops the cookie.
 *
 * POST only, and same-origin only: a GET sign-out can be triggered by any page
 * that embeds an image pointing here.
 */

import { NextRequest } from "next/server";

import { signOut } from "@/lib/authRoute";

export async function POST(request: NextRequest) {
  return signOut(request, "/auth/logout");
}
