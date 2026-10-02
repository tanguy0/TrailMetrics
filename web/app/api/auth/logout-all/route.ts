/** Sign out of every device: the API deletes all the account's sessions. */

import { NextRequest } from "next/server";

import { signOut } from "@/lib/authRoute";

export async function POST(request: NextRequest) {
  return signOut(request, "/auth/logout-all");
}
