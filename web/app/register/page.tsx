/** Create an account. One already signed in goes straight on to where it was headed. */

import type { Metadata } from "next";
import { redirect } from "next/navigation";

import { AuthCard } from "@/components/AuthCard";
import { getViewer, safeNext } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "TAGG" };

export default async function RegisterPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>;
}) {
  const next = safeNext((await searchParams).next);
  if (await getViewer()) redirect(next);
  return (
    <main className="container auth-page">
      <AuthCard mode="register" next={next} strings={await loadStrings()} />
    </main>
  );
}
