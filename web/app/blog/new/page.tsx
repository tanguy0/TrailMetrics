/**
 * Write a new article. Server-gated on `athlete.is_master`, like `welcome/page.tsx`
 * gates on session — the API enforces the same check independently on every write.
 */

import { redirect } from "next/navigation";
import type { Metadata } from "next";

import { BlogPostForm } from "@/components/BlogPostForm";
import { apiBaseUrl, lang, readSession } from "@/lib/session";
import type { Athlete } from "@/lib/types";

export const metadata: Metadata = { title: "Nouvel article — TrailMetrics Blog" };

export default async function NewBlogPostPage() {
  const session = await readSession();
  if (!session) redirect("/blog");

  const athlete = await fetchAthlete(session);
  if (!athlete?.is_master) redirect("/blog");

  return (
    <main className="container container--narrow">
      <h1>Nouvel article</h1>
      <BlogPostForm />
    </main>
  );
}

async function fetchAthlete(session: string): Promise<Athlete | null> {
  try {
    const response = await fetch(
      `${apiBaseUrl()}/auth/me?lang=${encodeURIComponent(await lang())}`,
      { headers: { authorization: `Bearer ${session}` }, cache: "no-store" },
    );
    return response.ok ? ((await response.json()) as Athlete) : null;
  } catch {
    return null;
  }
}
