/**
 * Edit an existing article, draft or published. Server-gated on `athlete.is_master`.
 *
 * Fetching the post with the session forwarded is what lets a draft resolve here
 * (see `api/routers/blog.py`'s `_is_master_request`) even though the same slug
 * 404s on the public `/blog/[slug]` page for anyone else.
 */

import { notFound, redirect } from "next/navigation";
import type { Metadata } from "next";

import { BlogPostForm } from "@/components/BlogPostForm";
import { apiBaseUrl, lang, readSession } from "@/lib/session";
import type { Athlete, BlogPost } from "@/lib/types";

export const metadata: Metadata = { title: "Modifier l'article — TrailMetrics Blog" };

export default async function EditBlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const session = await readSession();
  if (!session) redirect("/blog");

  const athlete = await fetchJson<Athlete>("/auth/me", session);
  if (!athlete?.is_master) redirect("/blog");

  const { slug } = await params;
  const post = await fetchJson<BlogPost>(`/blog/${encodeURIComponent(slug)}`, session);
  if (!post) notFound();

  return (
    <main className="container container--narrow">
      <h1>Modifier l&apos;article</h1>
      <BlogPostForm existing={post} />
    </main>
  );
}

async function fetchJson<T>(path: string, session: string): Promise<T | null> {
  try {
    const response = await fetch(
      `${apiBaseUrl()}${path}?lang=${encodeURIComponent(await lang())}`,
      { headers: { authorization: `Bearer ${session}` }, cache: "no-store" },
    );
    return response.ok ? ((await response.json()) as T) : null;
  } catch {
    return null;
  }
}
