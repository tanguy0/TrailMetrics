/**
 * The public blog index.
 *
 * A server component, like `welcome/page.tsx`: the session cookie (if any) is read
 * here so the master account sees drafts and a "New article" button, while every
 * other visitor — signed in or not — sees the same published list. Nothing here
 * requires a TrailMetrics account.
 */

import type { Metadata } from "next";

import { apiBaseUrl, lang, readSession } from "@/lib/session";
import type { Athlete, BlogPostSummary } from "@/lib/types";

export const metadata: Metadata = {
  title: "Blog — TrailMetrics",
};

async function fetchJson<T>(path: string, session: string | null): Promise<T | null> {
  try {
    const headers: Record<string, string> = {};
    if (session) headers.authorization = `Bearer ${session}`;
    const response = await fetch(
      `${apiBaseUrl()}${path}?lang=${encodeURIComponent(await lang())}`,
      { headers, cache: "no-store" },
    );
    return response.ok ? ((await response.json()) as T) : null;
  } catch {
    return null;
  }
}

export default async function BlogIndexPage() {
  const session = await readSession();
  const athlete = session ? await fetchJson<Athlete>("/auth/me", session) : null;
  const isMaster = athlete?.is_master ?? false;

  const data = await fetchJson<{ posts: BlogPostSummary[] }>(
    isMaster ? "/blog/admin" : "/blog",
    session,
  );
  const posts = data?.posts ?? [];

  return (
    <main className="container">
      <div className="blog-index__header">
        <h1>Blog</h1>
        {isMaster && (
          <a className="button" href="/blog/new">
            + Nouvel article
          </a>
        )}
      </div>

      {posts.length === 0 ? (
        <p className="muted">Aucun article pour l&apos;instant.</p>
      ) : (
        <div className="card-grid">
          {posts.map((post) => (
            <div key={post.id} className="card blog-card">
              <a className="blog-card__link" href={`/blog/${post.slug}`}>
                {post.cover_url && (
                  <img className="blog-card__cover" src={post.cover_url} alt="" />
                )}
                <div className="card__title">{post.title}</div>
                {isMaster && post.published === false && (
                  <span className="card__badge">Brouillon</span>
                )}
                <div className="card__description">{post.excerpt}</div>
              </a>
              {isMaster && (
                <a className="blog-card__edit" href={`/blog/${post.slug}/edit`}>
                  Modifier
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
