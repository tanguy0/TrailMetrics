/**
 * One blog article: title, the author's text, the PDF-turned-carousel, then the
 * fixed signature. Public — no session required to read it.
 */

import { notFound } from "next/navigation";
import type { Metadata } from "next";

import { BlogCarousel } from "@/components/BlogCarousel";
import { BlogSignature } from "@/components/BlogSignature";
import { apiBaseUrl, lang, readSession } from "@/lib/session";
import type { BlogPost } from "@/lib/types";

async function fetchPost(slug: string): Promise<BlogPost | null> {
  const session = await readSession();
  const headers: Record<string, string> = {};
  if (session) headers.authorization = `Bearer ${session}`;
  try {
    const response = await fetch(
      `${apiBaseUrl()}/blog/${encodeURIComponent(slug)}?lang=${encodeURIComponent(await lang())}`,
      { headers, cache: "no-store" },
    );
    return response.ok ? ((await response.json()) as BlogPost) : null;
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const post = await fetchPost((await params).slug);
  return { title: post ? `${post.title} — TrailMetrics Blog` : "Blog — TrailMetrics" };
}

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const post = await fetchPost((await params).slug);
  if (!post) notFound();

  return (
    <main className="container container--narrow blog-post">
      {post.published === false && (
        <p className="note">Brouillon — non visible dans la liste publique.</p>
      )}
      <h1>{post.title}</h1>
      {post.body_text.split("\n\n").map((paragraph, i) => (
        <p key={i} style={{ whiteSpace: "pre-wrap" }}>
          {paragraph}
        </p>
      ))}

      <BlogCarousel pageUrls={post.page_urls} title={post.title} />

      <BlogSignature />
    </main>
  );
}
