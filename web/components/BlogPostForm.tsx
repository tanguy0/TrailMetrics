"use client";

/**
 * Write or edit one article. Shared by `/blog/new` and `/blog/[slug]/edit` — the
 * two differ only in whether `existing` is set and whether the PDF is required.
 *
 * Access control is not this component's job: both pages check `athlete.is_master`
 * server-side before rendering it, and the API re-checks on every write regardless
 * (`api/deps.py`'s `require_master`) — this is only ever reachable by someone who
 * would pass that check anyway.
 */

import { useState } from "react";
import { useRouter } from "next/navigation";

import { createBlogPost, deleteBlogPost, updateBlogPost } from "@/lib/api";
import type { BlogPost } from "@/lib/types";

export function BlogPostForm({ existing }: { existing?: BlogPost }) {
  const router = useRouter();
  const [title, setTitle] = useState(existing?.title ?? "");
  const [bodyText, setBodyText] = useState(existing?.body_text ?? "");
  const [slug, setSlug] = useState(existing?.slug ?? "");
  const [published, setPublished] = useState(existing?.published ?? true);
  const [pdf, setPdf] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!existing && !pdf) {
      setError("Choisissez un PDF — chaque page en devient un slide du carrousel.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const post = existing
        ? await updateBlogPost(existing.id, {
            title,
            body_text: bodyText,
            slug,
            published,
            ...(pdf ? { pdf } : {}),
          })
        : await createBlogPost({ title, body_text: bodyText, slug, published, pdf: pdf! });
      router.push(`/blog/${post.slug}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Échec de l'enregistrement.");
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!existing) return;
    if (!window.confirm(`Supprimer « ${existing.title} » définitivement ?`)) return;
    setBusy(true);
    try {
      await deleteBlogPost(existing.id);
      router.push("/blog");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Échec de la suppression.");
      setBusy(false);
    }
  };

  return (
    <form className="blog-form" onSubmit={submit}>
      <label>
        Titre
        <input value={title} onChange={(e) => setTitle(e.target.value)} required />
      </label>

      <label>
        Texte d&apos;explication
        <textarea
          value={bodyText}
          onChange={(e) => setBodyText(e.target.value)}
          rows={8}
        />
      </label>

      <label>
        Slug (URL) — laissez vide pour le générer depuis le titre
        <input value={slug} onChange={(e) => setSlug(e.target.value)} />
      </label>

      <label>
        PDF {existing ? "(laissez vide pour garder le carrousel actuel)" : ""}
        <input type="file" accept="application/pdf" onChange={(e) => setPdf(e.target.files?.[0] ?? null)} />
      </label>

      <label className="blog-form__checkbox">
        <input
          type="checkbox"
          checked={published}
          onChange={(e) => setPublished(e.target.checked)}
        />
        Publié (visible dans la liste publique)
      </label>

      {error && <p className="note note--error">{error}</p>}

      <div className="blog-form__actions">
        <button type="submit" className="button" disabled={busy}>
          {busy ? "Enregistrement…" : existing ? "Enregistrer" : "Publier"}
        </button>
        {existing && (
          <button
            type="button"
            className="button button--danger"
            onClick={remove}
            disabled={busy}
          >
            Supprimer
          </button>
        )}
      </div>
    </form>
  );
}
