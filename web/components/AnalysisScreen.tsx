"use client";

/**
 * Analysis: every analysis this athlete has, then a button to add one.
 *
 * One list, not two. The three analyses the product ships are seeded into the
 * athlete's own pages the first time this screen loads, so they sit in the same grid
 * as everything else and are edited the same way — the only thing that marks them is
 * a badge saying they cannot be deleted.
 *
 * They used to live in a separate "Examples" section, read-only, to be duplicated
 * before use. That was the wrong model: it made the Race Comparator unusable (it needs
 * a hand-picked selection, and a read-only page cannot be given one) and it asked
 * every athlete to make a copy of something before it could tell them anything.
 *
 * The "add" button sits *below* the grid rather than in the header, so the reading
 * order matches the order it is needed in: see what you have, then make another.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createPage, listPages } from "@/lib/api";
import { plural, translator, type Strings, type Translate } from "@/lib/strings";
import type { PageSummary } from "@/lib/types";

export function AnalysisScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const router = useRouter();

  const [pages, setPages] = useState<PageSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    try {
      // This call is also what seeds the defaults, server-side, on a fresh account.
      const { pages: listed } = await listPages();
      setPages(listed);
      setError(null);
    } catch (caught) {
      if (caught instanceof ApiError && caught.isUnauthorized) {
        router.push("/");
        return;
      }
      setError((caught as Error).message);
    }
  }, [router]);

  useEffect(() => {
    load();
  }, [load]);

  const newAnalysis = async () => {
    const name = window.prompt(t("pages.new.prompt"), t("pages.new.default_name"));
    if (!name) return;
    setCreating(true);
    try {
      const page = await createPage(name);
      router.push(`/pages/${page.id}`);
    } catch (caught) {
      setError((caught as Error).message);
      setCreating(false);
    }
  };

  if (error) {
    return (
      <main className="container">
        <p className="note note--error">{error}</p>
      </main>
    );
  }

  const steps = [
    { key: "step1", scale: "scale-1", icon: "🎯" },
    { key: "step2", scale: "scale-4", icon: "📈" },
    { key: "step3", scale: "scale-6", icon: "💾" },
  ] as const;

  return (
    <main className="container">
      <h1>{t("pages.title")}</h1>

      <section className="explainer">
        <h2 className="explainer__title">{t("pages.how.title")}</h2>
        <p className="explainer__lede">{t("pages.how.body")}</p>
        <div className="step-grid">
          {steps.map((step) => (
            <div className={`step ${step.scale}`} key={step.key}>
              <span className="step__icon" aria-hidden="true">{step.icon}</span>
              <h3 className="step__title">{t(`pages.how.${step.key}.title`)}</h3>
              <p className="step__body">{t(`pages.how.${step.key}.body`)}</p>
            </div>
          ))}
        </div>
      </section>

      {pages === null ? (
        <p className="muted">{t("common.loading")}</p>
      ) : (
        <div className="card-grid">
          {pages.map((page) => (
            <AnalysisCard key={page.id} page={page} t={t} />
          ))}
        </div>
      )}

      <button
        type="button"
        className="new-page"
        onClick={newAnalysis}
        disabled={creating}
      >
        <span className="new-page__plus" aria-hidden="true">+</span>
        <span className="new-page__text">
          <span className="new-page__label">{t("pages.new.button")}</span>
          <span className="new-page__hint">{t("pages.new.hint")}</span>
        </span>
      </button>
    </main>
  );
}

function AnalysisCard({ page, t }: { page: PageSummary; t: Translate }) {
  return (
    <a className="card" href={`/pages/${page.id}`}>
      <span className="card__icon">{page.icon}</span>
      <span className="card__title">{page.name}</span>
      <span className="card__meta">
        {plural(t, "pages.panel_count", page.panel_count)} ·{" "}
        {plural(t, "pages.plot_count", page.plot_count)}
        {/* Only says what is *unusual* about it — that it cannot be removed. Everything
            else about a default analysis is the same as any other. */}
        {page.is_default && (
          <>
            {" · "}
            <span className="card__badge">{t("page.default")}</span>
          </>
        )}
      </span>
      {page.description && <span className="card__description">{page.description}</span>}
    </a>
  );
}
