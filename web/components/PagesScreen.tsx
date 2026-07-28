"use client";

/**
 * My Pages: how pages work, the built-in examples, then the athlete's own pages.
 *
 * The "New page" button sits at the bottom of the *My pages* section rather than in
 * a header, so the reading order matches the order someone actually needs it in:
 * understand the idea, see it done three times, then make one.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createPage, listBuiltinPages, listPages } from "@/lib/api";
import { plural, translator, type Strings, type Translate } from "@/lib/strings";
import type { PageSummary } from "@/lib/types";

export function PagesScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const router = useRouter();

  const [pages, setPages] = useState<PageSummary[] | null>(null);
  const [examples, setExamples] = useState<PageSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    try {
      const [mine, builtin] = await Promise.all([listPages(), listBuiltinPages()]);
      setPages(mine.pages);
      setExamples(builtin.pages);
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

  const newPage = async () => {
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
    { key: "step1", tone: "forest", icon: "🎯" },
    { key: "step2", tone: "terracotta", icon: "📈" },
    { key: "step3", tone: "sunrise", icon: "💾" },
  ] as const;

  return (
    <main className="container">
      <h1>{t("pages.title")}</h1>

      <section className="explainer">
        <h2 className="explainer__title">{t("pages.how.title")}</h2>
        <p className="explainer__lede">{t("pages.how.body")}</p>
        <div className="step-grid">
          {steps.map((step) => (
            <div className={`step step--${step.tone}`} key={step.key}>
              <span className="step__icon" aria-hidden="true">{step.icon}</span>
              <h3 className="step__title">{t(`pages.how.${step.key}.title`)}</h3>
              <p className="step__body">{t(`pages.how.${step.key}.body`)}</p>
            </div>
          ))}
        </div>
      </section>

      <h2>{t("pages.examples.title")}</h2>
      <p className="muted">{t("pages.examples.help")}</p>
      <div className="card-grid">
        {examples.map((page) => (
          <PageCard
            key={page.builtin_key}
            page={page}
            href={`/pages/builtin/${page.builtin_key}`}
            t={t}
          />
        ))}
      </div>

      <h2>{t("pages.mine.title")}</h2>
      {pages === null ? (
        <p className="muted">{t("common.loading")}</p>
      ) : pages.length ? (
        <div className="card-grid">
          {pages.map((page) => (
            <PageCard key={page.id} page={page} href={`/pages/${page.id}`} t={t} />
          ))}
        </div>
      ) : (
        <p className="muted">{t("pages.mine.empty")}</p>
      )}

      <button
        type="button"
        className="new-page"
        onClick={newPage}
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

function PageCard({
  page,
  href,
  t,
}: {
  page: PageSummary;
  href: string;
  t: Translate;
}) {
  return (
    <a className="card" href={href}>
      <span className="card__icon">{page.icon}</span>
      <span className="card__title">{page.name}</span>
      <span className="card__meta">
        {plural(t, "pages.panel_count", page.panel_count)} ·{" "}
        {plural(t, "pages.plot_count", page.plot_count)}
      </span>
      {page.description && <span className="card__description">{page.description}</span>}
    </a>
  );
}
