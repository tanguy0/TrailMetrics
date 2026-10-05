"use client";

/**
 * Tools → GAP Profile and Durability Profile (design/tagg/access.md § Outils).
 *
 * Built like Home, not like an analysis: a compact hero, then where the runner
 * stands against an average runner — one level per terrain (GAP) or quality
 * (durability), on the shared five-level scale — then the chart those levels
 * were read on. Levels and chart both come from the tools router's `summary`
 * routes, computed on the same fitted models the race plan uses.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { getDurabilitySummary, getGapSummary } from "@/lib/api";
import { chipClass, type ChipTone } from "@/lib/tone";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type { AssessmentLevel, DurabilitySummary, GapSummary } from "@/lib/types";

function ToolHero({ kicker, title }: { kicker: string; title: string }) {
  return (
    <header className="tm-hero tool-hero">
      <div className="tm-hero__body">
        <span className="tm-hero__kicker">{kicker}</span>
        <h1 className="tm-hero__title">{title}</h1>
      </div>
    </header>
  );
}

export function GapScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [summary, setSummary] = useState<GapSummary | null>(null);

  useEffect(() => {
    getGapSummary().then(setSummary).catch(() => setSummary({ available: false, terrains: [] }));
  }, []);

  return (
    <main className="container tool">
      <ToolHero kicker={t("gap_tool.kicker")} title={t("gap_tool.title")} />

      <section className="card-block">
        <p className="data-block__lede">{t("gap_tool.lede")}</p>
        {summary && !summary.available && summary.reason && <Callout>{summary.reason}</Callout>}
        <div className="kpi-grid">
          {(summary?.terrains ?? []).map((terrain) => (
            <AssessmentTile
              key={terrain.key}
              label={t(`gap_tool.terrain.${terrain.key}`)}
              sub={t(`gap_tool.range.${terrain.key}`)}
              numeric
              level={terrain.level}
              t={t}
            />
          ))}
        </div>
        {summary?.chart && (
          <>
            <h3 className="card-block__subtitle">{t("gap_tool.chart")}</h3>
            <div className="chart-frame">
              <ChartView chart={summary.chart} />
            </div>
          </>
        )}
        <p className="body-sm muted">
          {t("gap_tool.more")} <Link href="/pages">{t("nav.analysis")} →</Link>
        </p>
      </section>
    </main>
  );
}

/** Each level's chip tone (Chip.md): the alert red for poor, moss for the good side. */
const LEVEL_TONE: Record<AssessmentLevel, ChipTone> = {
  excellent: "moss",
  good: "moss",
  average: "forest",
  limited: "sun",
  poor: "danger",
  insufficient: "neutral",
};

/** One terrain (or effort) of a profile: its name, what it covers, its level. */
function AssessmentTile({
  label,
  sub,
  numeric = false,
  level,
  t,
}: {
  label: string;
  sub: string;
  /** `sub` is a range of numbers (a gradient), set in mono like every number. */
  numeric?: boolean;
  level: AssessmentLevel;
  t: Translate;
}) {
  return (
    <div className="tm-kpi tm-kpi--flat assessment-tile">
      <span className="tm-kpi__label">{label}</span>
      <span className={`assessment-tile__sub${numeric ? " is-num" : ""}`}>{sub}</span>
      <span className={chipClass(LEVEL_TONE[level], level === "excellent" ? "tm-chip--dot" : "")}>
        {t(`assessment.level.${level}`)}
      </span>
    </div>
  );
}

export function DurabilityScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [summary, setSummary] = useState<DurabilitySummary | null>(null);

  useEffect(() => {
    getDurabilitySummary().then(setSummary).catch(() => undefined);
  }, []);

  return (
    <main className="container tool">
      <ToolHero kicker={t("durability_tool.kicker")} title={t("durability_tool.title")} />

      <section className="card-block">
        <p className="data-block__lede">{t("durability_tool.lede")}</p>
        {summary && !summary.available && <Callout>{t("durability_tool.no_data")}</Callout>}
        <div className="kpi-grid">
          {(summary?.qualities ?? []).map((quality) => (
            <AssessmentTile
              key={quality.key}
              label={t(`durability_tool.quality.${quality.key}`)}
              sub={t(`durability_tool.scope.${quality.key}`)}
              level={quality.level}
              t={t}
            />
          ))}
        </div>
        {summary?.chart && (
          <>
            <h3 className="card-block__subtitle">{t("durability_tool.chart")}</h3>
            <div className="chart-frame">
              <ChartView chart={summary.chart} />
            </div>
          </>
        )}
      </section>
    </main>
  );
}
