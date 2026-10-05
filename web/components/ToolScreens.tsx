"use client";

/**
 * Tools → GAP Profile and Durability Profile (design/tagg/access.md § Outils).
 *
 * Built like Home, not like an analysis: a compact hero, then where the runner
 * stands against an average runner — one level per terrain (GAP) or quality
 * (durability), on the shared five-level scale — then the chart those levels
 * were read on. Levels and chart both come from the tools router's `summary`
 * routes, read on the same stored models the race plan uses — kept as last fitted
 * until the athlete presses Recompute.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { Recompute } from "@/components/Recompute";
import {
  getDurabilitySummary,
  getGapSummary,
  recomputeDurability,
  recomputeGap,
} from "@/lib/api";
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
  const [busy, setBusy] = useState(false);

  // Opening reads the stored fit; only a recompute says it is refitting.
  const load = (fetch: () => Promise<GapSummary>, refitting = false) => {
    setBusy(refitting);
    fetch()
      .then(setSummary)
      .catch(() =>
        setSummary((current) => current ?? { available: false, terrains: [], computed_at: null, new_runs: 0 }),
      )
      .finally(() => setBusy(false));
  };

  useEffect(() => load(getGapSummary), []);

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
        <ToolFooter summary={summary} busy={busy} onRecompute={() => load(recomputeGap, true)} t={t} />
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
  const [busy, setBusy] = useState(false);

  // Opening reads the stored fit; only a recompute says it is refitting.
  const load = (fetch: () => Promise<DurabilitySummary>, refitting = false) => {
    setBusy(refitting);
    fetch()
      .then(setSummary)
      .catch(() => undefined)
      .finally(() => setBusy(false));
  };

  useEffect(() => load(getDurabilitySummary), []);

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
        <ToolFooter
          summary={summary}
          busy={busy}
          onRecompute={() => load(recomputeDurability, true)}
          t={t}
        />
      </section>
    </main>
  );
}

/**
 * The end of a profile's card: how old the fit is with its Recompute button, and
 * the way to the full analysis panels.
 */
function ToolFooter({
  summary,
  busy,
  onRecompute,
  t,
}: {
  summary: { computed_at: string | null; new_runs: number } | null;
  busy: boolean;
  onRecompute: () => void;
  t: Translate;
}) {
  return (
    <div className="tool-more">
      <Recompute
        computedAt={summary?.computed_at}
        newRuns={summary?.new_runs}
        busy={busy}
        onRecompute={onRecompute}
        t={t}
      />
      <Link className="tm-btn tm-btn--secondary tm-btn--sm" href="/pages">
        {t("tools.more_details")}
      </Link>
    </div>
  );
}
