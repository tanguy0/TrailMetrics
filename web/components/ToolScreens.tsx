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
import { useEffect, useState, type ReactNode } from "react";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { Recompute } from "@/components/Recompute";
import {
  getDurabilitySummary,
  getGapSummary,
  recomputeDurability,
  recomputeGap,
} from "@/lib/api";
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
        <div className="tm-level-grid">
          {(summary?.terrains ?? []).map((terrain) => (
            <AssessmentTile
              key={terrain.key}
              icon={TERRAIN_ICON[terrain.key]}
              label={t(`gap_tool.terrain.${terrain.key}`)}
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

/** Filled segments of a level's meter, poor → excellent (LevelTile.md). */
const LEVEL_BARS: Record<AssessmentLevel, number> = {
  poor: 1,
  limited: 2,
  average: 3,
  good: 4,
  excellent: 5,
  insufficient: 0,
};

/**
 * One terrain (or effort) of a profile, read in a glance: its pictogram, its
 * name, its level as a word, a colour and a five-segment meter (LevelTile.md).
 */
function AssessmentTile({
  icon,
  label,
  level,
  t,
}: {
  icon: ReactNode;
  label: string;
  level: AssessmentLevel;
  t: Translate;
}) {
  const word = t(`assessment.level.${level}`);
  return (
    <div className={`tm-level tm-level--${level}`} aria-label={`${label} : ${word}`}>
      <div className="tm-level__head">
        {icon}
        <span className="tm-level__label">{label}</span>
      </div>
      <div className="tm-level__word">{word}</div>
      <div className="tm-level__meter" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((i) => (
          <i key={i} className={i <= LEVEL_BARS[level] ? "is-on" : undefined} />
        ))}
      </div>
    </div>
  );
}

/** Lucide-style pictogram: 24 grid, 1.75 stroke, `currentColor`. */
function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

/** A slope as its profile: the steeper the terrain, the steeper the triangle. */
const TERRAIN_ICON: Record<string, ReactNode> = {
  steep_downhill: <Icon><path d="M4 4v16h11z" /></Icon>,
  downhill: <Icon><path d="M3 11v9h18z" /></Icon>,
  uphill: <Icon><path d="M21 11v9H3z" /></Icon>,
  steep_uphill: <Icon><path d="M20 4v16H9z" /></Icon>,
};

const QUALITY_ICON: Record<string, ReactNode> = {
  // A stopwatch: time on feet.
  long_efforts: (
    <Icon>
      <path d="M10 2h4" />
      <path d="M12 14l3-3" />
      <circle cx="12" cy="14" r="8" />
    </Icon>
  ),
  // A bolt: time above threshold.
  hard_efforts: <Icon><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></Icon>,
  descents: TERRAIN_ICON.downhill,
};

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
        <div className="tm-level-grid">
          {(summary?.qualities ?? []).map((quality) => (
            <AssessmentTile
              key={quality.key}
              icon={QUALITY_ICON[quality.key]}
              label={t(`durability_tool.quality.${quality.key}`)}
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
