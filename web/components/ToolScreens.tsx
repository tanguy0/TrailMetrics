"use client";

/**
 * Tools → Slope Profile and Durability (design/tagg/access.md § Outils).
 *
 * Built like Home, not like an analysis: a compact hero, the headline tiles, one
 * sentence with a `tm-hl`, then the chart. The charts are the existing
 * `gap_curve` and `durability_curve` plots, posted to `/render/panel` as a spec
 * — the same call the page builder makes — so they inherit the palette, the
 * render cache and every fix to those plots. The tiles come from
 * the tools router's `summary` routes, computed on the same fitted models the race plan uses.
 */

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { Kpi } from "@/components/Kpi";
import { getDurabilitySummary, getGapSummary, renderPanel } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { RUNNING_SPORT_TYPES } from "@/lib/sport";
import { chipClass, type ChipTone } from "@/lib/tone";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type {
  AssessmentLevel,
  ChartData,
  DurabilitySummary,
  GapSummary,
  PanelSpec,
} from "@/lib/types";

const YEAR_DAYS = 365;

function isoDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** The past year of runs — what the GAP and durability models are fitted on. */
function pastYear(name: string): PanelSpec["source"] {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - YEAR_DAYS);
  return {
    mode: "window",
    activity_ids: [],
    selection_label: "",
    windows: [{ name, start: isoDate(start), end: isoDate(end) }],
    filters: { sport_types: RUNNING_SPORT_TYPES, min_distance_km: null, max_distance_km: null },
  };
}

function gapPanel(t: Translate): PanelSpec {
  return {
    id: "panel_tool_gap",
    title: t("gap_tool.chart"),
    description: "",
    columns: 1,
    source: pastYear(t("gap_tool.chart")),
    plots: [{
      id: "plot_tool_gap",
      plot_type: "gap_curve",
      title: null,
      params: { models: ["efficiency"], references: ["balanced_runner"], show_std: false, hr_bands: [] },
    }],
  };
}

function durabilityPanel(t: Translate): PanelSpec {
  return {
    id: "panel_tool_durability",
    title: t("tools.durability"),
    description: "",
    columns: 1,
    source: pastYear(t("tools.durability")),
    plots: [{
      id: "plot_tool_durability",
      plot_type: "durability_curve",
      title: null,
      params: { lookback_days: YEAR_DAYS, min_run_minutes: 45, bin_minutes: 20, show_observed: true },
    }],
  };
}

/** The panel's charts and notes, rendered once; `null` while it computes. */
function usePanel(build: () => PanelSpec) {
  const [charts, setCharts] = useState<ChartData[] | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  useEffect(() => {
    let live = true;
    renderPanel(build())
      .then((result) => {
        if (!live) return;
        const outputs = result.panel.plots.map((plot) => plot.output).filter(Boolean);
        setCharts(outputs.flatMap((output) => output?.charts ?? []));
        setNotes(outputs.flatMap((output) => output?.notes ?? []));
      })
      .catch(() => live && setCharts([]));
    return () => {
      live = false;
    };
    // Built once per mount: the spec depends on today's date only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return { charts, notes };
}

function Charts({ charts, t }: { charts: ChartData[] | null; t: Translate }) {
  if (charts === null) {
    return (
      <div className="pending">
        <span className="spinner" />
        <p className="muted">{t("common.loading")}</p>
      </div>
    );
  }
  return (
    <>
      {charts.map((chart, index) => (
        <div className="chart-frame" key={index}>
          <ChartView chart={chart} />
        </div>
      ))}
    </>
  );
}

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

/** A sentence with its one highlighted value (Highlights.md): `{value}` → tm-hl. */
function Highlighted({ text, value }: { text: string; value: string }): ReactNode {
  const [before, after = ""] = text.split("{value}");
  return (
    <p className="body tool-sentence">
      {before}
      <span className="tm-hl">{value}</span>
      {after}
    </p>
  );
}

function signedPct(value: number): string {
  const rounded = formatNumber(Math.abs(value), 1);
  return `${value > 0 ? "+" : value < 0 ? "−" : ""}${rounded} %`;
}

export function GapScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [summary, setSummary] = useState<GapSummary | null>(null);
  const { charts, notes } = usePanel(() => gapPanel(t));

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
              level={terrain.level}
              t={t}
            />
          ))}
        </div>
        <Charts charts={charts} t={t} />
        {notes.slice(0, 1).map((note) => <p className="body-sm muted" key={note}>{note}</p>)}
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
  level,
  t,
}: {
  label: string;
  sub: string;
  level: AssessmentLevel;
  t: Translate;
}) {
  return (
    <div className="tm-kpi tm-kpi--flat assessment-tile">
      <span className="tm-kpi__label">{label}</span>
      <span className="assessment-tile__range">{sub}</span>
      <span className={chipClass(LEVEL_TONE[level], level === "excellent" ? "tm-chip--dot" : "")}>
        {t(`assessment.level.${level}`)}
      </span>
    </div>
  );
}

export function DurabilityScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [summary, setSummary] = useState<DurabilitySummary | null>(null);
  const { charts, notes } = usePanel(() => durabilityPanel(t));

  useEffect(() => {
    getDurabilitySummary().then(setSummary).catch(() => undefined);
  }, []);

  const at = (hours: string) => summary?.extra_cost_pct[hours];
  const four = at("4h");
  const population = summary?.population_extra_cost_pct["4h"];

  return (
    <main className="container tool">
      <ToolHero kicker={t("durability_tool.kicker")} title={t("durability_tool.title")} />

      <section className="card-block">
        <div className="kpi-grid kpi-grid--headline">
          <Kpi
            label={t("durability_tool.at", { hours: 2 })}
            value={at("2h") != null ? signedPct(at("2h") as number) : "—"}
          />
          <Kpi
            label={t("durability_tool.at", { hours: 4 })}
            value={four != null ? signedPct(four) : "—"}
            tone="terra"
          />
          <Kpi
            label={t("durability_tool.confidence")}
            value={summary ? t(`durability_tool.confidence.${summary.confidence}`) : "—"}
            note={summary ? t("durability_tool.runs", { count: summary.n_activities }) : null}
          />
        </div>
        {four != null && (
          <Highlighted
            text={`${t("durability_tool.sentence", { value: "{value}" })}${
              population != null && summary?.personal
                ? ` ${t("durability_tool.vs_population", { value: signedPct(population) })}`
                : ""
            }`}
            value={`${formatNumber(four, 1)} %`}
          />
        )}
        <Charts charts={charts} t={t} />
        {notes.slice(0, 2).map((note) => <p className="body-sm muted" key={note}>{note}</p>)}
      </section>
    </main>
  );
}
