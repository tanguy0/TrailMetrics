"use client";

/**
 * Home: who you are, what your history adds up to, and how your form is trending.
 *
 * Two summary cards side by side — Profile (what you have run) and Health (what
 * you have told us) — then the latest activity, the last 30 weeks of volume, and the
 * same window of power-to-heart-rate.
 *
 * The charts are not bespoke plots. They are ordinary `metric_trend` panels posted
 * to `/render/panel`, the same call the page builder makes on every edit, drawn by
 * the same `ChartView`. That is the point of the architecture: a screen that wants a
 * chart assembles a spec rather than growing its own plotting code, so it inherits
 * the palette, the axis handling and the render cache for free.
 *
 * This screen is also where the app's two background passes are kicked off — the
 * Strava import and the model precompute — because it is the first thing an athlete
 * opens. Both are started here rather than at sign-in so they also run for a
 * returning session, which never passes through the OAuth callback again.
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { EmailForm } from "@/components/EmailForm";
import { Icon, type IconName } from "@/components/Icon";
import { ProgressBar } from "@/components/ProgressBar";
import { SessionDetail } from "@/components/SessionDetail";
import { Sparkline } from "@/components/Sparkline";
import {
  ApiError,
  getAthlete,
  getHomeSummary,
  getSyncStatus,
  renderPanel,
  startPrecompute,
  startSync,
  updateProfile,
} from "@/lib/api";
import {
  formatDate, formatHms, formatNumber, formatPaceInput, parsePaceInput,
} from "@/lib/format";
import { RUNNING_SPORT_TYPES } from "@/lib/sport";
import { TREND_TONE, chipClass } from "@/lib/tone";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type {
  ActivityCard,
  Athlete,
  ChartData,
  HomeRecord,
  HomeSummary,
  PanelSpec,
  Trace,
} from "@/lib/types";

const POLL_MS = 2000;
const WEEKS_SHOWN = 20;

/**
 * How stale the last import has to be before opening the app triggers another.
 *
 * Not zero. An incremental sync still walks the whole Strava activity listing, which
 * costs requests against a 100-per-15-minutes budget shared with everything else the
 * app does — so syncing on every visit to this screen would spend that budget on
 * navigation. Fifteen minutes is far shorter than the gap between a run finishing and
 * someone opening this page, so in practice the new activity is always there.
 */
const AUTO_SYNC_STALE_MS = 15 * 60 * 1000;

type T = Translate;

export function HomeScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const router = useRouter();

  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [summary, setSummary] = useState<HomeSummary | null>(null);
  const [volumeCharts, setVolumeCharts] = useState<ChartData[] | null>(null);
  const [efficiencyCharts, setEfficiencyCharts] = useState<ChartData[] | null>(null);
  const [formCharts, setFormCharts] = useState<ChartData[] | null>(null);
  const [feelCharts, setFeelCharts] = useState<ChartData[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Whether this mount has already fired the automatic passes. A ref, not state:
  // it must not cause a re-render, and a state update would land after the effect
  // it is meant to guard has already run again.
  const autoStarted = useRef(false);

  const load = useCallback(async () => {
    try {
      const [me, home] = await Promise.all([getAthlete(), getHomeSummary()]);
      setAthlete(me);
      setSummary(home);
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

  const activityCount = summary?.profile.activity_count ?? 0;

  // The charts are separate, slower requests; they must not hold up the cards, and
  // the two panels are fetched independently so a slow model fit in one does not
  // delay the other.
  useEffect(() => {
    if (!activityCount) {
      setVolumeCharts(null);
      setEfficiencyCharts(null);
      setFormCharts(null);
      setFeelCharts(null);
      return;
    }
    let live = true;
    const load = (panel: PanelSpec, set: (charts: ChartData[]) => void) =>
      renderPanel(panel)
        .then((result) => {
          if (!live) return;
          set(result.panel.plots.flatMap((plot) => plot.output?.charts ?? []));
        })
        // A failed chart leaves the rest of the screen intact; the cards are the
        // part that matters, and the panel error is not actionable here.
        .catch(() => live && set([]));

    load(recentHistoryPanel(t), setVolumeCharts);
    load(recentEfficiencyPanel(t), setEfficiencyCharts);
    load(recentFormPanel(t), setFormCharts);
    load(recentFeelPanel(t), setFeelCharts);
    return () => {
      live = false;
    };
    // Keyed on the activity count alone: `t` is rebuilt every render but its
    // strings never change, so including it would re-render the charts endlessly.
  }, [activityCount]);

  const syncing = athlete?.sync.status === "running";
  useEffect(() => {
    if (!syncing) return;
    const timer = setInterval(async () => {
      try {
        const status = await getSyncStatus();
        setAthlete((current) => (current ? { ...current, sync: status } : current));
        if (status.status !== "running") load();
      } catch {
        /* a transient failure shouldn't stop the poll */
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [syncing, load]);

  const importActivities = useCallback(async (force: boolean) => {
    setBusy(true);
    try {
      await startSync({ force });
      await load();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }, [load]);

  /**
   * Import new activities and warm the models, without being asked.
   *
   * Runs once per mount, as soon as the athlete's state is known. Both passes are
   * server-side background jobs that report progress through their own endpoints, so
   * this only has to start them — and both refuse a second concurrent run, which is
   * what makes firing them on every visit safe.
   */
  useEffect(() => {
    if (!athlete || autoStarted.current) return;
    autoStarted.current = true;

    const lastSynced = athlete.sync.last_synced_at
      ? Date.parse(athlete.sync.last_synced_at)
      : null;
    const stale =
      lastSynced === null || Date.now() - lastSynced > AUTO_SYNC_STALE_MS;

    (async () => {
      // A coach viewing another athlete's account can't trigger their Strava
      // import (the API refuses it, see api/routers/activities.py) — skip firing
      // a request that can only fail.
      if (!athlete.viewing_as && athlete.sync.status !== "running" && stale) {
        // A failed automatic import must not shout: the manual buttons are right
        // there, and this attempt was not something the athlete asked for.
        await startSync({}).catch(() => undefined);
        await load().catch(() => undefined);
      }
      // Fitting the GAP models is the slow one, so it starts regardless of whether
      // the import ran: with nothing new to compute the pass finds everything cached
      // and finishes immediately.
      //
      // Started here but *reported* on the page that shows those curves, not here.
      // Progress on a screen with no GAP curve on it is noise — the reader cannot act
      // on it and it is not what they came for.
      await startPrecompute().catch(() => undefined);
    })();
  }, [athlete, load]);

  if (error) {
    return (
      <main className="container">
        <Callout tone="terra">{error}</Callout>
      </main>
    );
  }
  if (!athlete || !summary) {
    return (
      <main className="container">
        <p className="muted">{t("common.loading")}</p>
      </main>
    );
  }

  return (
    <main className="container">
      <HomeHero
        athlete={athlete}
        summary={summary}
        volumeCharts={volumeCharts}
        formCharts={formCharts}
        busy={busy}
        onImport={() => importActivities(false)}
        email={
          <EditableTile
            label={t("home.health.email")}
            value={athlete.email}
            input={{ type: "email", value: athlete.email ?? "" }}
            onCommit={async (raw) =>
              setAthlete(await updateProfile({ email: raw.trim() === "" ? null : raw.trim() }))
            }
            t={t}
          />
        }
        t={t}
      />

      {/* Someone who reached this screen without answering the email question —
          an account created before it existed, or a skipped `/welcome`. */}
      {athlete.needs_email && (
        <section className="card-block card-block--welcome">
          <SectionTitle icon="mail" kicker={t("home.kicker.you")} role="terra">{t("email.missing")}</SectionTitle>
          <p className="muted">{t("email.body")}</p>
          <EmailForm
            strings={strings}
            submitLabel={t("email.provide")}
            onSaved={(email) =>
              setAthlete((current) =>
                current ? { ...current, email, needs_email: false } : current,
              )
            }
          />
        </section>
      )}

      <div className="home-grid">
        <ProfileCard summary={summary} volumeCharts={volumeCharts} t={t} />
        <HealthCard
          athlete={athlete}
          summary={summary}
          onSaved={setAthlete}
          t={t}
        />
      </div>

      <ZonesCard athlete={athlete} onSaved={setAthlete} t={t} />

      <RecordsCard records={summary.records} t={t} />

      {/* Last Run: the import controls, then the most recent activity itself. */}
      <section className="card-block card-block--sync">
        <SectionTitle icon="refresh" kicker={summary.last_activity ? formatDate(summary.last_activity.date) : null}>{t("home.last.title")}</SectionTitle>

        <SyncControls
          athlete={athlete}
          busy={busy}
          onImport={importActivities}
          viewingAs={athlete.viewing_as}
          t={t}
        />

        <div className="data-stack">
          <LastActivityBlock
            activity={summary.last_activity}
            t={t}
          />
        </div>
      </section>

      {/* Recent progress, one card per question (contrast.md § 1-4): what have
          I done, how efficient am I, am I in form, how do I feel. Each card has
          its own colour role and one headline tile, read off the last value of
          the series its chart already draws — no extra request. */}
      <RecentHistoryCard charts={volumeCharts} hasData={activityCount > 0} t={t} />
      <RecentEfficiencyCard
        charts={efficiencyCharts}
        hasData={activityCount > 0}
        hasWeight={athlete.weight_kg != null}
        t={t}
      />
      <RecentFormCard charts={formCharts} hasData={activityCount > 0} t={t} />
      <RecentFeelCard charts={feelCharts} hasData={activityCount > 0} t={t} />
    </main>
  );
}

/**
 * The Recent History panel: one `metric_trend` carrying both metrics.
 *
 * Distance as bars against the left axis, climb as a line against the right one —
 * a single plot spec, so this screen inherits the dual-axis handling rather than
 * owning any of it.
 */
function recentHistoryPanel(t: T): PanelSpec {
  return {
    id: "panel_home_recent",
    title: t("home.recent.title"),
    description: "",
    columns: 1,
    source: recentWindow(t("home.recent.title")),
    plots: [
      {
        id: "plot_home_volume",
        plot_type: "metric_trend",
        title: null,
        params: {
          metric: "distance_km",
          aggregation: "sum",
          // The second metric, drawn against its own right-hand axis.
          metric2: "elevation_gain_m",
          aggregation2: "sum",
          chart: "bar",
          chart2: "line",
          granularity: "week",
          x_mode: "calendar",
          cumulative: false,
          markers: true,
          split_by: "none",
          show_totals: false,
        },
      },
    ],
  };
}

/**
 * The Recent Efficiency panel: power-to-heart-rate per week.
 *
 * The one number on this screen that is about *fitness* rather than volume. Power is
 * modelled from speed and gradient, so the ratio says how much mechanical output the
 * athlete produced per heartbeat — it rises when the same effort buys more pace, and
 * it is insensitive to whether the week was hilly or flat, which raw pace is not.
 *
 * Aggregated as a mean over each week, then smoothed with a five-week rolling
 * average followed by a five-point Savitzky–Golay filter. Week to week the ratio is
 * noisy for reasons that have nothing to do with form — a hot day, a hard session, a
 * watch that lost heart rate for ten minutes — so the unsmoothed series invites
 * reading a bad Tuesday as lost fitness. The trend is the signal.
 */
function recentEfficiencyPanel(t: T): PanelSpec {
  return {
    id: "panel_home_efficiency",
    title: t("home.efficiency.title"),
    description: "",
    columns: 1,
    source: recentWindow(t("home.efficiency.title")),
    plots: [
      {
        id: "plot_home_efficiency",
        plot_type: "metric_trend",
        title: null,
        params: {
          metric: "power_to_hr",
          aggregation: "mean",
          metric2: "none",
          chart: "line",
          granularity: "week",
          x_mode: "calendar",
          cumulative: false,
          markers: true,
          split_by: "none",
          smooth_rolling: 5,
          smooth_savgol: 5,
          show_totals: false,
        },
      },
    ],
  };
}

/**
 * The Recent Form panel: the Banister fitness/fatigue model over the same window.
 *
 * Reuses the standalone `fitness_fatigue` plot type as-is (it takes no params and
 * draws both curves together) rather than re-deriving the same two series through
 * `metric_trend`, so this screen and the page builder can never disagree on what
 * "fitness" and "fatigue" mean.
 */
function recentFormPanel(t: T): PanelSpec {
  return {
    id: "panel_home_form",
    title: t("home.form.title"),
    description: "",
    columns: 1,
    source: recentWindow(t("home.form.title")),
    plots: [
      {
        id: "plot_home_form",
        plot_type: "fitness_fatigue",
        title: null,
        params: {},
      },
    ],
  };
}

/**
 * The Effort & Feel panel: the athlete's own weekly ratings, `weekly_feel`.
 *
 * The one chart on this screen that plots something the athlete *typed* rather
 * than something Strava measured — average RPE per week, the week's average
 * feeling as a background colour, and the fitness trend as the same tag the week
 * summary shows in Training. Like `fitness_fatigue` it takes no params and reads
 * the whole cross-sport history, so the window below only decides what is shown.
 */
function recentFeelPanel(t: T): PanelSpec {
  return {
    id: "panel_home_feel",
    title: t("home.feel.title"),
    description: "",
    columns: 1,
    source: recentWindow(t("home.feel.title"), WEEKS_SHOWN),
    plots: [
      {
        id: "plot_home_feel",
        plot_type: "weekly_feel",
        title: null,
        params: {},
      },
    ],
  };
}

/** The last `weeks` weeks as a single named window. */
function recentWindow(name: string, weeks: number = WEEKS_SHOWN): PanelSpec["source"] {
  const end = new Date();
  const start = new Date(end);
  // Inclusive of the current week, so the window spans exactly `weeks` weeks.
  start.setDate(start.getDate() - (weeks * 7 - 1));
  return {
    mode: "window",
    activity_ids: [],
    selection_label: "",
    windows: [{ name, start: isoDate(start), end: isoDate(end) }],
    // Explicit, not "every sport": Home is running-only even though cycling is
    // now imported too — see api/routers/home.py's `summary()` docstring.
    filters: {
      sport_types: RUNNING_SPORT_TYPES, min_distance_km: null, max_distance_km: null,
    },
  };
}

/** Local calendar date as `YYYY-MM-DD` — `toISOString` would shift the day in UTC-. */
function isoDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// --- Profile ---------------------------------------------------------------

/**
 * A section's colour role (contrast.md § 4): its kicker and icon take it, nothing
 * else in the heading does. `forest` is the default and needs no modifier.
 */
type SectionRole = "forest" | "terra" | "sun" | "moss";

/**
 * The heading of a Home card: a mono kicker (the time window, or "You") over the
 * title, both beside an icon in the section's role colour.
 */
function SectionTitle({
  icon,
  kicker,
  role = "forest",
  children,
}: {
  icon: IconName;
  kicker?: string | null;
  role?: SectionRole;
  children: ReactNode;
}) {
  return (
    <h2 className={`tm-section${role === "forest" ? "" : ` tm-section--${role}`} section-title`}>
      <Icon name={icon} size={18} />
      <span className="section-title__text">
        {kicker && <span className="tm-section__kicker">{kicker}</span>}
        {children}
      </span>
    </h2>
  );
}

/** Weeks of history in the Total distance sparkline (contrast.md § 3). */
const SPARK_WEEKS = 13;

function ProfileCard({
  summary,
  volumeCharts,
  t,
}: {
  summary: HomeSummary;
  volumeCharts: ChartData[] | null;
  t: T;
}) {
  const { profile } = summary;
  const weekly = volumeTraces(volumeCharts).distance?.y.slice(-SPARK_WEEKS);
  return (
    <section className="card-block card-block--profile">
      <SectionTitle icon="run" kicker={t("home.kicker.all_time")}>{t("home.profile.title")}</SectionTitle>

      <div className="kpi-grid kpi-grid--four">
        <Tile
          label={t("home.profile.activities")}
          value={String(profile.activity_count)}
        />
        <Tile
          label={t("home.profile.total_distance")}
          value={formatNumber(profile.total_distance_m / 1000, 0)}
          unit={t("common.km")}
          tone="forest"
          spark={weekly}
        />
        <Tile
          label={t("home.profile.total_elevation")}
          value={formatNumber(profile.total_elevation_gain_m, 0)}
          unit={t("common.metres")}
        />
        <Tile
          label={t("home.profile.total_time")}
          value={formatNumber(profile.total_moving_s / 3600, 0)}
          unit={t("common.hours")}
        />
        <Tile
          label={t("home.profile.oldest")}
          value={formatDate(profile.oldest_activity)}
        />
        <Tile
          label={t("home.profile.newest")}
          value={formatDate(profile.newest_activity)}
        />
        <Tile
          label={t("home.profile.furthest")}
          value={
            profile.furthest_activity?.distance_m != null
              ? formatNumber(profile.furthest_activity.distance_m / 1000, 1)
              : "—"
          }
          unit={t("common.km")}
          footnote={formatDate(profile.furthest_activity?.date ?? null)}
        />
        <Tile
          label={t("home.profile.longest")}
          value={formatHms(profile.longest_activity?.moving_s)}
          footnote={formatDate(profile.longest_activity?.date ?? null)}
        />
      </div>

    </section>
  );
}

/**
 * Records across the full width, one tile per distance.
 *
 * Its own block rather than a list inside Profile: these are the numbers an athlete
 * actually looks for, and a four-across row of them reads at a glance where a
 * stacked list in a half-width column does not.
 */
function RecordsCard({ records, t }: { records: HomeRecord[]; t: T }) {
  const newest = newestRecord(records);
  return (
    <section className="card-block card-block--records">
      <SectionTitle icon="award" kicker={t("home.kicker.all_time")} role="sun">{t("home.profile.records")}</SectionTitle>
      {records.length ? (
        <div className="kpi-grid kpi-grid--records">
          {records.map((record) => (
            <div className="tm-kpi tm-kpi--flat" key={record.label}>
              <span className="tm-kpi__label">{record.label}</span>
              <span className="tm-kpi__value">
                <span className="tm-kpi__num">{formatHms(record.seconds)}</span>
              </span>
              <span className="tm-kpi__delta">
                {formatDate(record.set_on)}
                {record === newest && (
                  <span className={chipClass("sun", "records__new")}>{t("home.records.new")}</span>
                )}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">{t("home.profile.records_empty")}</p>
      )}
    </section>
  );
}

/** Within this long, the most recent record still reads as "new". */
const NEW_RECORD_DAYS = 30;

/**
 * The most recently set record, if it is recent enough to call new — the one
 * tile of the card that gets the sun pill. Every other record stays neutral.
 */
function newestRecord(records: HomeRecord[]): HomeRecord | null {
  let newest: HomeRecord | null = null;
  for (const record of records) {
    if (record.set_on && (!newest?.set_on || record.set_on > newest.set_on)) newest = record;
  }
  if (!newest?.set_on) return null;
  const age = (Date.now() - Date.parse(newest.set_on)) / 86_400_000;
  return age <= NEW_RECORD_DAYS ? newest : null;
}

// --- Health ----------------------------------------------------------------

function HealthCard({
  athlete,
  summary,
  onSaved,
  t,
}: {
  athlete: Athlete;
  summary: HomeSummary;
  onSaved: (athlete: Athlete) => void;
  t: T;
}) {
  const experience = summary.health.experience_years;

  return (
    <section className="card-block card-block--health">
      <SectionTitle icon="heart" kicker={t("home.kicker.you")} role="terra">{t("home.health.title")}</SectionTitle>

      <div className="kpi-grid kpi-grid--square">
        <EditableTile
          label={t("home.health.age")}
          value={athlete.age != null ? String(athlete.age) : null}
          unit={athlete.age != null ? t("common.years") : undefined}
          input={{
            type: "date",
            value: athlete.birthdate ?? "",
            // A birthdate in the future, or implying an age over 120, is a typo.
            min: "1900-01-01",
            max: new Date().toISOString().slice(0, 10),
          }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ birthdate: raw === "" ? null : raw }))
          }
          t={t}
        />

        <Tile
          label={t("home.health.experience")}
          value={experience != null ? formatNumber(experience, 1) : "—"}
          unit={experience != null ? t("common.years") : undefined}
        />

        <EditableTile
          label={t("home.health.weight")}
          value={athlete.weight_kg != null ? formatNumber(athlete.weight_kg, 1) : null}
          unit={athlete.weight_kg != null ? t("common.kg") : undefined}
          help={t("home.health.weight_help")}
          tone="terra"
          input={{ type: "number", value: athlete.weight_kg?.toString() ?? "", min: 25, max: 250, step: 0.5 }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ weight_kg: raw === "" ? null : Number(raw) }))
          }
          t={t}
        />

        <EditableTile
          label={t("home.health.height")}
          value={athlete.height_cm != null ? formatNumber(athlete.height_cm, 0) : null}
          unit={athlete.height_cm != null ? t("common.cm") : undefined}
          input={{ type: "number", value: athlete.height_cm?.toString() ?? "", min: 100, max: 250, step: 1 }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ height_cm: raw === "" ? null : Number(raw) }))
          }
          t={t}
        />
      </div>
    </section>
  );
}

/**
 * Training zones and VMA pace — self-reported, shown back to the athlete, and
 * read by nothing else in the app. A reference to have written down in one
 * place, not an input to any computation.
 */
/** Reference paces at a %VMA range, in the order (and table) most training
 * plans quote them. The pace shown for the low end of the range comes first —
 * lower %VMA is the slower pace. */
const VMA_PACE_ZONES: { key: string; lowPct: number; highPct: number }[] = [
  { key: "z2", lowPct: 60, highPct: 65 },
  { key: "endurance", lowPct: 70, highPct: 75 },
  { key: "threshold", lowPct: 85, highPct: 90 },
  { key: "intervals", lowPct: 95, highPct: 100 },
  { key: "reps", lowPct: 105, highPct: 115 },
];

function vmaPaceRange(vmaSecondsPerKm: number, lowPct: number, highPct: number): string {
  const slow = vmaSecondsPerKm / (lowPct / 100);
  const fast = vmaSecondsPerKm / (highPct / 100);
  return `${formatPaceInput(slow)}–${formatPaceInput(fast)}`;
}

/**
 * Each heart-rate zone's ceiling as a %HRmax — replaces what used to be four
 * separately self-reported bpm values with one derived from HRmax alone, so
 * there is only ever one number to keep up to date.
 */
const HR_ZONE_MAX_PCT: { key: "z1" | "z2" | "z3" | "z4"; pct: number }[] = [
  { key: "z1", pct: 0.70 },
  { key: "z2", pct: 0.77 },
  { key: "z3", pct: 0.87 },
  { key: "z4", pct: 0.91 },
];

/** Where each named pace zone's effort sits on the same %HRmax scale — the
 * boundaries `HrZoneMap` draws, kept beside the tiles that read from it. */
const HR_PACE_ZONES: { key: string; lowPct: number; highPct: number }[] = [
  { key: "z2", lowPct: 68, highPct: 73 },
  { key: "endurance", lowPct: 75, highPct: 81 },
  { key: "threshold", lowPct: 84, highPct: 89 },
  { key: "intervals", lowPct: 91, highPct: 94 },
  { key: "reps", lowPct: 96, highPct: 100 },
];

/** `pct` as a 0–1 fraction of HRmax, truncated like a real monitor reads bpm. */
function bpmAtPct(hrMax: number, pct: number): number {
  return Math.trunc(hrMax * pct);
}

function ZonesCard({
  athlete,
  onSaved,
  t,
}: {
  athlete: Athlete;
  onSaved: (athlete: Athlete) => void;
  t: T;
}) {
  const vma = athlete.vma_pace_s_per_km;
  const hrMax = athlete.hr_max;

  return (
    <section className="card-block card-block--zones">
      <SectionTitle icon="target" kicker={t("home.kicker.you")} role="terra">{t("home.zones.title")}</SectionTitle>
      <p className="data-block__lede">{t("home.zones.subtitle")}</p>

      <div className="kpi-grid">
        <EditableTile
          label={t("home.zones.vma")}
          value={vma != null ? formatPaceInput(vma) : null}
          unit={vma != null ? "/km" : undefined}
          input={{ type: "text", value: formatPaceInput(vma), placeholder: "4:00" }}
          onCommit={async (raw) => {
            if (raw.trim() === "") {
              onSaved(await updateProfile({ vma_pace_s_per_km: null }));
              return;
            }
            const parsed = parsePaceInput(raw);
            if (parsed == null) throw new Error("invalid pace");
            onSaved(await updateProfile({ vma_pace_s_per_km: parsed }));
          }}
          t={t}
        />

        {VMA_PACE_ZONES.map((zone) => (
          <Tile
            key={zone.key}
            label={t(`home.zones.pace_${zone.key}`)}
            value={vma != null ? vmaPaceRange(vma, zone.lowPct, zone.highPct) : "—"}
            unit={vma != null ? "/km" : undefined}
            footnote={t("home.zones.unlocked_by_vma")}
          />
        ))}
      </div>

      <div className="kpi-grid kpi-grid--two">
        {HR_ZONE_MAX_PCT.map((zone) => (
          <Tile
            key={zone.key}
            label={t(`home.zones.${zone.key}`)}
            value={hrMax != null ? String(bpmAtPct(hrMax, zone.pct)) : "—"}
            unit={hrMax != null ? "bpm" : undefined}
            footnote={t("home.zones.unlocked_by_hrmax")}
          />
        ))}
        <EditableTile
          label={t("home.zones.hr_max")}
          value={hrMax != null ? String(hrMax) : null}
          unit={hrMax != null ? "bpm" : undefined}
          input={{ type: "number", value: hrMax?.toString() ?? "", min: 30, max: 250, step: 1 }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ hr_max: raw === "" ? null : Number(raw) }))
          }
          t={t}
        />
      </div>

      <HrZoneMap hrMax={hrMax} t={t} />
    </section>
  );
}

/** The map's visible window: below 60% is all Z1 and none of the pace zones
 * ever reach it, so cropping there gives the part that matters the whole
 * width instead of shrinking it into a corner. */
const HR_MAP_MIN_PCT = 60;
const HR_MAP_MAX_PCT = 100;

function hrMapPosition(pct: number): number {
  return ((pct - HR_MAP_MIN_PCT) / (HR_MAP_MAX_PCT - HR_MAP_MIN_PCT)) * 100;
}

/** The heart-rate zone bands the map's background shows — derived from
 * `HR_ZONE_MAX_PCT` itself (plus the open-ended top zone above Z4max) so the
 * tiles and the graph can never drift out of step with each other again. */
const HR_MAP_BANDS: { key: string; label: string; endPct: number }[] = [
  ...HR_ZONE_MAX_PCT.map((zone) => ({
    key: zone.key,
    label: zone.key.toUpperCase(),
    endPct: zone.pct * 100,
  })),
  { key: "z5", label: "Z5", endPct: 100 },
];

/**
 * Where each named pace zone's effort falls in heart rate — a picture, not
 * another table, so the relationship between the two KPI grids above reads
 * at a glance instead of being cross-referenced by hand.
 */
function HrZoneMap({ hrMax, t }: { hrMax: number | null; t: T }) {
  return (
    <div className="hr-map">
      <h3 className="card-block__subtitle">{t("home.zones.hr_map_title")}</h3>
      {hrMax == null ? (
        <p className="muted">{t("home.zones.hr_map_needs_hrmax")}</p>
      ) : (
        <div className="hr-map__scroll">
          <div className="hr-map__chart">
            <div className="hr-map__zones">
              {HR_MAP_BANDS.map((band, index) => {
                const startPct = index === 0 ? HR_MAP_MIN_PCT : HR_MAP_BANDS[index - 1].endPct;
                const left = hrMapPosition(startPct);
                return (
                  <div
                    key={band.key}
                    className={`hr-map__zone hr-map__zone--${band.key}`}
                    style={{ left: `${left}%`, width: `${hrMapPosition(band.endPct) - left}%` }}
                  >
                    <span className="hr-map__zone-label">{band.label}</span>
                  </div>
                );
              })}
            </div>
            <div className="hr-map__paces">
              {HR_PACE_ZONES.map((zone) => {
                const left = hrMapPosition(zone.lowPct);
                return (
                  <div
                    key={zone.key}
                    className="hr-map__pace"
                    style={{ left: `${left}%`, width: `${hrMapPosition(zone.highPct) - left}%` }}
                  >
                    <span className="hr-map__pace-label">
                      {t(`home.zones.pace_${zone.key}`)}
                    </span>
                    <span className="hr-map__pace-range">
                      {bpmAtPct(hrMax, zone.lowPct / 100)}–{bpmAtPct(hrMax, zone.highPct / 100)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Latest activity and weekly volume -------------------------------------

/** The latest activity: its numbers, its route, and its pace/GAP/HR traces. */
function LastActivityBlock({
  activity,
  t,
}: {
  activity: ActivityCard | null;
  t: T;
}) {
  return (
    <div className="data-block">
      {activity ? (
        <SessionDetail activity={activity} t={t} />
      ) : (
        <p className="muted">{t("home.last.empty")}</p>
      )}
    </div>
  );
}

// --- Hero ------------------------------------------------------------------

/** Monday of the current week, local time — the bin `granularity: "week"` opens. */
function currentWeekStart(): Date {
  const date = new Date();
  date.setHours(0, 0, 0, 0);
  date.setDate(date.getDate() - ((date.getDay() + 6) % 7));
  return date;
}

/** ISO 8601 week number — the "Semaine N" the hero's kicker names. */
function isoWeekNumber(date: Date): number {
  const day = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  day.setUTCDate(day.getUTCDate() + 4 - (day.getUTCDay() || 7));
  const yearStart = Date.UTC(day.getUTCFullYear(), 0, 1);
  return Math.ceil(((day.getTime() - yearStart) / 86_400_000 + 1) / 7);
}

/** The last non-missing value of a series, or null when it has none. */
function lastValue(values: (number | null)[] | undefined): number | null {
  if (!values) return null;
  for (let i = values.length - 1; i >= 0; i--) {
    const value = values[i];
    if (value != null && !Number.isNaN(value)) return value;
  }
  return null;
}

/**
 * A weekly trace's value for the current week. A week with no activity has no
 * bin at all, so a missing current bin reads as zero, not as unknown.
 */
function currentWeekValue(trace: Trace | undefined): number | null {
  if (!trace) return null;
  const monday = isoDate(currentWeekStart());
  const index = trace.x.findIndex((x) => String(x).slice(0, 10) === monday);
  return index < 0 ? 0 : (trace.y[index] ?? 0);
}

/** The volume chart's distance (left axis) and climb (right axis) traces. */
function volumeTraces(charts: ChartData[] | null) {
  const traces = charts?.[0]?.traces ?? [];
  return {
    distance: traces.find((trace) => trace.axis !== "y2"),
    climb: traces.find((trace) => trace.axis === "y2"),
  };
}

/**
 * Banister form today, fitness minus fatigue — from the two series the Form
 * chart draws (fitness first, fatigue second; see plots/fitness_fatigue.py).
 */
function formValues(charts: ChartData[] | null) {
  const traces = charts?.[0]?.traces ?? [];
  const fitness = lastValue(traces[0]?.y);
  const fatigue = lastValue(traces[1]?.y);
  return {
    fitness,
    fatigue,
    form: fitness != null && fatigue != null ? fitness - fatigue : null,
  };
}

/**
 * The page's one hero (`tm-hero`, design/tagg/components/Hero.md): who, the
 * current week, and its three numbers — volume, climb and form, form being the
 * key figure in sun. The import action is secondary here; the page's primary
 * button stays in the Last Run card.
 */
function HomeHero({
  athlete,
  summary,
  volumeCharts,
  formCharts,
  busy,
  onImport,
  email,
  t,
}: {
  athlete: Athlete;
  summary: HomeSummary;
  volumeCharts: ChartData[] | null;
  formCharts: ChartData[] | null;
  busy: boolean;
  onImport: () => void;
  email: ReactNode;
  t: T;
}) {
  const monday = currentWeekStart();
  const sunday = new Date(monday);
  sunday.setDate(sunday.getDate() + 6);
  const day = (date: Date) => date.toLocaleDateString(undefined, { day: "numeric", month: "short" });

  const { distance, climb } = volumeTraces(volumeCharts);
  const km = currentWeekValue(distance);
  const gain = currentWeekValue(climb);
  const { form } = formValues(formCharts);

  const meta = [
    athlete.sync.last_synced_at
      ? `${t("home.import.last")} ${formatDate(athlete.sync.last_synced_at)}`
      : null,
    `${summary.profile.activity_count} ${t("home.profile.activities").toLowerCase()}`,
  ].filter(Boolean).join(" · ");

  const stats = [
    { label: t("home.hero.volume"), value: km != null ? formatNumber(km, 1) : "—", unit: t("common.km") },
    { label: t("home.hero.climb"), value: gain != null ? formatNumber(gain, 0) : "—", unit: t("common.metres") },
    { label: t("home.hero.form"), value: form != null ? formatSigned(form) : "—", key: true },
  ];

  return (
    <header className="tm-hero home-hero">
      {/* A plain <img>: the source is Strava's CDN, so there is nothing for the
          Next image loader to optimise, and it needs no remote-host allowlist. */}
      {athlete.profile_url && (
        <img className="tm-hero__avatar" src={athlete.profile_url} alt="" />
      )}
      <div className="tm-hero__body">
        <span className="tm-hero__kicker">
          {t("home.hero.week", { number: isoWeekNumber(monday), range: `${day(monday)} – ${day(sunday)}` })}
        </span>
        <h1 className="tm-hero__title">{athlete.display_name}</h1>
        <div className="hero-email">{email}</div>
        <span className="tm-hero__meta">{meta}</span>
      </div>
      <div className="tm-hero__stats">
        {stats.map((stat) => (
          <div className={`tm-hero__stat${stat.key ? " is-key" : ""}`} key={stat.label}>
            <span className="l">{stat.label}</span>
            <span className="v">
              {stat.value}
              {stat.unit && stat.value !== "—" && <small>{stat.unit}</small>}
            </span>
          </div>
        ))}
      </div>
      {/* Hidden, not disabled, for a coach: only the athlete can import. */}
      {!athlete.viewing_as && athlete.sync.status !== "running" && (
        <div className="home-hero__action">
          <button type="button" className="tm-btn tm-btn--secondary" onClick={onImport} disabled={busy}>
            <Icon name="refresh" /> {t("home.hero.import")}
          </button>
        </div>
      )}
    </header>
  );
}

/** A signed figure — form is read against zero, so "+4" and "−7", never "4". */
function formatSigned(value: number): string {
  const rounded = Math.round(value);
  return rounded > 0 ? `+${rounded}` : rounded < 0 ? `−${Math.abs(rounded)}` : "0";
}

// --- Recent progress -------------------------------------------------------

/** A recent-progress card's chart, or why there is none. */
function ChartBody({
  charts,
  hasData,
  empty,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  /** What to show when the panel came back with no chart. */
  empty?: ReactNode;
  t: T;
}) {
  if (!hasData) return <p className="muted">{t("home.last.empty")}</p>;
  if (charts === null) {
    return (
      <div className="pending">
        <span className="spinner" />
        <p className="muted">{t("common.loading")}</p>
      </div>
    );
  }
  if (charts.length === 0) return <>{empty ?? <p className="muted">{t("home.last.empty")}</p>}</>;
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

/** A card's heading row: the section title, and the trend badges beside it. */
function CardHeading({ children, badges }: { children: ReactNode; badges?: ReactNode }) {
  return (
    <div className="data-block__heading progress-card__heading">
      {children}
      {badges}
    </div>
  );
}

/** The last 20 weeks: distance and climb on one chart. Forest — what was done. */
function RecentHistoryCard({
  charts,
  hasData,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  t: T;
}) {
  const { distance } = volumeTraces(charts);
  const km = currentWeekValue(distance);
  const weekly = distance?.y.filter((v): v is number => v != null) ?? [];
  const average = weekly.length ? weekly.reduce((a, b) => a + b, 0) / weekly.length : null;

  return (
    <section className="card-block card-block--progress">
      <CardHeading>
        <SectionTitle icon="trending" kicker={t("home.kicker.weeks", { count: WEEKS_SHOWN })}>
          {t("home.recent.title")}
        </SectionTitle>
      </CardHeading>
      <p className="data-block__lede">{t("home.recent.subtitle")}</p>
      {hasData && charts && charts.length > 0 && (
        <div className="kpi-grid kpi-grid--headline">
          <Tile
            label={t("home.hero.this_week")}
            value={km != null ? formatNumber(km, 1) : "—"}
            unit={t("common.km")}
            tone="forest"
          />
          <Tile
            label={t("home.recent.weekly_average")}
            value={average != null ? formatNumber(average, 1) : "—"}
            unit={t("common.km")}
          />
        </div>
      )}
      <ChartBody charts={charts} hasData={hasData} t={t} />
    </section>
  );
}

/** The last 20 weeks of power-to-heart-rate. Terra — your own cost per beat. */
function RecentEfficiencyCard({
  charts,
  hasData,
  hasWeight,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  hasWeight: boolean;
  t: T;
}) {
  const latest = lastValue(charts?.[0]?.traces[0]?.y);
  return (
    <section className="card-block card-block--progress">
      <CardHeading
        badges={
          // Weekly points already — 4 and 12 of them are 4 and 12 weeks.
          <TrendBadgePair
            values={charts?.[0]?.traces[0]?.y}
            shortWindow={4}
            longWindow={12}
            shortThreshold={0.005}
            longThreshold={0.02}
            t={t}
          />
        }
      >
        <SectionTitle icon="zap" kicker={t("home.kicker.weeks", { count: WEEKS_SHOWN })} role="terra">
          {t("home.efficiency.title")}
        </SectionTitle>
      </CardHeading>
      <p className="data-block__lede">{t("home.efficiency.subtitle")}</p>

      {hasData && !hasWeight ? (
        // Power is stored per kilogram, so this chart is empty without a weight.
        // Said here rather than drawn blank — and the field to fix it is on the
        // Health card a few centimetres up.
        <Callout>{t("home.efficiency.needs_weight")}</Callout>
      ) : (
        <>
          {latest != null && (
            <div className="kpi-grid kpi-grid--headline">
              <Tile label={t("home.efficiency.latest")} value={formatNumber(latest, 2)} tone="terra" />
            </div>
          )}
          <ChartBody charts={charts} hasData={hasData} t={t} />
        </>
      )}
    </section>
  );
}

/**
 * The last 20 weeks of fitness and fatigue (Banister model). Sun — the signal of
 * the moment: form, fitness minus fatigue, is the headline; the two curves it is
 * read from stay in ink.
 */
function RecentFormCard({
  charts,
  hasData,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  t: T;
}) {
  const { fitness, fatigue, form } = formValues(charts);
  return (
    <section className="card-block card-block--progress">
      <CardHeading
        badges={
          // Fitness (the model's first, slow-moving trace) is the one that
          // answers "is training working?" — fatigue reacts to the last few
          // days and would make either badge flicker on the day's session
          // alone. Two windows because a short build and a long one answer
          // different questions: is this week's load landing, versus is the
          // block as a whole working. `fitness_fatigue_series` is one point
          // per *calendar day*, not per week (see training_load.py), so "4
          // weeks" and "12 weeks" here are 28 and 84 trailing daily points —
          // unlike the weekly-binned charts, where the window size and the
          // point count are the same number.
          <TrendBadgePair
            values={charts?.[0]?.traces[0]?.y}
            shortWindow={28}
            longWindow={84}
            shortThreshold={0.005}
            longThreshold={0.02}
            t={t}
          />
        }
      >
        <SectionTitle icon="flame" kicker={t("home.kicker.weeks", { count: WEEKS_SHOWN })} role="sun">
          {t("home.form.title")}
        </SectionTitle>
      </CardHeading>
      <p className="data-block__lede">{t("home.form.subtitle")}</p>
      {form != null && (
        <div className="kpi-grid kpi-grid--headline">
          <Tile label={t("home.hero.form")} value={formatSigned(form)} tone="sun" />
          <Tile label={t("home.form.fitness")} value={formatNumber(fitness ?? 0, 0)} />
          <Tile label={t("home.form.fatigue")} value={formatNumber(fatigue ?? 0, 0)} />
        </div>
      )}
      <ChartBody charts={charts} hasData={hasData} t={t} />
    </section>
  );
}

/** The feeling scale of plots/weekly_feel.py, by its score on the right axis. */
const FEELING_BY_SCORE = ["faible", "ok", "fort"] as const;

/**
 * The feeling of the last rated week, read off the Feel chart's feeling line —
 * the series on its right axis, one point per week, scored 1 (weak) to 3 (strong).
 */
function lastFeeling(chart: ChartData | undefined): "faible" | "ok" | "fort" | null {
  const score = lastValue(chart?.traces.find((trace) => trace.axis === "y2")?.y);
  return score != null ? FEELING_BY_SCORE[Math.round(score) - 1] ?? null : null;
}

/**
 * The last 20 weeks as the athlete rated them: RPE, feeling, fitness tag. Moss —
 * how it felt.
 *
 * No trend badge on this one. The three series answer *together* ("a hard week
 * that felt strong and moved fitness up") and a single arrow over any one of
 * them would say something the block is specifically there not to say. The empty
 * state is its own message rather than the shared "no activity" line: with
 * activities but no ratings, the chart is empty for a reason the athlete can fix
 * in two taps on the Training screen.
 */
function RecentFeelCard({
  charts,
  hasData,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  t: T;
}) {
  const feeling = lastFeeling(charts?.[0]);
  const rpe = lastValue(charts?.[0]?.traces[0]?.y);
  return (
    <section className="card-block card-block--progress">
      <CardHeading>
        <SectionTitle icon="heart" kicker={t("home.kicker.weeks", { count: WEEKS_SHOWN })} role="moss">
          {t("home.feel.title")}
        </SectionTitle>
      </CardHeading>
      <p className="data-block__lede">{t("home.feel.subtitle")}</p>
      {(feeling || rpe != null) && (
        <div className="kpi-grid kpi-grid--headline">
          <Tile
            label={t("home.feel.latest")}
            value={feeling ? t(`training.session.feeling_${feeling}`) : "—"}
            tone="moss"
          />
          <Tile label={t("home.feel.rpe")} value={rpe != null ? formatNumber(rpe, 1) : "—"} />
        </div>
      )}
      <ChartBody
        charts={charts}
        hasData={hasData}
        empty={<Callout>{t("home.feel.empty")}</Callout>}
        t={t}
      />
    </section>
  );
}

/**
 * A large "Recent · Increasing" style tag — sized and coloured to read at a
 * glance, without opening the chart itself. `label` names the window
 * ("Recent", "Sustained") without saying how many weeks it covers — that's an
 * implementation detail of the trend call, not something to expose in the UI.
 */
function TrendBadge({
  label,
  direction,
  t,
}: {
  label: string;
  direction: "increasing" | "stable" | "decreasing" | null;
  t: T;
}) {
  if (!direction) return null;
  return (
    <span className={chipClass(TREND_TONE[direction], "tm-chip--dot")}>
      {label} · {t(`home.trend.${direction}`)}
    </span>
  );
}

/**
 * A "Recent" and a "Sustained" {@link TrendBadge}, side by side.
 *
 * `shortWindow`/`longWindow` are counts of *trailing data points*, not weeks —
 * callers on a daily series (fitness/fatigue) and a weekly one (metric_trend)
 * need different numbers to mean the same span of real time. See
 * {@link trendDirection}.
 */
function TrendBadgePair({
  values,
  shortWindow,
  longWindow,
  shortThreshold,
  longThreshold,
  t,
}: {
  values: (number | null | undefined)[] | undefined;
  shortWindow: number;
  longWindow: number;
  shortThreshold: number;
  longThreshold: number;
  t: T;
}) {
  return (
    <div className="trend-badges">
      <TrendBadge
        label={t("home.trend.short_term")}
        direction={trendDirection(values, shortWindow, shortThreshold)}
        t={t}
      />
      <TrendBadge
        label={t("home.trend.long_term")}
        direction={trendDirection(values, longWindow, longThreshold)}
        t={t}
      />
    </div>
  );
}

/**
 * "Increasing" / "Stable" / "Decreasing" over the trailing `windowPoints`
 * points of a trend series — a least-squares slope over that window,
 * normalised by the window's own mean so the same `thresholdPct` applies
 * whether the series sits near 0 or near 1000.
 *
 * `windowPoints` counts *data points*, not weeks — the caller has to convert:
 * a weekly-binned series (metric_trend) has one point per week, but the
 * fitness/fatigue series is one point per calendar day (see
 * `fitness_fatigue_series` in training_load.py), so "4 weeks" there is 28
 * trailing points, not 4.
 *
 * `null` when there are fewer than four usable points: too little to call a
 * trend rather than noise, even for a short window.
 */
function trendDirection(
  values: (number | null | undefined)[] | undefined,
  windowPoints: number,
  thresholdPct: number,
): "increasing" | "stable" | "decreasing" | null {
  const points = (values ?? []).filter(
    (v): v is number => v != null && Number.isFinite(v),
  ).slice(-windowPoints);
  const n = points.length;
  if (n < 4) return null;

  const xMean = (n - 1) / 2;
  const yMean = points.reduce((sum, y) => sum + y, 0) / n;
  let num = 0;
  let den = 0;
  points.forEach((y, i) => {
    num += (i - xMean) * (y - yMean);
    den += (i - xMean) ** 2;
  });
  const slope = den === 0 ? 0 : num / den;
  const totalChange = slope * (n - 1);
  const scale = Math.abs(yMean) > 1e-9 ? Math.abs(yMean) : 1;
  const relativeChange = totalChange / scale;

  if (relativeChange > thresholdPct) return "increasing";
  if (relativeChange < -thresholdPct) return "decreasing";
  return "stable";
}

function SyncControls({
  athlete,
  busy,
  onImport,
  viewingAs,
  t,
}: {
  athlete: Athlete;
  busy: boolean;
  onImport: (force: boolean) => void;
  /** A coach browsing this athlete's account: only they can import their own
   *  Strava data, so the buttons that would trigger it are hidden, not disabled. */
  viewingAs: boolean;
  t: T;
}) {
  const syncing = athlete.sync.status === "running";

  return (
    <>
      {syncing ? (
        <div className="sync__progress">
          <ProgressBar
            value={athlete.sync.done}
            // 0 until the listing pass finishes, which the bar reads as "unknown"
            // and sweeps for — rather than sitting at 0% looking stuck.
            total={athlete.sync.total}
            label={t("home.import.running")}
            detail={athlete.sync.message || t("home.import.auto_help")}
          />
        </div>
      ) : viewingAs ? (
        <div className="sync__actions">
          <span className="muted">
            Only {athlete.display_name} can import their own Strava data.
          </span>
          {athlete.sync.last_synced_at && (
            <span className="muted sync__last">
              {t("home.import.last")} {formatDate(athlete.sync.last_synced_at)}
            </span>
          )}
        </div>
      ) : (
        <div className="sync__actions">
          <button
            type="button"
            className="tm-btn"
            onClick={() => onImport(false)}
            disabled={busy}
          >
            {athlete.activity_count
              ? t("home.import.more")
              : t("home.import.first")}
          </button>
          {athlete.activity_count > 0 && (
            <button
              type="button"
              className="tm-btn tm-btn--secondary"
              onClick={() => onImport(true)}
              disabled={busy}
              title={t("home.import.again_help")}
            >
              {t("home.import.again")}
            </button>
          )}
          {athlete.sync.status === "error" && (
            <Callout tone="terra">
              {t("home.import.failed")}: {athlete.sync.message}
            </Callout>
          )}
          {athlete.sync.last_synced_at && (
            <span className="muted sync__last">
              {t("home.import.last")} {formatDate(athlete.sync.last_synced_at)}
            </span>
          )}
        </div>
      )}

      {!syncing && !viewingAs && athlete.activity_count > 0 && (
        <p className="muted">{t("home.import.auto_help")}</p>
      )}

      {athlete.activity_count === 0 && !syncing && !viewingAs && (
        <Callout>{t("home.import.empty")}</Callout>
      )}
    </>
  );
}

// --- Small presentational pieces -------------------------------------------

/**
 * A card's headline tile colours its number in the card's role
 * (contrast.md § 3) — one per card, two at most; every other tile stays ink.
 */
type KpiTone = SectionRole;

function kpiClass(tone: KpiTone | undefined, extra = ""): string {
  return `tm-kpi tm-kpi--flat${tone ? ` tm-kpi--${tone}` : ""}${extra ? ` ${extra}` : ""}`;
}

function Tile({
  label,
  value,
  unit,
  footnote,
  tone,
  spark,
}: {
  label: string;
  value: string;
  unit?: string;
  footnote?: string | null;
  tone?: KpiTone;
  /** A sparkline under the number, in its colour — headline tiles only. */
  spark?: (number | null)[];
}) {
  return (
    <div className={kpiClass(tone)}>
      <span className="tm-kpi__label">{label}</span>
      <span className="tm-kpi__value">
        <span className="tm-kpi__num">{value}</span>
        {unit && <span className="tm-kpi__unit">{unit}</span>}
      </span>
      {spark && <Sparkline values={spark} />}
      {footnote && footnote !== "—" && <span className="kpi__note">{footnote}</span>}
    </div>
  );
}

/**
 * A tile that turns into an input when clicked.
 *
 * Reads as a value, edits in place. Committing on blur (not on every keystroke)
 * is what keeps this to one request per edit; an unchanged value sends nothing.
 */
function EditableTile({
  label,
  value,
  unit,
  help,
  input,
  onCommit,
  tone,
  t,
}: {
  label: string;
  value: string | null;
  unit?: string;
  help?: string;
  tone?: KpiTone;
  input: {
    type: "number" | "date" | "email" | "text";
    value: string;
    min?: number | string;
    max?: number | string;
    step?: number;
    placeholder?: string;
  };
  onCommit: (raw: string) => Promise<void>;
  t: T;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(input.value);
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);

  // Re-sync when a save elsewhere replaces the athlete.
  useEffect(() => setDraft(input.value), [input.value]);

  const commit = async () => {
    setEditing(false);
    if (draft === input.value) return;
    setSaving(true);
    setFailed(false);
    try {
      await onCommit(draft);
    } catch {
      // A rejected value (a malformed email) must not be left looking saved, and
      // must not lose what was typed.
      setFailed(true);
      setEditing(true);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={kpiClass(tone, "kpi--editable")}>
      <span className="tm-kpi__label">
        {label}
        {saving && <span> · {t("common.saving")}</span>}
      </span>

      {editing ? (
        <input
          className="tm-input"
          autoFocus
          type={input.type}
          value={draft}
          min={input.min}
          max={input.max}
          step={input.step}
          placeholder={input.placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setDraft(input.value);
              setEditing(false);
            }
          }}
        />
      ) : (
        <button
          type="button"
          className="tm-kpi__value kpi__button"
          onClick={() => setEditing(true)}
        >
          {value != null ? (
            <span className="tm-kpi__num">{value}</span>
          ) : (
            <span className="kpi__unset">{t("common.not_set")}</span>
          )}
          {unit && <span className="tm-kpi__unit">{unit}</span>}
        </button>
      )}

      {failed && <span className="kpi__note kpi__error">{t("common.not_saved")}</span>}
      {help && <span className="kpi__note">{help}</span>}
    </div>
  );
}
