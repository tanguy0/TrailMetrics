"use client";

/**
 * Home: who you are, what your history adds up to, and how your form is trending.
 *
 * Two summary cards side by side — Profile (what you have run) and Health (what
 * you have told us) — then the latest activity, the last 20 weeks of volume, and the
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

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { ChartView } from "@/components/ChartView";
import { EmailForm } from "@/components/EmailForm";
import { ProgressBar } from "@/components/ProgressBar";
import { RouteMap } from "@/components/RouteMap";
import {
  ApiError,
  getAthlete,
  getHomeSummary,
  getLastActivityRoute,
  getSyncStatus,
  renderPanel,
  startPrecompute,
  startSync,
  updateProfile,
} from "@/lib/api";
import { formatDate, formatHms, formatNumber, formatPace } from "@/lib/format";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type {
  ActivityCard,
  Athlete,
  ChartData,
  HomeRecord,
  HomeSummary,
  PanelSpec,
  RouteResult,
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
  const [formCharts, setFormCharts] = useState<ChartData[] | null>(null);
  const [route, setRoute] = useState<RouteResult | null>(null);
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
      setFormCharts(null);
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
    load(recentFormPanel(t), setFormCharts);
    return () => {
      live = false;
    };
    // Keyed on the activity count alone: `t` is rebuilt every render but its
    // strings never change, so including it would re-render the charts endlessly.
  }, [activityCount]);

  // The route is its own request: it can involve a call out to Strava, and the
  // rest of the screen should never wait on that.
  const lastActivityId = summary?.last_activity?.activity_id ?? null;
  useEffect(() => {
    if (lastActivityId == null) {
      setRoute(null);
      return;
    }
    let live = true;
    getLastActivityRoute()
      .then((result) => live && setRoute(result))
      .catch(() => live && setRoute(null));
    return () => {
      live = false;
    };
  }, [lastActivityId]);

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
      if (athlete.sync.status !== "running" && stale) {
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
        <p className="note note--error">{error}</p>
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
      <header className="hero">
        <div>
          <h1 className="hero__name">{athlete.display_name}</h1>
          <p className="hero__meta">
            {formatNumber(summary.profile.total_distance_m / 1000, 0)}{" "}
            {t("common.km")} · {summary.profile.activity_count}{" "}
            {t("home.profile.activities").toLowerCase()}
          </p>
        </div>
        {/* A plain <img>: the source is Strava's CDN, so there is nothing for the
            Next image loader to optimise, and it needs no remote-host allowlist. */}
        {athlete.profile_url && (
          <img className="hero__avatar" src={athlete.profile_url} alt="" />
        )}
      </header>

      {/* Someone who reached this screen without answering the email question —
          an account created before it existed, or a skipped `/welcome`. */}
      {athlete.needs_email && (
        <section className="card-block card-block--welcome">
          <h2 className="card-block__title">
            <span aria-hidden="true">✉️</span> {t("email.missing")}
          </h2>
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
        <ProfileCard summary={summary} t={t} />
        <HealthCard
          athlete={athlete}
          summary={summary}
          onSaved={setAthlete}
          t={t}
        />
      </div>

      <RecordsCard records={summary.records} t={t} />

      {/* Your data: the import controls, then what the data most recently says. */}
      <section className="card-block card-block--sync">
        <h2 className="card-block__title">
          <span aria-hidden="true">🔄</span> {t("home.import.title")}
        </h2>

        <SyncControls
          athlete={athlete}
          busy={busy}
          onImport={importActivities}
          t={t}
        />

        <div className="data-stack">
          <LastActivityBlock
            activity={summary.last_activity}
            route={route}
            t={t}
          />
          <RecentHistoryBlock
            charts={volumeCharts}
            hasData={activityCount > 0}
            t={t}
          />
          <RecentFormBlock
            charts={formCharts}
            hasData={activityCount > 0}
            hasWeight={athlete.weight_kg != null}
            t={t}
          />
        </div>
      </section>
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
 * The Recent Form panel: power-to-heart-rate per week.
 *
 * The one number on this screen that is about *fitness* rather than volume. Power is
 * modelled from speed and gradient, so the ratio says how much mechanical output the
 * athlete produced per heartbeat — it rises when the same effort buys more pace, and
 * it is insensitive to whether the week was hilly or flat, which raw pace is not.
 *
 * Aggregated as a mean over each week and smoothed over four of them. Week to week
 * the ratio is noisy for reasons that have nothing to do with form — a hot day, a
 * hard session, a watch that lost heart rate for ten minutes — so the unsmoothed
 * series invites reading a bad Tuesday as lost fitness. The trend is the signal.
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
          smooth_rolling: 4,
          show_totals: false,
        },
      },
    ],
  };
}

/** The last `WEEKS_SHOWN` weeks as a single named window. */
function recentWindow(name: string): PanelSpec["source"] {
  const end = new Date();
  const start = new Date(end);
  // Inclusive of the current week, so the window spans exactly WEEKS_SHOWN weeks.
  start.setDate(start.getDate() - (WEEKS_SHOWN * 7 - 1));
  return {
    mode: "window",
    activity_ids: [],
    selection_label: "",
    windows: [{ name, start: isoDate(start), end: isoDate(end) }],
    filters: { sport_types: [], min_distance_km: null, max_distance_km: null },
  };
}

/** Local calendar date as `YYYY-MM-DD` — `toISOString` would shift the day in UTC-. */
function isoDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// --- Profile ---------------------------------------------------------------

function ProfileCard({ summary, t }: { summary: HomeSummary; t: T }) {
  const { profile, records } = summary;
  return (
    <section className="card-block card-block--profile">
      <h2 className="card-block__title">
        <span aria-hidden="true">🏃</span> {t("home.profile.title")}
      </h2>

      <div className="tile-grid">
        <Tile
          tone="forest"
          label={t("home.profile.activities")}
          value={String(profile.activity_count)}
        />
        <Tile
          tone="terracotta"
          label={t("home.profile.total_distance")}
          value={formatNumber(profile.total_distance_m / 1000, 0)}
          unit={t("common.km")}
        />
        <Tile
          tone="sunrise"
          label={t("home.profile.total_elevation")}
          value={formatNumber(profile.total_elevation_gain_m, 0)}
          unit={t("common.metres")}
        />
        <Tile
          tone="moss"
          label={t("home.profile.total_time")}
          value={formatHms(profile.total_moving_s)}
        />
        <Tile
          tone="slate"
          label={t("home.profile.oldest")}
          value={formatDate(profile.oldest_activity)}
        />
        <Tile
          tone="slate"
          label={t("home.profile.newest")}
          value={formatDate(profile.newest_activity)}
        />
        <Tile
          tone="plum"
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
          tone="teal"
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
  return (
    <section className="card-block card-block--records">
      <h2 className="card-block__title">
        <span aria-hidden="true">🏅</span> {t("home.profile.records")}
      </h2>
      {records.length ? (
        <div className="record-grid">
          {records.map((record) => (
            <div className="record" key={record.label}>
              <span className="record__distance">{record.label}</span>
              <span className="record__time">{formatHms(record.seconds)}</span>
              <span className="record__date">{formatDate(record.set_on)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">{t("home.profile.records_empty")}</p>
      )}
    </section>
  );
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
      <h2 className="card-block__title">
        <span aria-hidden="true">❤️</span> {t("home.health.title")}
      </h2>

      <div className="tile-grid tile-grid--two">
        <EditableTile
          tone="rose"
          label={t("home.health.age")}
          value={athlete.age != null ? String(athlete.age) : null}
          unit={athlete.age != null ? t("common.years") : undefined}
          help={t("home.health.age_help")}
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
          tone="amber"
          label={t("home.health.experience")}
          value={experience != null ? formatNumber(experience, 1) : "—"}
          unit={experience != null ? t("common.years") : undefined}
          footnote={t("home.health.experience_help")}
        />

        <EditableTile
          tone="forest"
          label={t("home.health.weight")}
          value={athlete.weight_kg != null ? formatNumber(athlete.weight_kg, 1) : null}
          unit={athlete.weight_kg != null ? t("common.kg") : undefined}
          help={t("home.health.weight_help")}
          input={{ type: "number", value: athlete.weight_kg?.toString() ?? "", min: 25, max: 250, step: 0.5 }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ weight_kg: raw === "" ? null : Number(raw) }))
          }
          t={t}
        />

        <EditableTile
          tone="teal"
          label={t("home.health.height")}
          value={athlete.height_cm != null ? formatNumber(athlete.height_cm, 0) : null}
          unit={athlete.height_cm != null ? t("common.cm") : undefined}
          input={{ type: "number", value: athlete.height_cm?.toString() ?? "", min: 100, max: 250, step: 1 }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ height_cm: raw === "" ? null : Number(raw) }))
          }
          t={t}
        />

        {/* Editable like the rest: the athlete gave it, so they can correct it. */}
        <EditableTile
          tone="slate"
          label={t("home.health.email")}
          value={athlete.email}
          input={{ type: "email", value: athlete.email ?? "" }}
          onCommit={async (raw) =>
            onSaved(await updateProfile({ email: raw.trim() === "" ? null : raw.trim() }))
          }
          t={t}
        />
      </div>
    </section>
  );
}

// --- Latest activity and weekly volume -------------------------------------

/** The latest activity: its numbers, and its route on a map when there is one. */
function LastActivityBlock({
  activity,
  route,
  t,
}: {
  activity: ActivityCard | null;
  route: RouteResult | null;
  t: T;
}) {
  if (!activity) {
    return (
      <div className="data-block">
        <h3 className="data-block__title">
          <span aria-hidden="true">📍</span> {t("home.last.title")}
        </h3>
        <p className="muted">{t("home.last.empty")}</p>
      </div>
    );
  }

  const km = activity.distance_m != null ? activity.distance_m / 1000 : null;
  const pace =
    km && km > 0 && activity.moving_s != null ? activity.moving_s / km : null;

  return (
    <div className="data-block">
      <h3 className="data-block__title">
        <span aria-hidden="true">📍</span> {t("home.last.title")}
      </h3>

      <p className="last-activity__head">
        <span className="last-activity__sport">{activity.sport_type}</span>
        <span className="last-activity__date">{formatDate(activity.date)}</span>
      </p>

      <dl className="metric-row">
        <Metric
          label={t("home.last.distance")}
          value={km != null ? `${formatNumber(km, 2)} ${t("common.km")}` : "—"}
        />
        <Metric label={t("home.last.time")} value={formatHms(activity.moving_s)} />
        <Metric label={t("home.last.pace")} value={formatPace(pace)} />
        <Metric
          label={t("home.last.climb")}
          value={
            activity.elevation_gain_m != null
              ? `${formatNumber(activity.elevation_gain_m, 0)} ${t("common.metres")}`
              : "—"
          }
        />
        {activity.avg_hr != null && (
          <Metric
            label={t("home.last.heart_rate")}
            value={`${formatNumber(activity.avg_hr, 0)} bpm`}
          />
        )}
      </dl>

      {route === null ? (
        <div className="pending">
          <span className="spinner" />
          <p className="muted">{t("home.last.map_loading")}</p>
        </div>
      ) : route.points.length ? (
        <RouteMap points={route.points} />
      ) : (
        // Says which kind of nothing this is: a treadmill run has no route to draw,
        // an unreachable Strava is a different problem with the same blank space.
        <p className="muted">
          {route.source === "unavailable"
            ? t("home.last.map_unavailable")
            : t("home.last.map_none")}
        </p>
      )}
    </div>
  );
}

/** The last 20 weeks: distance and climb on one chart, across the full width. */
function RecentHistoryBlock({
  charts,
  hasData,
  t,
}: {
  charts: ChartData[] | null;
  hasData: boolean;
  t: T;
}) {
  return (
    <div className="data-block">
      <h3 className="data-block__title">
        <span aria-hidden="true">📈</span> {t("home.recent.title")}
      </h3>
      <p className="data-block__lede">{t("home.recent.subtitle")}</p>

      {!hasData ? (
        <p className="muted">{t("home.last.empty")}</p>
      ) : charts === null ? (
        <div className="pending">
          <span className="spinner" />
          <p className="muted">{t("common.loading")}</p>
        </div>
      ) : charts.length === 0 ? (
        <p className="muted">{t("home.last.empty")}</p>
      ) : (
        charts.map((chart, index) => (
          <div className="chart-frame" key={index}>
            <ChartView chart={chart} />
          </div>
        ))
      )}
    </div>
  );
}

/** The last 20 weeks of power-to-heart-rate: recent form at a glance. */
function RecentFormBlock({
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
  return (
    <div className="data-block">
      <h3 className="data-block__title">
        <span aria-hidden="true">⚡</span> {t("home.form.title")}
      </h3>
      <p className="data-block__lede">{t("home.form.subtitle")}</p>

      {!hasData ? (
        <p className="muted">{t("home.last.empty")}</p>
      ) : !hasWeight ? (
        // Power is stored per kilogram, so this chart is empty without a weight.
        // Said here rather than drawn blank — and the field to fix it is on the
        // Health card a few centimetres up.
        <p className="note">{t("home.form.needs_weight")}</p>
      ) : charts === null ? (
        <div className="pending">
          <span className="spinner" />
          <p className="muted">{t("common.loading")}</p>
        </div>
      ) : charts.length === 0 ? (
        <p className="muted">{t("home.last.empty")}</p>
      ) : (
        charts.map((chart, index) => (
          <div className="chart-frame" key={index}>
            <ChartView chart={chart} />
          </div>
        ))
      )}
    </div>
  );
}

function SyncControls({
  athlete,
  busy,
  onImport,
  t,
}: {
  athlete: Athlete;
  busy: boolean;
  onImport: (force: boolean) => void;
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
      ) : (
        <div className="sync__actions">
          <button
            type="button"
            className="button"
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
              className="button button--ghost"
              onClick={() => onImport(true)}
              disabled={busy}
              title={t("home.import.again_help")}
            >
              {t("home.import.again")}
            </button>
          )}
          {athlete.sync.status === "error" && (
            <span className="note note--error">
              {t("home.import.failed")}: {athlete.sync.message}
            </span>
          )}
          {athlete.sync.last_synced_at && (
            <span className="muted">
              {t("home.import.last")} {formatDate(athlete.sync.last_synced_at)}
            </span>
          )}
        </div>
      )}

      {!syncing && athlete.activity_count > 0 && (
        <p className="muted">{t("home.import.auto_help")}</p>
      )}

      {athlete.activity_count === 0 && !syncing && (
        <p className="note">{t("home.import.empty")}</p>
      )}
    </>
  );
}

// --- Small presentational pieces -------------------------------------------

type Tone =
  | "forest" | "terracotta" | "sunrise" | "moss" | "slate"
  | "plum" | "teal" | "rose" | "amber";

function Tile({
  tone,
  label,
  value,
  unit,
  footnote,
}: {
  tone: Tone;
  label: string;
  value: string;
  unit?: string;
  footnote?: string | null;
}) {
  return (
    <div className={`tile tile--${tone}`}>
      <span className="tile__label">{label}</span>
      <span className="tile__value">
        {value}
        {unit && <span className="tile__unit">{unit}</span>}
      </span>
      {footnote && footnote !== "—" && (
        <span className="tile__footnote">{footnote}</span>
      )}
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
  tone,
  label,
  value,
  unit,
  help,
  input,
  onCommit,
  t,
}: {
  tone: Tone;
  label: string;
  value: string | null;
  unit?: string;
  help?: string;
  input: {
    type: "number" | "date" | "email";
    value: string;
    min?: number | string;
    max?: number | string;
    step?: number;
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
    <div className={`tile tile--${tone} tile--editable`}>
      <span className="tile__label">
        {label}
        {saving && <span className="tile__saving"> · {t("common.saving")}</span>}
      </span>

      {editing ? (
        <input
          className="tile__input"
          autoFocus
          type={input.type}
          value={draft}
          min={input.min}
          max={input.max}
          step={input.step}
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
          className="tile__value tile__value--button"
          onClick={() => setEditing(true)}
        >
          {value ?? <span className="tile__unset">{t("common.not_set")}</span>}
          {unit && <span className="tile__unit">{unit}</span>}
        </button>
      )}

      {failed && <span className="tile__footnote tile__footnote--error">
        {t("common.not_saved")}
      </span>}
      {help && <span className="tile__footnote">{help}</span>}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <dt className="metric__label">{label}</dt>
      <dd className="metric__value">{value}</dd>
    </div>
  );
}
