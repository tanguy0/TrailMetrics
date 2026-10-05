"use client";

/**
 * Race plan ("Plan de course"): one plan's inputs on top, its pace profile below.
 *
 * The same screen serves three cases:
 *
 *  - a visitor (`signedIn` false): computes on the reference curves only, cannot
 *    save, and is told — once, with the result — that an account keeps it;
 *  - a new plan (`planId` null): computes from the chosen file, and the first save
 *    creates it and moves the URL to `/tools/race-planning/{id}` without a reload;
 *  - a saved plan: loads its inputs and computes straight away from the stored
 *    GPX, so a plan opens already drawn. Choosing a new file replaces the stored
 *    one on the next save.
 *
 * All the numbers — pacing, sections, legs — are computed server-side and arrive
 * as chart IR, drawn by the same `ChartView`/`TableView` as every analysis panel.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Callout } from "@/components/Callout";
import { ChartView } from "@/components/ChartView";
import { PageHeader } from "@/components/PageHeader";
import { TableView } from "@/components/TableView";
import {
  deleteRacePlan,
  getRacePlan,
  getRacePlanOptions,
  planRace,
  saveRacePlan,
} from "@/lib/api";
import { registerHref } from "@/lib/auth";
import { formatHms, formatNumber, formatPaceInput, kpiNumClass } from "@/lib/format";
import { plural, translator, type Strings, type Translate } from "@/lib/strings";
import type {
  PlotOutput,
  RacePlanCurveOption,
  RacePlanImportance,
  RacePlanParams,
  RacePlanResult,
} from "@/lib/types";

interface AidRow {
  km: string;
  name: string;
}

/** `4:30`, `4:30:15`, `4h30` or `4h30m15` → seconds; `null` when unreadable. */
export function parseDuration(text: string): number | null {
  const match = text.trim().match(/^(\d+)\s*[:h]\s*(\d{1,2})(?:\s*[:m]\s*(\d{1,2}))?\s*s?$/i);
  if (!match) return null;
  const [hours, minutes, seconds] = [match[1], match[2], match[3] ?? "0"].map(Number);
  if (minutes > 59 || seconds > 59) return null;
  const total = hours * 3600 + minutes * 60 + seconds;
  return total > 0 ? total : null;
}

/** `07:30` → seconds after midnight. */
function parseClock(text: string): number | null {
  const match = text.trim().match(/^(\d{1,2})[:h](\d{2})$/);
  if (!match) return null;
  const [hours, minutes] = [Number(match[1]), Number(match[2])];
  return hours < 24 && minutes < 60 ? hours * 3600 + minutes * 60 : null;
}

function formatClock(seconds: number): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.floor(seconds / 3600))}:${pad(Math.floor((seconds % 3600) / 60))}`;
}

/** French users type `12,5`; both mean twelve and a half kilometres. */
function parseKm(text: string): number | null {
  const value = Number(text.trim().replace(",", "."));
  return text.trim() && Number.isFinite(value) ? value : null;
}

/** An optional bounded number: `null` when empty, `undefined` when invalid. */
function parseOptional(text: string, min: number, max: number): number | null | undefined {
  if (!text.trim()) return null;
  const value = parseKm(text);
  return value != null && value >= min && value <= max ? value : undefined;
}

const numberText = (value: number | null | undefined) => (value == null ? "" : String(value));

export function RacePlanScreen({
  strings,
  signedIn,
  planId: initialPlanId,
}: {
  strings: Strings;
  signedIn: boolean;
  planId: string | null;
}) {
  const t = translator(strings);

  const [planId, setPlanId] = useState<string | null>(initialPlanId);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [storedGpxName, setStoredGpxName] = useState<string | null>(null);
  const [targetTime, setTargetTime] = useState("");
  const [startTime, setStartTime] = useState("");
  const [aidRows, setAidRows] = useState<AidRow[]>([]);
  const [curve, setCurve] = useState<string | null>(null);
  const [curves, setCurves] = useState<RacePlanCurveOption[]>([]);
  const [durability, setDurability] = useState(true);
  const [temperatureStart, setTemperatureStart] = useState("");
  const [temperatureEnd, setTemperatureEnd] = useState("");
  const [humidity, setHumidity] = useState("");
  // The race itself, saved with the plan (not a planning input).
  const [eventDate, setEventDate] = useState("");
  const [importance, setImportance] = useState<RacePlanImportance | "">("");

  const [result, setResult] = useState<RacePlanResult | null>(null);
  const [computing, setComputing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(Boolean(initialPlanId));

  // Validated form → request params, or a translated message saying what's wrong.
  const buildParams = useCallback((): RacePlanParams | string => {
    const target = parseDuration(targetTime);
    if (target == null) return t("race_plan.error.target_time");
    let start: number | null = null;
    if (startTime.trim()) {
      start = parseClock(startTime);
      if (start == null) return t("race_plan.error.start_time");
    }
    const aid_stations = aidRows
      .map((row) => ({ km: parseKm(row.km), name: row.name.trim() }))
      .filter((row): row is { km: number; name: string } => row.km != null);
    const temperature_start_c = parseOptional(temperatureStart, -40, 55);
    const temperature_end_c = parseOptional(temperatureEnd, -40, 55);
    const relative_humidity_pct = parseOptional(humidity, 0, 100);
    if (
      temperature_start_c === undefined ||
      temperature_end_c === undefined ||
      relative_humidity_pct === undefined
    ) {
      return t("race_plan.error.weather");
    }
    return {
      target_time_s: target,
      aid_stations,
      start_time_s: start,
      curve,
      durability,
      temperature_start_c,
      temperature_end_c,
      relative_humidity_pct,
    };
  }, [
    targetTime,
    startTime,
    aidRows,
    curve,
    durability,
    temperatureStart,
    temperatureEnd,
    humidity,
    t,
  ]);

  const compute = useCallback(
    async (params: RacePlanParams, source: { gpx: File } | { planId: string }) => {
      setComputing(true);
      setError(null);
      try {
        const planned = await planRace(source, params);
        setResult(planned);
        setCurves(planned.curves);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setComputing(false);
      }
    },
    [],
  );

  useEffect(() => {
    getRacePlanOptions()
      .then((options) => setCurves(options.curves))
      .catch(() => {});
  }, []);

  // A saved plan opens with its inputs filled in and its result already computing.
  useEffect(() => {
    if (!initialPlanId) return;
    getRacePlan(initialPlanId)
      .then((saved) => {
        const p = saved.params;
        setTitle(saved.title);
        setStoredGpxName(saved.gpx_name || "course.gpx");
        setTargetTime(formatHms(p.target_time_s, { exact: true }));
        setStartTime(p.start_time_s != null ? formatClock(p.start_time_s) : "");
        setAidRows(p.aid_stations.map((s) => ({ km: String(s.km), name: s.name })));
        setCurve(p.curve);
        setDurability(p.durability ?? true);
        setTemperatureStart(numberText(p.temperature_start_c));
        setTemperatureEnd(numberText(p.temperature_end_c));
        setHumidity(numberText(p.relative_humidity_pct));
        setEventDate(saved.event_date ?? "");
        setImportance(saved.importance ?? "");
        setLoading(false);
        return compute(p, { planId: initialPlanId });
      })
      .catch((e: Error) => {
        setError(e.message);
        setLoading(false);
      });
  }, [initialPlanId, compute]);

  const source = (): { gpx: File } | { planId: string } | null =>
    file ? { gpx: file } : planId ? { planId } : null;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const params = buildParams();
    if (typeof params === "string") return setError(params);
    const from = source();
    if (!from) return setError(t("race_plan.error.no_gpx"));
    void compute(params, from);
  };

  const save = async () => {
    const params = buildParams();
    if (typeof params === "string") return setError(params);
    if (!file && !planId) return setError(t("race_plan.error.no_gpx"));
    setSaving(true);
    setError(null);
    try {
      const saved = await saveRacePlan(planId, title.trim(), params, file, {
        event_date: eventDate || null,
        importance: importance || null,
      });
      if (!planId) {
        // Now a saved plan: give it its own URL without remounting the screen.
        window.history.replaceState(null, "", `/tools/race-planning/${saved.id}`);
      }
      setPlanId(saved.id);
      setStoredGpxName(saved.gpx_name || storedGpxName);
      setFile(null);
      setSavedAt(Date.now());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!planId) return;
    const name = title.trim() || t("race_plan.untitled");
    if (!window.confirm(t("race_plan.delete_confirm", { title: name }))) return;
    try {
      await deleteRacePlan(planId);
      window.location.href = "/tools/race-planning";
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const updateRow = (index: number, patch: Partial<AidRow>) =>
    setAidRows((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  // The saved hint fades out after a few seconds rather than lingering as a
  // claim about a form that may have changed since.
  useEffect(() => {
    if (savedAt == null) return;
    const timer = window.setTimeout(() => setSavedAt(null), 2500);
    return () => window.clearTimeout(timer);
  }, [savedAt]);

  if (loading) {
    return (
      <main className="container">
        <p className="muted">{t("common.loading")}</p>
      </main>
    );
  }

  const selectedCurve = curve ?? curves.find((c) => c.available)?.key ?? "";
  const personalSelected = curves.find((c) => c.key === selectedCurve)?.personal ?? false;

  return (
    <main className="container race-plan">
      {signedIn && (
        <Link className="race-plan__back" href="/tools/race-planning">
          {t("race_plan.back")}
        </Link>
      )}
      <PageHeader
        kicker={t("tools.race_planning")}
        title={
          <input
            className="page-title-input"
            aria-label={t("race_plan.plan_title")}
            placeholder={t("race_plan.title_placeholder")}
            value={title}
            maxLength={200}
            onChange={(e) => setTitle(e.target.value)}
          />
        }
        sub={t("race_plan.intro")}
        actions={
          signedIn && (
            <>
              {savedAt != null && <span className="muted">{t("race_plan.saved")}</span>}
              <button
                type="button"
                className="tm-btn tm-btn--secondary tm-btn--sm"
                onClick={save}
                disabled={saving}
              >
                {saving ? t("race_plan.saving") : t("race_plan.save")}
              </button>
              {planId && (
                <button type="button" className="tm-btn tm-btn--danger tm-btn--sm" onClick={remove}>
                  {t("race_plan.delete")}
                </button>
              )}
            </>
          )
        }
      />

      {result && (
        <RacePlanHero result={result} name={title.trim() || t("race_plan.untitled")} t={t} />
      )}

      <form className="tm-panel panel race-plan__form" onSubmit={submit}>
        <div className="race-plan__fields">
          <label className="race-plan__field">
            <span>{t("race_plan.gpx")}</span>
            <input
              type="file"
              accept=".gpx,application/gpx+xml,application/xml,text/xml"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {storedGpxName && !file && (
              <span className="muted race-plan__help">
                {t("race_plan.gpx_current", { name: storedGpxName })} ·{" "}
                {t("race_plan.gpx_replace")}
              </span>
            )}
          </label>

          <label className="race-plan__field">
            <span>{t("race_plan.target_time")}</span>
            <input
              className="tm-input"
              type="text"
              inputMode="numeric"
              placeholder="4:30:00"
              value={targetTime}
              onChange={(e) => setTargetTime(e.target.value)}
            />
            <span className="muted race-plan__help">{t("race_plan.target_time_help")}</span>
          </label>

          <label className="race-plan__field">
            <span>{t("race_plan.start_time")}</span>
            <input
              className="tm-input"
              type="text"
              inputMode="numeric"
              placeholder="07:00"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
            />
            <span className="muted race-plan__help">{t("race_plan.start_time_help")}</span>
          </label>

          {signedIn && (
            <label className="race-plan__field">
              <span>{t("race_plan.event_date")}</span>
              <input
                className="tm-input"
                type="date"
                value={eventDate}
                onChange={(e) => setEventDate(e.target.value)}
              />
            </label>
          )}

          {signedIn && (
            <label className="race-plan__field">
              <span>{t("race_plan.importance")}</span>
              <select
                className="tm-select"
                value={importance}
                onChange={(e) => setImportance(e.target.value as RacePlanImportance | "")}
              >
                <option value="">{t("race_plan.importance.none")}</option>
                <option value="primary">{t("race_plan.importance.primary")}</option>
                <option value="secondary">{t("race_plan.importance.secondary")}</option>
              </select>
              <span className="muted race-plan__help">{t("race_plan.importance_help")}</span>
            </label>
          )}

          <label className="race-plan__field">
            <span>{t("race_plan.curve")}</span>
            <select className="tm-select" value={selectedCurve} onChange={(e) => setCurve(e.target.value)}>
              {curves.map((option) => (
                <option key={option.key} value={option.key} disabled={!option.available}>
                  {option.label}
                  {!option.available ? ` (${t("race_plan.curve_sign_in")})` : ""}
                </option>
              ))}
            </select>
          </label>
        </div>

        <fieldset className="race-plan__aid">
          <legend>{t("race_plan.aid_stations")}</legend>
          {aidRows.map((row, index) => (
            <div className="race-plan__aid-row" key={index}>
              <input
                type="text"
                inputMode="decimal"
                className="tm-input race-plan__aid-km"
                aria-label={t("race_plan.aid_station_km")}
                placeholder={t("race_plan.aid_station_km")}
                value={row.km}
                onChange={(e) => updateRow(index, { km: e.target.value })}
              />
              <input
                type="text"
                className="tm-input race-plan__aid-name"
                aria-label={t("race_plan.aid_station_name")}
                placeholder={t("race_plan.aid_station_name")}
                value={row.name}
                maxLength={80}
                onChange={(e) => updateRow(index, { name: e.target.value })}
              />
              <button
                type="button"
                className="tm-btn tm-btn--secondary tm-btn--sm"
                onClick={() => setAidRows((rows) => rows.filter((_, i) => i !== index))}
              >
                {t("race_plan.remove")}
              </button>
            </div>
          ))}
          <button
            type="button"
            className="tm-btn tm-btn--secondary tm-btn--sm"
            onClick={() => setAidRows((rows) => [...rows, { km: "", name: "" }])}
          >
            {t("race_plan.add_aid_station")}
          </button>
        </fieldset>

        <fieldset className="race-plan__aid">
          <legend>{t("race_plan.conditions")}</legend>
          <label className="tm-toggle">
            <input
              type="checkbox"
              checked={durability}
              onChange={(e) => setDurability(e.target.checked)}
            />
            <span className="tm-toggle__track" aria-hidden="true" />
            <span>{t("race_plan.durability")}</span>
          </label>
          <div className="race-plan__fields">
            <label className="race-plan__field">
              <span>{t("race_plan.temperature_start")}</span>
              <input
                className="tm-input"
                type="text"
                inputMode="decimal"
                placeholder="12"
                value={temperatureStart}
                disabled={!durability}
                onChange={(e) => setTemperatureStart(e.target.value)}
              />
            </label>
            <label className="race-plan__field">
              <span>{t("race_plan.temperature_end")}</span>
              <input
                className="tm-input"
                type="text"
                inputMode="decimal"
                placeholder="22"
                value={temperatureEnd}
                disabled={!durability}
                onChange={(e) => setTemperatureEnd(e.target.value)}
              />
            </label>
            <label className="race-plan__field">
              <span>{t("race_plan.humidity")}</span>
              <input
                className="tm-input"
                type="text"
                inputMode="decimal"
                placeholder="60"
                value={humidity}
                disabled={!durability}
                onChange={(e) => setHumidity(e.target.value)}
              />
            </label>
          </div>
          <span className="muted race-plan__help">{t("race_plan.weather_help")}</span>
        </fieldset>

        <div className="race-plan__actions">
          <button type="submit" className="tm-btn" disabled={computing}>
            {computing ? t("race_plan.computing") : t("race_plan.submit")}
          </button>
          {computing && (
            <span className="pending">
              <span className="spinner" aria-hidden="true" />
              {personalSelected && (
                <span className="muted">{t("race_plan.computing_personal")}</span>
              )}
            </span>
          )}
        </div>
        {error && <Callout tone="terra">{error}</Callout>}
      </form>

      {/* A visitor gets the result, and one line on what an account keeps
          (access.md § Visiteur) — under the result, once there is one to keep. */}
      {result && !signedIn && (
        <Callout tone="forest">
          {t("tools.keep_plans")}{" "}
          <a href={registerHref("/tools/race-planning/new")}>{t("visitor.register")}</a>
        </Callout>
      )}
      {result && <RacePlanResultView result={result} t={t} />}
    </main>
  );
}

/**
 * The plan's one hero (`tm-hero` compact, design/tagg/components/Hero.md § Plan
 * de course): which curve paced it, the race's name, the strategy in one line,
 * and four numbers — the target time in sun first. "Target time" appears once,
 * in its stat; the title is the race.
 */
function RacePlanHero({ result, name, t }: { result: RacePlanResult; name: string; t: Translate }) {
  const s = result.summary;
  const kicker = [
    t("race_plan.hero.kicker", { curve: result.curve_label }),
    result.personalized ? t("race_plan.hero.personalized") : null,
  ].filter(Boolean).join(" · ");
  // Each fragment is left out when the summary does not carry it.
  const meta = [
    Number.isFinite(s.gap_pace_s_per_km) ? t("race_plan.hero.gap", { pace: formatPaceInput(s.gap_pace_s_per_km) }) : null,
    Number.isFinite(s.average_pace_s_per_km) ? t("race_plan.hero.real", { pace: formatPaceInput(s.average_pace_s_per_km) }) : null,
    s.section_count ? plural(t, "race_plan.hero.sections", s.section_count) : null,
    s.aid_station_count ? plural(t, "race_plan.hero.aid_stations", s.aid_station_count) : null,
    s.durability_enabled && s.durability_multiplier_finish != null
      ? t("race_plan.hero.drift", {
          factor: formatNumber(s.durability_multiplier_finish, 2),
          confidence: t(`race_plan.confidence.${s.durability_confidence ?? "population_only"}`),
        })
      : null,
  ].filter(Boolean).join(" · ");
  const stats = [
    { label: t("race_plan.summary.target"), value: formatHms(s.target_time_s, { exact: true }), key: true },
    { label: t("race_plan.summary.distance"), value: formatNumber(s.distance_m / 1000, 1), unit: "km" },
    {
      label: t("race_plan.hero.gain_loss"),
      value: `+${formatNumber(s.elevation_gain_m, 0)} / −${formatNumber(s.elevation_loss_m, 0)}`,
      unit: "m",
    },
    { label: t("race_plan.hero.gap_pace"), value: formatPaceInput(s.gap_pace_s_per_km), unit: "/km" },
  ];
  return (
    <header className="tm-hero tm-hero--compact race-plan__hero">
      <div className="tm-hero__body">
        <span className="tm-hero__kicker">{kicker}</span>
        <h2 className="tm-hero__title">{name}</h2>
        {meta && <span className="tm-hero__meta">{meta}</span>}
      </div>
      <div className="tm-hero__sep" role="separator" />
      <div className="tm-hero__stats">
        {stats.map((stat) => (
          <div className={`tm-hero__stat${stat.key ? " is-key" : ""}`} key={stat.label}>
            <span className="l">{stat.label}</span>
            <span className="v">
              {stat.value}
              {stat.unit && <small>{stat.unit}</small>}
            </span>
          </div>
        ))}
      </div>
    </header>
  );
}

function RacePlanResultView({ result, t }: { result: RacePlanResult; t: Translate }) {
  const s = result.summary;
  // [label, value, unit] — the unit apart, so the value's length alone sizes it.
  const perKm = t("common.per_km");
  const tiles: [string, string, string?][] = [
    [t("race_plan.summary.distance"), formatNumber(s.distance_m / 1000, 1), "km"],
    [
      t("race_plan.summary.elevation"),
      `+${formatNumber(s.elevation_gain_m, 0)} / −${formatNumber(s.elevation_loss_m, 0)}`,
      "m",
    ],
    [t("race_plan.summary.gap_pace"), formatPaceInput(s.gap_pace_s_per_km), perKm],
    [t("race_plan.summary.avg_pace"), formatPaceInput(s.average_pace_s_per_km), perKm],
    [t("race_plan.summary.curve"), result.curve_label],
  ];
  if (s.durability_multiplier_finish != null && s.durability_enabled) {
    tiles.push(
      [
        t("race_plan.summary.durability_finish"),
        `+${formatNumber((s.durability_multiplier_finish - 1) * 100, 1)}`,
        "%",
      ],
      [t("race_plan.summary.gap_finish"), formatPaceInput(s.gap_pace_finish_s_per_km ?? NaN), perKm],
      [
        t("race_plan.summary.durability_model"),
        t(`race_plan.confidence.${s.durability_confidence ?? "population_only"}`),
      ],
    );
  }

  return (
    <div className="race-plan__result">
      <div className="kpi-grid race-plan__summary">
        {tiles.map(([label, value, unit]) => (
          <div className="tm-kpi" key={label}>
            <span className="tm-kpi__label">{label}</span>
            <span className="tm-kpi__value">
              <span className={kpiNumClass(value)}>{value}</span>
              {unit && <span className="tm-kpi__unit">{unit}</span>}
            </span>
          </div>
        ))}
      </div>
      {/* One callout per card at most: the notes go in it together. */}
      {result.notes.length > 0 && (
        <Callout>
          {result.notes.map((note, index) => (
            <span className="callout__line" key={index}>{note}</span>
          ))}
        </Callout>
      )}

      <OutputSection title={t("race_plan.section.profile")} output={result.outputs.profile} />
      <OutputSection title={t("race_plan.section.sections")} output={result.outputs.sections} />
      <OutputSection
        title={t("race_plan.section.aid_stations")}
        output={result.outputs.aid_stations}
      />
      {result.outputs.durability && (
        <OutputSection
          title={t("race_plan.section.durability")}
          output={result.outputs.durability}
        />
      )}
    </div>
  );
}

function OutputSection({ title, output }: { title: string; output: PlotOutput }) {
  return (
    <section className="tm-panel panel">
      <div className="panel__header">
        <h2 className="panel__title">{title}</h2>
      </div>
      {output.charts.map((chart, index) => (
        <ChartView key={index} chart={chart} />
      ))}
      {output.tables.map((table, index) => (
        <TableView key={index} table={table} />
      ))}
      {/* One callout per card at most: the notes go in it together. */}
      {output.notes.length > 0 && (
        <Callout>
          {output.notes.map((note, index) => (
            <span className="callout__line" key={index}>{note}</span>
          ))}
        </Callout>
      )}
    </section>
  );
}
