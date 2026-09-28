"use client";

/**
 * Race plan ("Plan de course"): one plan's inputs on top, its pace profile below.
 *
 * The same screen serves three cases:
 *
 *  - a visitor (`signedIn` false): computes on the reference curves only, cannot
 *    save, and is told — once, above the form — what signing in would change;
 *  - a new plan (`planId` null): computes from the chosen file, and the first save
 *    creates it and moves the URL to `/race-plan/{id}` without a reload;
 *  - a saved plan: loads its inputs and computes straight away from the stored
 *    GPX, so a plan opens already drawn. Choosing a new file replaces the stored
 *    one on the next save.
 *
 * All the numbers — pacing, sections, legs — are computed server-side and arrive
 * as chart IR, drawn by the same `ChartView`/`TableView` as every analysis panel.
 */

import { useCallback, useEffect, useState } from "react";

import { ChartView } from "@/components/ChartView";
import { TableView } from "@/components/TableView";
import {
  deleteRacePlan,
  getRacePlan,
  getRacePlanOptions,
  planRace,
  saveRacePlan,
} from "@/lib/api";
import { formatHms, formatNumber, formatPace } from "@/lib/format";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type {
  PlotOutput,
  RacePlanCurveOption,
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

const SIGN_IN_HREF = "/api/auth/strava/start?next=/race-plan";

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
    return { target_time_s: target, aid_stations, start_time_s: start, curve };
  }, [targetTime, startTime, aidRows, curve, t]);

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
        setTargetTime(formatHms(p.target_time_s));
        setStartTime(p.start_time_s != null ? formatClock(p.start_time_s) : "");
        setAidRows(p.aid_stations.map((s) => ({ km: String(s.km), name: s.name })));
        setCurve(p.curve);
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
      const saved = await saveRacePlan(planId, title.trim(), params, file);
      if (!planId) {
        // Now a saved plan: give it its own URL without remounting the screen.
        window.history.replaceState(null, "", `/race-plan/${saved.id}`);
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
      window.location.href = "/race-plan";
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
        <a className="race-plan__back" href="/race-plan">
          {t("race_plan.back")}
        </a>
      )}
      <div className="page-header">
        <div className="page-header__title">
          <span className="page-header__icon" aria-hidden="true">🏁</span>
          <input
            className="page-header__name"
            aria-label={t("race_plan.plan_title")}
            placeholder={t("race_plan.title_placeholder")}
            value={title}
            maxLength={200}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>
        {signedIn && (
          <div className="page-header__actions">
            {savedAt != null && <span className="muted">{t("race_plan.saved")}</span>}
            <button type="button" className="button" onClick={save} disabled={saving}>
              {saving ? t("race_plan.saving") : t("race_plan.save")}
            </button>
            {planId && (
              <button type="button" className="button button--danger" onClick={remove}>
                {t("race_plan.delete")}
              </button>
            )}
          </div>
        )}
      </div>
      <p className="page-description">{t("race_plan.intro")}</p>

      {!signedIn && (
        <div className="note race-plan__warning">
          <p>{t("race_plan.public_warning")}</p>
          <p>{t("race_plan.sign_in_to_save")}</p>
          <a className="button button--strava button--small" href={SIGN_IN_HREF}>
            {t("race_plan.sign_in")}
          </a>
        </div>
      )}

      <form className="panel race-plan__form" onSubmit={submit}>
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
              type="text"
              inputMode="numeric"
              placeholder="07:00"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
            />
            <span className="muted race-plan__help">{t("race_plan.start_time_help")}</span>
          </label>

          <label className="race-plan__field">
            <span>{t("race_plan.curve")}</span>
            <select value={selectedCurve} onChange={(e) => setCurve(e.target.value)}>
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
                className="race-plan__aid-km"
                aria-label={t("race_plan.aid_station_km")}
                placeholder={t("race_plan.aid_station_km")}
                value={row.km}
                onChange={(e) => updateRow(index, { km: e.target.value })}
              />
              <input
                type="text"
                className="race-plan__aid-name"
                aria-label={t("race_plan.aid_station_name")}
                placeholder={t("race_plan.aid_station_name")}
                value={row.name}
                maxLength={80}
                onChange={(e) => updateRow(index, { name: e.target.value })}
              />
              <button
                type="button"
                className="button button--ghost button--small"
                onClick={() => setAidRows((rows) => rows.filter((_, i) => i !== index))}
              >
                {t("race_plan.remove")}
              </button>
            </div>
          ))}
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={() => setAidRows((rows) => [...rows, { km: "", name: "" }])}
          >
            {t("race_plan.add_aid_station")}
          </button>
        </fieldset>

        <div className="race-plan__actions">
          <button type="submit" className="button" disabled={computing}>
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
        {error && <p className="note note--error">{error}</p>}
      </form>

      {result && <RacePlanResultView result={result} t={t} />}
    </main>
  );
}

function RacePlanResultView({ result, t }: { result: RacePlanResult; t: Translate }) {
  const s = result.summary;
  const tiles: [string, string][] = [
    [t("race_plan.summary.distance"), `${formatNumber(s.distance_m / 1000, 1)} km`],
    [
      t("race_plan.summary.elevation"),
      `+${formatNumber(s.elevation_gain_m, 0)} / −${formatNumber(s.elevation_loss_m, 0)} m`,
    ],
    [t("race_plan.summary.target"), formatHms(s.target_time_s)],
    [t("race_plan.summary.gap_pace"), formatPace(s.gap_pace_s_per_km)],
    [t("race_plan.summary.avg_pace"), formatPace(s.average_pace_s_per_km)],
    [t("race_plan.summary.curve"), result.curve_label],
  ];

  return (
    <div className="race-plan__result">
      <div className="tile-grid race-plan__summary scale-1">
        {tiles.map(([label, value]) => (
          <div className="tile tile--dot" key={label}>
            <span className="tile__label">{label}</span>
            <span className="tile__value race-plan__tile-value">{value}</span>
          </div>
        ))}
      </div>
      {result.notes.map((note, index) => (
        <p key={index} className="note">
          {note}
        </p>
      ))}

      <OutputSection title={t("race_plan.section.profile")} output={result.outputs.profile} />
      <OutputSection title={t("race_plan.section.sections")} output={result.outputs.sections} />
      <OutputSection
        title={t("race_plan.section.aid_stations")}
        output={result.outputs.aid_stations}
      />
    </div>
  );
}

function OutputSection({ title, output }: { title: string; output: PlotOutput }) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h2 className="panel__title">{title}</h2>
      </div>
      {output.charts.map((chart, index) => (
        <ChartView key={index} chart={chart} />
      ))}
      {output.tables.map((table, index) => (
        <TableView key={index} table={table} />
      ))}
      {output.notes.map((note, index) => (
        <p key={index} className="note">
          {note}
        </p>
      ))}
    </section>
  );
}
