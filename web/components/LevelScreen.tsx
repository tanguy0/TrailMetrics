"use client";

/**
 * Tools → Level Assessment (design/specs/level.md § Page `/tools/level`).
 *
 * A compact hero, three tests behind a `tm-segment` (half-Cooper, critical speed,
 * records), a short form, and the result as one card: the VMA as the headline
 * tile, its pace, the VDOT, then the five pace zones exactly as Home's Zones card
 * lists them — and the heart-rate zones when HRmax is given. Every number is
 * computed server-side (`/tools/level/estimate`), so this screen and Home can
 * never disagree.
 *
 * Estimating never saves. With Strava connected, a "save updated estimate"
 * button at the bottom makes the shown result the athlete's level — their Home
 * zones from then on; the latest saved one is shown on arrival. A visitor gets
 * one callout instead: an account (with Strava) keeps it.
 */

import { useEffect, useState, type FormEvent } from "react";

import { Callout } from "@/components/Callout";
import { Kpi } from "@/components/Kpi";
import { ApiError, estimateLevel, getLatestLevel, saveLevel } from "@/lib/api";
import { registerHref } from "@/lib/auth";
import { formatNumber, formatPaceInput, formatPaceRange } from "@/lib/format";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type { LevelMethod, LevelResult } from "@/lib/types";

const METHODS: LevelMethod[] = ["half_cooper", "critical_speed", "records"];
const SUGGESTED_DISTANCES = [1000, 1609, 3000, 5000, 10000, 21097, 42195];

interface RecordRow {
  distance: string;
  time: string;
}

/** `h:mm:ss`, `mm:ss` or plain minutes → seconds; null when unreadable. */
function parseDuration(text: string): number | null {
  const parts = text.trim().split(":").map((part) => part.trim());
  if (parts.some((part) => part === "" || !/^\d+$/.test(part))) return null;
  const numbers = parts.map(Number);
  if (numbers.length === 1) return numbers[0] * 60;
  if (numbers.length === 2) return numbers[0] * 60 + numbers[1];
  if (numbers.length === 3) return numbers[0] * 3600 + numbers[1] * 60 + numbers[2];
  return null;
}

type EstimateBody = Parameters<typeof estimateLevel>[0];

export function LevelScreen({
  strings,
  signedIn,
  hasStrava,
}: {
  strings: Strings;
  signedIn: boolean;
  hasStrava: boolean;
}) {
  const t = translator(strings);
  const [method, setMethod] = useState<LevelMethod>("records");
  const [distance6, setDistance6] = useState("");
  const [d3, setD3] = useState("");
  const [d12, setD12] = useState("");
  const [records, setRecords] = useState<RecordRow[]>([{ distance: "10000", time: "" }]);
  const [hrMax, setHrMax] = useState("");
  const [result, setResult] = useState<LevelResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The request behind the shown result, so saving stores exactly what is shown.
  const [estimated, setEstimated] = useState<EstimateBody | null>(null);
  const [saving, setSaving] = useState(false);

  // An account arrives on its latest estimate, the one its Home zones use.
  useEffect(() => {
    if (!signedIn) return;
    getLatestLevel()
      .then(({ estimate }) => {
        if (!estimate) return;
        setMethod(estimate.result.method);
        setResult({ ...estimate.result, notes: [], saved_at: estimate.created_at });
      })
      .catch(() => undefined);
  }, [signedIn]);

  function inputs(): Record<string, unknown> | null {
    if (method === "half_cooper") return { distance_m: Number(distance6) };
    if (method === "critical_speed") return { d3_m: Number(d3), d12_m: Number(d12) };
    const rows = [];
    for (const row of records) {
      const seconds = parseDuration(row.time);
      if (seconds == null || !row.distance) return null;
      rows.push({ distance_m: Number(row.distance), seconds });
    }
    return { records: rows };
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const body = inputs();
    if (!body) {
      setError(t("level.error.invalid"));
      return;
    }
    setBusy(true);
    setError(null);
    const request: EstimateBody = { method, inputs: body, hr_max: hrMax ? Number(hrMax) : null };
    try {
      setResult(await estimateLevel(request));
      setEstimated(request);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : t("auth.error.generic"));
    }
    setBusy(false);
  }

  async function save() {
    if (!estimated) return;
    setSaving(true);
    setError(null);
    try {
      setResult(await saveLevel(estimated));
      setEstimated(null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : t("auth.error.generic"));
    }
    setSaving(false);
  }

  return (
    <main className="container level">
      <header className="tm-hero level-hero">
        <div className="tm-hero__body">
          <span className="tm-hero__kicker">{t("level.hero.kicker")}</span>
          <h1 className="tm-hero__title">{t("level.hero.title")}</h1>
          <span className="tm-hero__meta">{t("level.hero.lede")}</span>
        </div>
      </header>

      <form className="card-block level-form" onSubmit={submit} noValidate>
        <div className="tm-segment" role="tablist">
          {METHODS.map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              className="tm-segment__item"
              aria-selected={method === key}
              onClick={() => {
                setMethod(key);
                setError(null);
              }}
            >
              {t(`level.method.${key}`)}
            </button>
          ))}
        </div>
        <p className="body-sm muted">{t(`level.help.${method}`)}</p>

        {error && <Callout tone="terra">{error}</Callout>}

        <div className="level-form__fields">
          {method === "half_cooper" && (
            <NumberField label={t("level.field.distance_6")} value={distance6} onChange={setDistance6} />
          )}
          {method === "critical_speed" && (
            <>
              <NumberField label={t("level.field.d3")} value={d3} onChange={setD3} />
              <NumberField label={t("level.field.d12")} value={d12} onChange={setD12} />
            </>
          )}
          {method === "records" && (
            <RecordRows rows={records} onChange={setRecords} t={t} />
          )}
          <NumberField label={t("level.field.hr_max")} value={hrMax} onChange={setHrMax} />
        </div>

        <div>
          <button type="submit" className="tm-btn" disabled={busy}>
            {t("level.submit")}
          </button>
        </div>
      </form>

      {result && <LevelResultCard result={result} signedIn={signedIn} t={t} />}

      {hasStrava && result && estimated && !result.saved_at && (
        <div className="level-save">
          <button type="button" className="tm-btn" onClick={save} disabled={saving}>
            {saving ? t("level.saving") : t("level.save")}
          </button>
        </div>
      )}
    </main>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="tm-field">
      {label}
      <input
        className="tm-input"
        inputMode="numeric"
        value={value}
        onChange={(event) => onChange(event.target.value.replace(/[^\d]/g, ""))}
      />
    </label>
  );
}

function RecordRows({
  rows,
  onChange,
  t,
}: {
  rows: RecordRow[];
  onChange: (rows: RecordRow[]) => void;
  t: Translate;
}) {
  const update = (index: number, patch: Partial<RecordRow>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  return (
    <div className="level-records">
      <datalist id="level-distances">
        {SUGGESTED_DISTANCES.map((d) => <option key={d} value={d} />)}
      </datalist>
      {rows.map((row, index) => (
        <div className="level-records__row" key={index}>
          <label className="tm-field">
            {t("level.field.record_distance")}
            <input
              className="tm-input"
              inputMode="numeric"
              list="level-distances"
              value={row.distance}
              onChange={(event) => update(index, { distance: event.target.value.replace(/[^\d]/g, "") })}
            />
          </label>
          <label className="tm-field">
            {t("level.field.record_time")}
            <input
              className="tm-input"
              placeholder="0:42:30"
              value={row.time}
              onChange={(event) => update(index, { time: event.target.value })}
            />
          </label>
          {rows.length > 1 && (
            <button
              type="button"
              className="tm-btn tm-btn--ghost tm-btn--sm"
              onClick={() => onChange(rows.filter((_, i) => i !== index))}
            >
              {t("level.remove_record")}
            </button>
          )}
        </div>
      ))}
      <div>
        <button
          type="button"
          className="tm-btn tm-btn--secondary tm-btn--sm"
          onClick={() => onChange([...rows, { distance: "", time: "" }])}
        >
          {t("level.add_record")}
        </button>
      </div>
    </div>
  );
}

function LevelResultCard({
  result,
  signedIn,
  t,
}: {
  result: LevelResult;
  signedIn: boolean;
  t: Translate;
}) {
  const vma = `${formatNumber(result.vma_kmh, 1)} km/h`;
  const endurance = result.zones.find((zone) => zone.key === "endurance");
  const [before, after] = t("level.result.sentence", {
    vma: "\u0000",
    endurance: endurance ? formatPaceRange(endurance.fast_s_per_km, endurance.slow_s_per_km) : "—",
  }).split("\u0000");

  return (
    <section className="card-block level-result">
      <div className="data-block__heading">
        <h2 className="tm-section section-title">
          <span className="section-title__text">{t("level.result.title")}</span>
        </h2>
        <span className="tm-chip">{t(`level.confidence.${result.confidence}`)}</span>
      </div>

      {/* The one highlighted phrase of the card (Highlights.md). */}
      <p className="body level-result__sentence">
        {before}
        <span className="tm-hl">{vma}</span>
        {after}
      </p>

      <div className="kpi-grid">
        <Kpi label={t("level.result.vma")} value={formatNumber(result.vma_kmh, 1)} unit="km/h" tone="forest" large />
        <Kpi label={t("level.result.vma_pace")} value={formatPaceInput(result.vma_pace_s_per_km)} unit="/km" />
        <Kpi label={t("level.result.vdot")} value={formatNumber(result.vdot, 1)} />
        {result.extras.critical_pace_s_per_km != null && (
          <Kpi
            label={t("level.result.critical_pace")}
            value={formatPaceInput(result.extras.critical_pace_s_per_km)}
            unit="/km"
          />
        )}
        {result.extras.d_prime_m != null && (
          <Kpi label={t("level.result.d_prime")} value={formatNumber(result.extras.d_prime_m, 0)} unit="m" />
        )}
      </div>

      <h3 className="card-block__subtitle">{t("level.result.zones")}</h3>
      <div className="kpi-grid">
        {result.zones.map((zone) => (
          <Kpi
            key={zone.key}
            label={`${t(`home.zones.pace_${zone.key}`)} (${t("common.per_km")})`}
            value={formatPaceRange(zone.fast_s_per_km, zone.slow_s_per_km)}
            note={`${zone.low_pct}–${zone.high_pct} % VMA`}
          />
        ))}
      </div>

      {result.hr_zones.length > 0 && (
        <>
          <h3 className="card-block__subtitle">{t("level.result.hr_zones")}</h3>
          <div className="kpi-grid kpi-grid--two">
            {result.hr_zones.map((zone) => (
              <Kpi key={zone.key} label={t(`home.zones.${zone.key}`)} value={String(zone.bpm)} unit="bpm" />
            ))}
          </div>
        </>
      )}

      {result.notes.map((note) => <Callout key={note}>{note}</Callout>)}

      {signedIn ? (
        result.saved_at && <p className="body-sm muted">{t("tools.saved_zones")}</p>
      ) : (
        <Callout tone="forest">
          {t("tools.keep_zones")}{" "}
          <a href={registerHref("/tools/level")}>{t("visitor.register")}</a>
        </Callout>
      )}
    </section>
  );
}
