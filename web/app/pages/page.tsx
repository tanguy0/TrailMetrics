"use client";

/**
 * The pages screen: your pages, the built-in examples, and the data you have.
 *
 * Also where the first Strava import is started and watched. That import is the one
 * genuinely slow operation in the product (Strava allows 100 requests per 15
 * minutes), so it runs in the background and reports progress here rather than
 * blocking a request.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  createPage,
  getAthlete,
  getSyncStatus,
  listBuiltinPages,
  listPages,
  setWeight,
  startSync,
} from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { Athlete, PageSummary } from "@/lib/types";

const POLL_MS = 2000;

export default function PagesScreen() {
  const router = useRouter();
  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [pages, setPages] = useState<PageSummary[]>([]);
  const [examples, setExamples] = useState<PageSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [me, mine, builtin] = await Promise.all([
        getAthlete(),
        listPages(),
        listBuiltinPages(),
      ]);
      setAthlete(me);
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

  // Poll only while an import is actually running.
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

  const importActivities = async (force: boolean) => {
    setBusy(true);
    try {
      await startSync({ force });
      await load();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const newPage = async () => {
    const name = window.prompt("Name your page", "My analysis");
    if (!name) return;
    const page = await createPage(name);
    router.push(`/pages/${page.id}`);
  };

  if (error) {
    return (
      <main className="container">
        <p className="note note--error">{error}</p>
      </main>
    );
  }
  if (!athlete) {
    return (
      <main className="container">
        <p className="muted">Loading…</p>
      </main>
    );
  }

  return (
    <main className="container">
      <header className="page-header">
        <h1>{athlete.display_name}</h1>
        <div className="page-header__actions">
          <button type="button" className="button" onClick={newPage}>
            New page
          </button>
        </div>
      </header>

      <section className="stats">
        <Stat label="Activities" value={String(athlete.activity_count)} />
        <Stat label="Oldest" value={formatDate(athlete.oldest_activity)} />
        <Stat label="Newest" value={formatDate(athlete.newest_activity)} />
        <WeightStat athlete={athlete} onSaved={setAthlete} />
      </section>

      <section className="sync">
        {syncing ? (
          <>
            <p>
              Importing from Strava… {athlete.sync.done}
              {athlete.sync.total ? ` of ${athlete.sync.total}` : ""} — {athlete.sync.message}
            </p>
            <progress
              value={athlete.sync.done}
              max={Math.max(athlete.sync.total, 1)}
              className="progress"
            />
          </>
        ) : (
          <div className="sync__actions">
            <button
              type="button"
              className="button"
              onClick={() => importActivities(false)}
              disabled={busy}
            >
              {athlete.activity_count ? "Import new activities" : "Import my activities"}
            </button>
            {athlete.activity_count > 0 && (
              <button
                type="button"
                className="button button--ghost"
                onClick={() => importActivities(true)}
                disabled={busy}
                title="Re-fetch and recompute everything. Slow, and spends the Strava rate limit."
              >
                Re-import everything
              </button>
            )}
            {athlete.sync.status === "error" && (
              <span className="note note--error">Last import failed: {athlete.sync.message}</span>
            )}
            {athlete.sync.last_synced_at && (
              <span className="muted">
                Last import {formatDate(athlete.sync.last_synced_at)}
              </span>
            )}
          </div>
        )}
      </section>

      {athlete.activity_count === 0 && !syncing && (
        <p className="note">
          Import your activities to start building pages — everything below works off that
          data.
        </p>
      )}

      <h2>My pages</h2>
      {pages.length ? (
        <div className="card-grid">
          {pages.map((page) => (
            <PageCard key={page.id} page={page} href={`/pages/${page.id}`} />
          ))}
        </div>
      ) : (
        <p className="muted">
          None yet. Start from scratch with <em>New page</em>, or duplicate an example
          below.
        </p>
      )}

      <h2>Examples</h2>
      <p className="muted">
        Built from the same panels and plots you get. Open one to read it, or duplicate it
        to make it yours.
      </p>
      <div className="card-grid">
        {examples.map((page) => (
          <PageCard
            key={page.builtin_key}
            page={page}
            href={`/pages/builtin/${page.builtin_key}`}
          />
        ))}
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
    </div>
  );
}

/**
 * Weight unlocks the power metrics. Stored power is per-kilogram, so changing this
 * rescales the whole history immediately — nothing is recomputed.
 */
function WeightStat({
  athlete,
  onSaved,
}: {
  athlete: Athlete;
  onSaved: (athlete: Athlete) => void;
}) {
  const [value, setValue] = useState(athlete.weight_kg?.toString() ?? "");
  const [saving, setSaving] = useState(false);

  const commit = async () => {
    const parsed = value === "" ? null : Number(value);
    if (parsed !== null && (!Number.isFinite(parsed) || parsed < 25 || parsed > 250)) return;
    if (parsed === athlete.weight_kg) return;
    setSaving(true);
    try {
      onSaved(await setWeight(parsed));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="stat">
      <span className="stat__label">Weight (kg){saving ? " · saving" : ""}</span>
      <input
        className="stat__input"
        type="number"
        min={25}
        max={250}
        step={0.5}
        value={value}
        placeholder="—"
        onChange={(event) => setValue(event.target.value)}
        onBlur={commit}
      />
    </div>
  );
}

function PageCard({ page, href }: { page: PageSummary; href: string }) {
  return (
    <a className="card" href={href}>
      <span className="card__icon">{page.icon}</span>
      <span className="card__title">{page.name}</span>
      <span className="card__meta">
        {page.panel_count} panel{page.panel_count === 1 ? "" : "s"} · {page.plot_count} plot
        {page.plot_count === 1 ? "" : "s"}
      </span>
      {page.description && <span className="card__description">{page.description}</span>}
    </a>
  );
}
