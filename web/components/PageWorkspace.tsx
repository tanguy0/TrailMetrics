"use client";

/**
 * One analysis: load it, render it, edit it, save it.
 *
 * Every analysis goes through here, including the three every athlete starts with.
 * There is no read-only mode: a default analysis is a stored page like any other and
 * is edited in place. It differs in exactly one way — no Delete, because it ships with
 * the app — and "Duplicate" is how you get a version you can remove or diverge.
 *
 * Autosave is debounced and driven by the serialized spec, so it fires on real
 * content changes rather than on every keystroke's re-render.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { PanelEditor, newId } from "./PanelEditor";
import { ProgressBar } from "./ProgressBar";
import {
  ApiError,
  deletePage,
  duplicatePage,
  getAthlete,
  getPage,
  getPrecomputeStatus,
  getRegistry,
  listActivities,
  savePage,
} from "@/lib/api";
import { RUNNING_SPORT_TYPES } from "@/lib/sport";
import { translator, type Strings } from "@/lib/strings";
import type {
  ActivitySummary,
  Athlete,
  PageSpec,
  PanelSpec,
  PrecomputeStatus,
  Registry,
} from "@/lib/types";

const AUTOSAVE_MS = 1200;
const PRECOMPUTE_POLL_MS = 3000;

type SaveState = "idle" | "saving" | "saved" | "error";

export function PageWorkspace({
  pageId,
  strings,
}: {
  pageId: string;
  strings: Strings;
}) {
  const t = translator(strings);
  const router = useRouter();

  const [spec, setSpec] = useState<PageSpec | null>(null);
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [activities, setActivities] = useState<ActivitySummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  // Incremented by "Recompute"; every panel re-renders ignoring what was cached.
  const [refreshToken, setRefreshToken] = useState(0);

  // --- Load ---------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [loadedRegistry, loadedAthlete, page] = await Promise.all([
          getRegistry(),
          getAthlete(),
          getPage(pageId),
        ]);
        if (cancelled) return;
        setRegistry(loadedRegistry);
        setAthlete(loadedAthlete);
        setSpec(page);

        // Only needed by the hand-picked-activities mode, but cheap: summaries never
        // touch per-second data.
        const { activities: list } = await listActivities();
        if (!cancelled) setActivities(list);
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && error.isUnauthorized) {
          router.push("/");
          return;
        }
        setLoadError((error as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pageId, router]);

  // --- Autosave -----------------------------------------------------------

  const signature = useMemo(() => (spec ? JSON.stringify(spec) : ""), [spec]);
  // What is already stored, so the first render after loading doesn't save.
  const savedSignature = useRef<string | null>(null);

  useEffect(() => {
    if (!spec) return;
    if (savedSignature.current === null) {
      savedSignature.current = signature;
      return;
    }
    if (savedSignature.current === signature) return;

    const timer = setTimeout(() => {
      setSaveState("saving");
      savePage(spec.id, spec)
        .then(() => {
          savedSignature.current = signature;
          setSaveState("saved");
          setSaveError(null);
        })
        .catch((error: Error) => {
          setSaveState("error");
          setSaveError(error.message);
        });
    }, AUTOSAVE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  // --- Panel operations ---------------------------------------------------

  const updatePanel = useCallback(
    (panelId: string, panel: PanelSpec) =>
      setSpec((current) =>
        current
          ? { ...current, panels: current.panels.map((p) => (p.id === panelId ? panel : p)) }
          : current,
      ),
    [],
  );

  const addPanel = () =>
    setSpec((current) => {
      if (!current) return current;
      // A new panel copies the last panel's source, which is nearly always what is
      // wanted — the alternative is an empty panel the user has to configure twice.
      const template = current.panels[current.panels.length - 1];
      return {
        ...current,
        panels: [
          ...current.panels,
          {
            id: newId("panel"),
            title: "New panel",
            description: "",
            source: template
              ? JSON.parse(JSON.stringify(template.source))
              : blankSource(athlete),
            plots: [],
            columns: 1,
          },
        ],
      };
    });

  const movePanel = (index: number, direction: -1 | 1) =>
    setSpec((current) => {
      if (!current) return current;
      const target = index + direction;
      if (target < 0 || target >= current.panels.length) return current;
      const panels = [...current.panels];
      [panels[index], panels[target]] = [panels[target], panels[index]];
      return { ...current, panels };
    });

  const removePanel = (panelId: string) =>
    setSpec((current) =>
      current ? { ...current, panels: current.panels.filter((p) => p.id !== panelId) } : current,
    );

  // --- Page actions -------------------------------------------------------

  const duplicate = async () => {
    if (!spec) return;
    const copy = await duplicatePage(spec.id, `${spec.name} (copy)`);
    router.push(`/pages/${copy.id}`);
  };

  // Absent for a default analysis; the server refuses those too.
  const remove = async () => {
    if (!spec || spec.builtin_key) return;
    if (!window.confirm(`Delete “${spec.name}”? This cannot be undone.`)) return;
    await deletePage(spec.id);
    router.push("/pages");
  };

  // --- Render -------------------------------------------------------------

  if (loadError) {
    return (
      <main className="container">
        <p className="note note--error">Could not load this page: {loadError}</p>
      </main>
    );
  }
  if (!spec || !registry || !athlete) {
    return (
      <main className="container">
        <p className="muted">Loading…</p>
      </main>
    );
  }

  return (
    <main className="container">
      <header className="page-header">
        <div className="page-header__title">
          <span className="page-header__icon">{spec.icon}</span>
          <input
            className="page-header__name"
            value={spec.name}
            onChange={(event) => setSpec({ ...spec, name: event.target.value })}
            aria-label="Analysis name"
          />
        </div>

        <div className="page-header__actions">
          <SaveBadge state={saveState} error={saveError} />
          <button
            type="button"
            className="button button--ghost"
            onClick={() => setRefreshToken((token) => token + 1)}
            title="Ignore what was computed before and fit everything again."
          >
            ↻ {t("page.recompute")}
          </button>
          <button type="button" className="button button--ghost" onClick={duplicate}>
            {t("page.duplicate")}
          </button>
          {/* A default analysis ships with the app, so there is nothing to delete it
              back to. Duplicating gives a copy that *can* be removed. */}
          {spec.builtin_key ? (
            <span className="tag" title={t("page.default_help")}>
              {t("page.default")}
            </span>
          ) : (
            <button type="button" className="button button--danger" onClick={remove}>
              {t("page.delete")}
            </button>
          )}
        </div>
      </header>

      {spec.description && <p className="page-description">{spec.description}</p>}

      <PrecomputeNotice spec={spec} registry={registry} strings={strings} />

      {athlete.weight_kg == null && (
        <p className="note">
          Set your weight on the <a href="/home">Home screen</a> to unlock the power
          and power-to-heart-rate metrics.
        </p>
      )}

      {spec.panels.map((panel, index) => (
        <PanelEditor
          key={panel.id}
          panel={panel}
          onChange={(next) => updatePanel(panel.id, next)}
          onMove={(direction) => movePanel(index, direction)}
          onRemove={() => removePanel(panel.id)}
          registry={registry}
          activities={activities}
          oldest={athlete.oldest_activity}
          newest={athlete.newest_activity}
          editable
          refreshToken={refreshToken}
          accentIndex={index}
          accentCount={spec.panels.length}
        />
      ))}

      <button
        type="button"
        className="button button--wide"
        onClick={addPanel}
        aria-label={t("page.add_panel")}
        title={t("page.add_panel")}
      >
        +
      </button>
    </main>
  );
}

/**
 * Progress of the background model fits, shown on a page that has some.
 *
 * This lives here rather than on Home because it is only actionable next to the
 * curves it is producing: "fitting your GAP models" above a volume chart is noise,
 * and above an empty GAP panel it is the explanation for why the panel is empty.
 *
 * Which pages qualify is read from the registry (`cost === "expensive"`), not from a
 * hard-coded page key — so a new model-fitting plot type gets this for free, and a
 * page that has none never shows it.
 */
function PrecomputeNotice({
  spec,
  registry,
  strings,
}: {
  spec: PageSpec;
  registry: Registry;
  strings: Strings;
}) {
  const t = translator(strings);
  const [status, setStatus] = useState<PrecomputeStatus | null>(null);

  const hasExpensivePlot = useMemo(() => {
    const expensive = new Set(
      registry.plots.filter((d) => d.cost === "expensive").map((d) => d.key),
    );
    return spec.panels.some((panel) =>
      panel.plots.some((plot) => expensive.has(plot.plot_type)),
    );
  }, [spec, registry.plots]);

  // Poll while it runs. Kept keyed on the *status* rather than on a timer that always
  // runs, so a page whose fits are already cached makes exactly one request.
  const running = status?.status === "running";
  useEffect(() => {
    if (!hasExpensivePlot) return;
    let live = true;
    const read = () =>
      getPrecomputeStatus()
        .then((next) => live && setStatus(next))
        .catch(() => undefined);
    read();
    if (!running) return () => { live = false; };
    const timer = setInterval(read, PRECOMPUTE_POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [hasExpensivePlot, running]);

  if (!hasExpensivePlot || !status) return null;

  if (status.status === "error") {
    return (
      <p className="note note--error">
        {t("precompute.failed")}: {status.message}
      </p>
    );
  }
  if (status.status !== "running") return null;

  return (
    <div className="page-precompute">
      <ProgressBar
        tone="sunrise"
        value={status.done}
        total={status.total}
        label={t("precompute.running")}
        // The server's message names the panel being fitted, which is the only thing
        // that changes during the minutes the counter does not. The static
        // explanation is the fallback, not the headline.
        detail={status.message || t("precompute.help")}
      />
      <p className="muted">{t("precompute.help")}</p>
    </div>
  );
}

function SaveBadge({ state, error }: { state: SaveState; error: string | null }) {
  if (state === "saving") return <span className="muted">Saving…</span>;
  if (state === "saved") return <span className="muted">Saved</span>;
  if (state === "error") {
    return <span className="note note--error">Not saved: {error}</span>;
  }
  return null;
}

function blankSource(athlete: Athlete | null): PanelSpec["source"] {
  const today = new Date().toISOString().slice(0, 10);
  return {
    mode: "window",
    activity_ids: [],
    selection_label: "",
    windows: [
      {
        name: "All history",
        start: (athlete?.oldest_activity ?? today).slice(0, 10),
        end: (athlete?.newest_activity ?? today).slice(0, 10),
      },
    ],
    // Explicit, not empty: a brand-new panel should read as "Running, every
    // sub-sport" in the sport picker rather than lean on the implicit
    // empty-means-everything fallback (see DataSourceEditor.tsx).
    filters: {
      sport_types: [...RUNNING_SPORT_TYPES],
      min_distance_km: null,
      max_distance_km: null,
    },
  };
}
