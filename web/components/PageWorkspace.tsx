"use client";

/**
 * A whole page: load it, render it, edit it, save it.
 *
 * Used for both a user's own pages and the built-in examples. Examples render
 * through exactly the same component in read-only mode — if this workspace can't
 * express one, the example is lying about what the builder can do. The only
 * difference is that an example offers "Duplicate" instead of saving in place.
 *
 * Autosave is debounced and driven by the serialized spec, so it fires on real
 * content changes rather than on every keystroke's re-render.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { PanelEditor, newId } from "./PanelEditor";
import {
  ApiError,
  deletePage,
  duplicateBuiltin,
  duplicatePage,
  getAthlete,
  getBuiltinPage,
  getPage,
  getRegistry,
  listActivities,
  savePage,
} from "@/lib/api";
import type {
  ActivitySummary,
  Athlete,
  PageSpec,
  PanelSpec,
  Registry,
} from "@/lib/types";

const AUTOSAVE_MS = 1200;

type SaveState = "idle" | "saving" | "saved" | "error";

export function PageWorkspace({
  pageId,
  builtinKey,
}: {
  pageId?: string;
  builtinKey?: string;
}) {
  const router = useRouter();
  const readOnly = Boolean(builtinKey);

  const [spec, setSpec] = useState<PageSpec | null>(null);
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [activities, setActivities] = useState<ActivitySummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);

  // --- Load ---------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [loadedRegistry, loadedAthlete, page] = await Promise.all([
          getRegistry(),
          getAthlete(),
          builtinKey ? getBuiltinPage(builtinKey) : getPage(pageId!),
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
  }, [builtinKey, pageId, router]);

  // --- Autosave -----------------------------------------------------------

  const signature = useMemo(() => (spec ? JSON.stringify(spec) : ""), [spec]);
  // What is already stored, so the first render after loading doesn't save.
  const savedSignature = useRef<string | null>(null);

  useEffect(() => {
    if (!spec || readOnly) return;
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
  }, [signature, readOnly]);

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
            collapsed: false,
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
    const copy = builtinKey
      ? await duplicateBuiltin(builtinKey, `${spec.name} (mine)`)
      : await duplicatePage(spec.id, `${spec.name} (copy)`);
    router.push(`/pages/${copy.id}`);
  };

  const remove = async () => {
    if (!spec || readOnly) return;
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
          {readOnly ? (
            <h1>{spec.name}</h1>
          ) : (
            <input
              className="page-header__name"
              value={spec.name}
              onChange={(event) => setSpec({ ...spec, name: event.target.value })}
              aria-label="Page name"
            />
          )}
        </div>

        <div className="page-header__actions">
          {readOnly ? (
            <>
              <span className="tag">Example — read only</span>
              <button type="button" className="button" onClick={duplicate}>
                Duplicate to my pages
              </button>
            </>
          ) : (
            <>
              <SaveBadge state={saveState} error={saveError} />
              <button type="button" className="button button--ghost" onClick={duplicate}>
                Duplicate
              </button>
              <button type="button" className="button button--danger" onClick={remove}>
                Delete
              </button>
            </>
          )}
        </div>
      </header>

      {spec.description && <p className="page-description">{spec.description}</p>}

      {athlete.weight_kg == null && (
        <p className="note">
          Set your weight on the <a href="/pages">pages screen</a> to unlock the power
          and power-to-heart-rate metrics.
        </p>
      )}

      {spec.panels.map((panel, index) => (
        <PanelEditor
          key={panel.id}
          panel={panel}
          onChange={(next) => updatePanel(panel.id, next)}
          onMove={readOnly ? undefined : (direction) => movePanel(index, direction)}
          onRemove={readOnly ? undefined : () => removePanel(panel.id)}
          registry={registry}
          activities={activities}
          sportTypes={athlete.sport_types}
          oldest={athlete.oldest_activity}
          newest={athlete.newest_activity}
          editable={!readOnly}
        />
      ))}

      {!readOnly && (
        <button type="button" className="button button--wide" onClick={addPanel}>
          Add a panel
        </button>
      )}
    </main>
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
    filters: { sport_types: [], min_distance_km: null, max_distance_km: null },
  };
}
