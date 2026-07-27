"use client";

/**
 * One panel: its data source, its plots, and their live output.
 *
 * The panel is the unit the user actually thinks in — *one* data source, *many*
 * plots over it — so it is also the unit of re-rendering. Editing a plot's
 * parameters re-renders only this panel, debounced, with the previous result left
 * on screen so the page never flashes empty while a request is in flight.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { PlotOutputView } from "./PlotOutputView";
import { ParamForm } from "./ParamForm";
import { DataSourceEditor } from "./DataSourceEditor";
import { ApiError, renderPanel } from "@/lib/api";
import type {
  ActivitySummary,
  PanelResult,
  PanelSpec,
  PlotSpec,
  Registry,
} from "@/lib/types";

// How long to wait after the last edit before re-rendering.
const DEBOUNCE_MS = 400;

interface Props {
  panel: PanelSpec;
  onChange: (panel: PanelSpec) => void;
  onRemove?: () => void;
  onMove?: (direction: -1 | 1) => void;
  registry: Registry;
  activities: ActivitySummary[];
  sportTypes: string[];
  oldest: string | null;
  newest: string | null;
  editable: boolean;
  initialResult?: PanelResult;
}

export function PanelEditor({
  panel,
  onChange,
  onRemove,
  onMove,
  registry,
  activities,
  sportTypes,
  oldest,
  newest,
  editable,
  initialResult,
}: Props) {
  const [result, setResult] = useState<PanelResult | undefined>(initialResult);
  const [loading, setLoading] = useState(!initialResult);
  const [failure, setFailure] = useState<string | null>(null);
  const [forced, setForced] = useState<string[]>([]);
  const [showSource, setShowSource] = useState(false);

  // Serialized spec as the render trigger: structural identity changes on every
  // keystroke, but the *content* is what the server cares about.
  const signature = useMemo(
    () => JSON.stringify({ panel, forced }),
    [panel, forced],
  );

  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;
      setLoading(true);
      renderPanel(panel, forced, controller.signal)
        .then((response) => {
          setResult(response.panel);
          setFailure(null);
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          setFailure(
            error instanceof ApiError ? error.message : (error as Error).message,
          );
        })
        .finally(() => !controller.signal.aborted && setLoading(false));
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  const definitions = useMemo(
    () => new Map(registry.plots.map((d) => [d.key, d])),
    [registry.plots],
  );

  const updatePlot = useCallback(
    (plotId: string, changes: Partial<PlotSpec>) =>
      onChange({
        ...panel,
        plots: panel.plots.map((p) => (p.id === plotId ? { ...p, ...changes } : p)),
      }),
    [onChange, panel],
  );

  const addPlot = (plotType: string) => {
    const definition = definitions.get(plotType);
    if (!definition) return;
    const params: Record<string, unknown> = {};
    for (const spec of definition.params) params[spec.key] = spec.default;
    onChange({
      ...panel,
      plots: [
        ...panel.plots,
        { id: newId("plot"), plot_type: plotType, params, title: null, collapsed: false },
      ],
    });
  };

  const movePlot = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= panel.plots.length) return;
    const plots = [...panel.plots];
    [plots[index], plots[target]] = [plots[target], plots[index]];
    onChange({ ...panel, plots });
  };

  const resultsByPlot = new Map((result?.plots ?? []).map((p) => [p.plot_id, p]));

  return (
    <section className="panel">
      <header className="panel__header">
        {editable ? (
          <input
            className="panel__title-input"
            value={panel.title}
            placeholder="Panel title"
            onChange={(event) => onChange({ ...panel, title: event.target.value })}
          />
        ) : (
          <h2 className="panel__title">{panel.title || "Panel"}</h2>
        )}

        <div className="panel__meta">
          {result && !result.error && (
            <span className="muted">
              {result.activity_count} activities
              {result.groups.length > 1 &&
                ` · ${result.groups.map((g) => `${g.label} (${g.size})`).join(", ")}`}
            </span>
          )}
          {loading && <span className="spinner" aria-label="Rendering" />}
        </div>

        {editable && (
          <div className="panel__actions">
            <button
              type="button"
              className="button button--ghost button--small"
              onClick={() => setShowSource((v) => !v)}
            >
              {showSource ? "Hide data source" : "Data source"}
            </button>
            <select
              className="select--small"
              value={panel.columns}
              onChange={(event) => onChange({ ...panel, columns: Number(event.target.value) })}
              aria-label="Columns"
            >
              <option value={1}>1 column</option>
              <option value={2}>2 columns</option>
            </select>
            {onMove && (
              <>
                <button
                  type="button"
                  className="button button--ghost button--small"
                  onClick={() => onMove(-1)}
                  aria-label="Move panel up"
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="button button--ghost button--small"
                  onClick={() => onMove(1)}
                  aria-label="Move panel down"
                >
                  ↓
                </button>
              </>
            )}
            {onRemove && (
              <button type="button" className="button button--danger button--small" onClick={onRemove}>
                Delete panel
              </button>
            )}
          </div>
        )}
      </header>

      {panel.description && <p className="panel__description">{panel.description}</p>}

      {editable && showSource && (
        <DataSourceEditor
          source={panel.source}
          onChange={(source) => onChange({ ...panel, source })}
          activities={activities}
          sportTypes={sportTypes}
          oldest={oldest}
          newest={newest}
          groupName={panel.id}
        />
      )}

      {failure && <p className="note note--error">Could not render this panel: {failure}</p>}
      {result?.error && <p className="note note--error">{result.error}</p>}

      <div className={`plot-grid plot-grid--${panel.columns === 2 ? "two" : "one"}`}>
        {panel.plots.map((plot, index) => {
          const definition = definitions.get(plot.plot_type);
          const plotResult = resultsByPlot.get(plot.id);
          // Tables read badly in a narrow column, so they always span the grid.
          const spans = !plotResult || plotResult.output.tables.length > 0;
          return (
            <article
              key={plot.id}
              className={`plot-card ${spans ? "plot-card--wide" : ""}`}
            >
              <header className="plot-card__header">
                <h3>{plot.title || definition?.label || plot.plot_type}</h3>
                {editable && (
                  <div className="plot-card__actions">
                    <button
                      type="button"
                      className="button button--ghost button--small"
                      onClick={() => updatePlot(plot.id, { collapsed: !plot.collapsed })}
                    >
                      {plot.collapsed ? "Settings" : "Hide settings"}
                    </button>
                    <button
                      type="button"
                      className="button button--ghost button--small"
                      onClick={() => movePlot(index, -1)}
                      aria-label="Move plot up"
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="button button--ghost button--small"
                      onClick={() => movePlot(index, 1)}
                      aria-label="Move plot down"
                    >
                      ↓
                    </button>
                    <button
                      type="button"
                      className="button button--danger button--small"
                      onClick={() =>
                        onChange({
                          ...panel,
                          plots: panel.plots.filter((p) => p.id !== plot.id),
                        })
                      }
                    >
                      Remove
                    </button>
                  </div>
                )}
              </header>

              {definition && <p className="muted">{definition.description}</p>}

              {editable && !plot.collapsed && definition && (
                <ParamForm
                  specs={definition.params}
                  values={plot.params}
                  onChange={(params) => updatePlot(plot.id, { params })}
                  providers={registry.providers}
                  metrics={registry.metrics}
                  idPrefix={plot.id}
                />
              )}

              {plotResult ? (
                <PlotOutputView
                  result={plotResult}
                  onCompute={(id) => setForced((ids) => [...new Set([...ids, id])])}
                />
              ) : (
                <p className="muted">Rendering…</p>
              )}
            </article>
          );
        })}
      </div>

      {editable && <PlotPicker registry={registry} onAdd={addPlot} />}
    </section>
  );
}

/** Grouped by category, so the catalogue stays navigable as it grows. */
function PlotPicker({
  registry,
  onAdd,
}: {
  registry: Registry;
  onAdd: (plotType: string) => void;
}) {
  const [choice, setChoice] = useState("");

  const grouped = useMemo(() => {
    const byCategory = new Map<string, typeof registry.plots>();
    for (const definition of registry.plots) {
      const list = byCategory.get(definition.category) ?? [];
      list.push(definition);
      byCategory.set(definition.category, list);
    }
    return [...byCategory.entries()];
  }, [registry.plots]);

  return (
    <div className="plot-picker">
      <select value={choice} onChange={(event) => setChoice(event.target.value)}>
        <option value="">Add a plot…</option>
        {grouped.map(([category, definitions]) => (
          <optgroup key={category} label={category}>
            {definitions.map((definition) => (
              <option key={definition.key} value={definition.key}>
                {definition.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <button
        type="button"
        className="button"
        disabled={!choice}
        onClick={() => {
          onAdd(choice);
          setChoice("");
        }}
      >
        Add
      </button>
    </div>
  );
}

/** Ids only have to be unique within a document; the server keeps its own. */
export function newId(prefix: string): string {
  const random = Math.random().toString(36).slice(2, 10);
  return `${prefix}_${random}`;
}
