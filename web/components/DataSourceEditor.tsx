"use client";

/**
 * Editor for a panel's data source — *which activities* the panel is about.
 *
 * The three modes are the app's core idea: a hand-picked set, one time window, or
 * several named windows compared side by side. "Several windows" is what used to be
 * a bespoke season editor on one page and a bespoke time-scale editor on another;
 * here it is one control that every plot understands.
 */

import { useMemo, useState } from "react";

import { formatDistanceKm } from "@/lib/format";
import type {
  ActivitySummary,
  DataSourceSpec,
  SourceMode,
  TimeWindow,
} from "@/lib/types";

const MODES: { value: SourceMode; label: string; hint: string }[] = [
  {
    value: "activities",
    label: "Chosen activities",
    hint: "Pick specific runs — for comparing races.",
  },
  {
    value: "window",
    label: "One time window",
    hint: "Every activity between two dates.",
  },
  {
    value: "windows",
    label: "Several time windows",
    hint: "Named periods compared side by side — seasons, training blocks.",
  },
];

interface Props {
  source: DataSourceSpec;
  onChange: (source: DataSourceSpec) => void;
  activities: ActivitySummary[];
  sportTypes: string[];
  oldest: string | null;
  newest: string | null;
  /** Stable per editor instance, so each panel's radios form their own group. */
  groupName: string;
}

export function DataSourceEditor({
  source,
  onChange,
  activities,
  sportTypes,
  oldest,
  newest,
  groupName,
}: Props) {
  const patch = (changes: Partial<DataSourceSpec>) => onChange({ ...source, ...changes });

  return (
    <div className="source-editor">
      <div className="source-editor__modes">
        {MODES.map((mode) => (
          <label
            key={mode.value}
            className={`mode ${source.mode === mode.value ? "mode--active" : ""}`}
          >
            <input
              type="radio"
              name={`source-mode-${groupName}`}
              checked={source.mode === mode.value}
              onChange={() => patch({ mode: mode.value, ...defaultsFor(mode.value, source, oldest, newest) })}
            />
            <span className="mode__label">{mode.label}</span>
            <span className="mode__hint">{mode.hint}</span>
          </label>
        ))}
      </div>

      {source.mode === "activities" && (
        <ActivityPicker
          activities={activities}
          selected={source.activity_ids}
          onChange={(ids) => patch({ activity_ids: ids })}
        />
      )}

      {source.mode === "window" && (
        <WindowRow
          window={source.windows[0] ?? blankWindow("All history", oldest, newest)}
          onChange={(w) => patch({ windows: [w] })}
          showName={false}
        />
      )}

      {source.mode === "windows" && (
        <WindowList
          windows={source.windows}
          onChange={(windows) => patch({ windows })}
          oldest={oldest}
          newest={newest}
        />
      )}

      <Filters source={source} onChange={patch} sportTypes={sportTypes} />
    </div>
  );
}

/** Sensible starting values when switching mode, so nothing renders empty. */
function defaultsFor(
  mode: SourceMode,
  source: DataSourceSpec,
  oldest: string | null,
  newest: string | null,
): Partial<DataSourceSpec> {
  if (mode === "window" && source.windows.length !== 1) {
    return { windows: [blankWindow("All history", oldest, newest)] };
  }
  if (mode === "windows" && source.windows.length < 2) {
    return { windows: calendarYears(oldest, newest) };
  }
  return {};
}

function blankWindow(name: string, oldest: string | null, newest: string | null): TimeWindow {
  const today = new Date().toISOString().slice(0, 10);
  return {
    name,
    start: (oldest ?? `${new Date().getFullYear()}-01-01`).slice(0, 10),
    end: (newest ?? today).slice(0, 10),
  };
}

/** One window per calendar year covered by the athlete's history. */
function calendarYears(oldest: string | null, newest: string | null): TimeWindow[] {
  const first = oldest ? new Date(oldest).getUTCFullYear() : new Date().getUTCFullYear();
  const last = newest ? new Date(newest).getUTCFullYear() : new Date().getUTCFullYear();
  const windows: TimeWindow[] = [];
  for (let year = first; year <= last; year += 1) {
    windows.push({ name: String(year), start: `${year}-01-01`, end: `${year}-12-31` });
  }
  return windows;
}

function WindowRow({
  window,
  onChange,
  showName,
  onRemove,
}: {
  window: TimeWindow;
  onChange: (w: TimeWindow) => void;
  showName: boolean;
  onRemove?: () => void;
}) {
  const invalid = window.start > window.end;
  return (
    <div className="window-row">
      {showName && (
        <input
          type="text"
          value={window.name}
          placeholder="Name (e.g. Marathon block)"
          onChange={(event) => onChange({ ...window, name: event.target.value })}
        />
      )}
      <input
        type="date"
        value={window.start}
        onChange={(event) => onChange({ ...window, start: event.target.value })}
      />
      <input
        type="date"
        value={window.end}
        onChange={(event) => onChange({ ...window, end: event.target.value })}
      />
      {onRemove && (
        <button type="button" className="button button--ghost button--small" onClick={onRemove}>
          Remove
        </button>
      )}
      {invalid && <span className="note note--error">Start is after end.</span>}
    </div>
  );
}

function WindowList({
  windows,
  onChange,
  oldest,
  newest,
}: {
  windows: TimeWindow[];
  onChange: (windows: TimeWindow[]) => void;
  oldest: string | null;
  newest: string | null;
}) {
  // Overlap is legal — an activity lands in both groups — but almost always a typo.
  const overlaps = useMemo(() => {
    const pairs: string[] = [];
    for (let i = 0; i < windows.length; i += 1) {
      for (let j = i + 1; j < windows.length; j += 1) {
        const a = windows[i];
        const b = windows[j];
        if (a.start <= b.end && b.start <= a.end) pairs.push(`${a.name} ⇄ ${b.name}`);
      }
    }
    return pairs;
  }, [windows]);

  return (
    <div className="window-list">
      {windows.map((window, index) => (
        <WindowRow
          key={index}
          window={window}
          showName
          onChange={(next) => onChange(windows.map((w, i) => (i === index ? next : w)))}
          onRemove={
            windows.length > 1 ? () => onChange(windows.filter((_, i) => i !== index)) : undefined
          }
        />
      ))}
      <div className="row-actions">
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => onChange([...windows, blankWindow(`Window ${windows.length + 1}`, oldest, newest)])}
        >
          Add window
        </button>
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => onChange(calendarYears(oldest, newest))}
        >
          Use calendar years
        </button>
      </div>
      {overlaps.length > 0 && (
        <p className="note">Windows overlap: {overlaps.join(", ")}. Activities count in both.</p>
      )}
    </div>
  );
}

function ActivityPicker({
  activities,
  selected,
  onChange,
}: {
  activities: ActivitySummary[];
  selected: number[];
  onChange: (ids: number[]) => void;
}) {
  const [query, setQuery] = useState("");

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matches = needle
      ? activities.filter((a) => a.label.toLowerCase().includes(needle))
      : activities;
    // Long histories make an unbounded list unusable; selected ones always show.
    return matches.slice(0, 200);
  }, [activities, query]);

  const toggle = (id: number) =>
    onChange(selected.includes(id) ? selected.filter((v) => v !== id) : [...selected, id]);

  const selectedSet = new Set(selected);
  const chosen = activities.filter((a) => selectedSet.has(a.activity_id));

  return (
    <div className="activity-picker">
      <input
        type="search"
        placeholder="Filter by date, sport or distance…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <p className="muted">
        {selected.length} selected
        {chosen.length > 0 && ` · ${formatDistanceKm(chosen.reduce((sum, a) => sum + a.distance_m, 0))} total`}
      </p>
      <div className="activity-list">
        {shown.map((activity) => (
          <label key={activity.activity_id} className="activity-item">
            <input
              type="checkbox"
              checked={selectedSet.has(activity.activity_id)}
              onChange={() => toggle(activity.activity_id)}
            />
            <span>{activity.label}</span>
            {!activity.has_streams && <span className="tag">summary only</span>}
          </label>
        ))}
        {!shown.length && <p className="muted">No activity matches that filter.</p>}
      </div>
      {activities.length > shown.length && (
        <p className="muted">
          Showing {shown.length} of {activities.length} — narrow the filter to see the rest.
        </p>
      )}
    </div>
  );
}

function Filters({
  source,
  onChange,
  sportTypes,
}: {
  source: DataSourceSpec;
  onChange: (changes: Partial<DataSourceSpec>) => void;
  sportTypes: string[];
}) {
  const filters = source.filters;
  const setFilters = (changes: Partial<DataSourceSpec["filters"]>) =>
    onChange({ filters: { ...filters, ...changes } });

  const toggleSport = (sport: string) =>
    setFilters({
      sport_types: filters.sport_types.includes(sport)
        ? filters.sport_types.filter((s) => s !== sport)
        : [...filters.sport_types, sport],
    });

  return (
    <details className="filters">
      <summary>
        Filters
        {filters.sport_types.length > 0 && ` · ${filters.sport_types.join(", ")}`}
      </summary>
      <div className="filters__body">
        <div className="param">
          <span className="param__label">Sport types</span>
          <div className="multichoice">
            {sportTypes.map((sport) => (
              <label key={sport} className="multichoice__item">
                <input
                  type="checkbox"
                  checked={filters.sport_types.includes(sport)}
                  onChange={() => toggleSport(sport)}
                />
                <span>{sport}</span>
              </label>
            ))}
          </div>
          <p className="param__help">None checked means every sport.</p>
        </div>
        <div className="param">
          <label className="param__label">Min distance (km)</label>
          <input
            type="number"
            min={0}
            value={filters.min_distance_km ?? ""}
            onChange={(event) =>
              setFilters({
                min_distance_km: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </div>
        <div className="param">
          <label className="param__label">Max distance (km)</label>
          <input
            type="number"
            min={0}
            value={filters.max_distance_km ?? ""}
            onChange={(event) =>
              setFilters({
                max_distance_km: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </div>
      </div>
    </details>
  );
}
